# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple
import numpy as np
import math

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        print("Initialized self.self.actor_optimizer, module and tokenizer: ", str(actor_module), str(actor_optimizer), str(tokenizer))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                # Log inputs before forward pass
                print(f"\n🔍 [RMPAD] Inputs to model:")
                print(f"  input_ids_rmpad: shape={input_ids_rmpad.shape}, min={input_ids_rmpad.min().item()}, max={input_ids_rmpad.max().item()}, mean={input_ids_rmpad.float().mean().item():.2f}")
                if position_ids_rmpad is not None:
                    print(f"  position_ids_rmpad: shape={position_ids_rmpad.shape}, min={position_ids_rmpad.min().item()}, max={position_ids_rmpad.max().item()}, mean={position_ids_rmpad.float().mean().item():.2f}")

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                print(f"\n🔍 [RMPAD] Model output:")
                if hasattr(output, 'logits') and output.logits is not None:
                    print(f"  output.logits: shape={output.logits.shape}, min={output.logits.min().item():.4f}, max={output.logits.max().item():.4f}, mean={output.logits.mean().item():.4f}")
                if hasattr(output, 'log_probs') and output.log_probs is not None:
                    print(f"  output.log_probs: shape={output.log_probs.shape}, min={output.log_probs.min().item():.6f}, max={output.log_probs.max().item():.6f}, mean={output.log_probs.mean().item():.6f}")
                if hasattr(output, 'entropy') and output.entropy is not None:
                    print(f"  output.entropy: shape={output.entropy.shape}, min={output.entropy.min().item():.6f}, max={output.entropy.max().item():.6f}, mean={output.entropy.mean().item():.6f}")

                if self.use_fused_kernels:
                    print("Using fused kernels")
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    print(f"\n🔍 [RMPAD] logits_rmpad before temp scaling:")
                    print(f"  shape={logits_rmpad.shape}, min={logits_rmpad.min().item():.6f}, max={logits_rmpad.max().item():.6f}, mean={logits_rmpad.mean().item():.6f}")
                    
                    logits_rmpad.div_(temperature)
                    print(f"🔍 [RMPAD] logits_rmpad after temp scaling (temp={temperature:.6f}):")
                    print(f"  min={logits_rmpad.min().item():.6f}, max={logits_rmpad.max().item():.6f}, mean={logits_rmpad.mean().item():.6f}")
                    
                    print(f"🔍 [RMPAD] input_ids_rmpad_rolled (labels):")
                    print(f"  shape={input_ids_rmpad_rolled.shape}, min={input_ids_rmpad_rolled.min().item()}, max={input_ids_rmpad_rolled.max().item()}, mean={input_ids_rmpad_rolled.float().mean().item():.2f}")

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    print(f"🔍 [RMPAD] Calling logprobs_from_logits with inplace_backward={inplace_backward}")
                    
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    
                    print(f"🔍 [RMPAD] log_probs result:")
                    print(f"  shape={log_probs.shape}, min={log_probs.min().item():.6f}, max={log_probs.max().item():.6f}, mean={log_probs.mean().item():.6f}")

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        print(f"🔍 [RMPAD] entropy_rmpad: min={entropy_rmpad.min().item():.6f}, max={entropy_rmpad.max().item():.6f}, mean={entropy_rmpad.mean().item():.6f}")

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm) or grad_norm >= self.config.grad_norm_threshold:
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches =     data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        mb_input_ids_lst = []
        mb_position_ids_lst = []
        mb_attention_mask_lst = []
        mb_responses_lst = []
        
        for mb_idx, micro_batch in enumerate(micro_batches):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            
            # Log micro-batch size
            batch_size = micro_batch['input_ids'].shape[0]
            seq_len = micro_batch['input_ids'].shape[1]
            print(f"\n🔍 [MB{mb_idx}] Size: batch_size={batch_size}, seq_len={seq_len}")
            
            # Decode first micro-batch for inspection
            if mb_idx == 0 and self.tokenizer is not None and batch_size > 0:
                import os
                import json
                import time
                import torch.distributed as dist
                
                # Build output as single string to avoid async interleaving
                output_lines = []
                output_lines.append("="*100)
                output_lines.append(f"🔍 [MB0 DECODE] Decoding all {batch_size} samples in micro-batch 0:")
                output_lines.append("="*100)
                
                # Get worker/rank info
                rank = dist.get_rank() if dist.is_initialized() else 0
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                
                # Loop through all samples in first micro-batch
                for sample_idx in range(batch_size):
                    output_lines.append("")
                    output_lines.append("─"*100)
                    output_lines.append(f"SAMPLE {sample_idx}/{batch_size-1}")
                    output_lines.append("─"*100)
                    
                    sample_input_ids = micro_batch['input_ids'][sample_idx]
                    sample_responses = micro_batch['responses'][sample_idx]
                    sample_attn_mask = micro_batch['attention_mask'][sample_idx]
                    
                    # Find prompt vs response split
                    response_len = sample_responses.shape[0]
                    prompt_len = seq_len - response_len
                    
                    output_lines.append(f"📊 Lengths: total_seq={seq_len}, prompt={prompt_len}, response={response_len}")
                    output_lines.append(f"📊 Attention: total_valid={sample_attn_mask.sum().item()}, ratio={sample_attn_mask.float().mean().item():.4f}")
                    
                    # Extract prompt (everything before response)
                    prompt_ids = sample_input_ids[:prompt_len]
                    prompt_attn = sample_attn_mask[:prompt_len]
                    
                    # Decode prompt (only non-padding)
                    prompt_valid_mask = prompt_attn.bool()
                    num_prompt_attended = prompt_valid_mask.sum().item()
                    num_prompt_masked = (~prompt_valid_mask).sum().item()
                    
                    if prompt_valid_mask.any():
                        prompt_valid_ids = prompt_ids[prompt_valid_mask]
                        decoded_prompt = self.tokenizer.decode(prompt_valid_ids, skip_special_tokens=False)
                        output_lines.append("")
                        output_lines.append(f"📝 PROMPT ({num_prompt_attended} attended, {num_prompt_masked} masked/padding):")
                        output_lines.append(decoded_prompt)
                    else:
                        output_lines.append("")
                        output_lines.append(f"📝 PROMPT: (all padding)")
                    
                    # Decode response
                    response_ids_from_input = sample_input_ids[prompt_len:prompt_len+response_len]
                    response_attn = sample_attn_mask[prompt_len:prompt_len+response_len]
                    response_valid_mask = response_attn.bool()
                    num_response_attended = response_valid_mask.sum().item()
                    num_response_masked = (~response_valid_mask).sum().item()
                    
                    if response_valid_mask.any():
                        response_valid_ids = response_ids_from_input[response_valid_mask]
                        decoded_response = self.tokenizer.decode(response_valid_ids, skip_special_tokens=False)
                        output_lines.append("")
                        output_lines.append(f"💬 RESPONSE from input_ids ({num_response_attended} attended, {num_response_masked} masked/padding):")
                        output_lines.append(decoded_response)
                    else:
                        output_lines.append("")
                        output_lines.append(f"💬 RESPONSE from input_ids: (all padding)")
                    
                    # Decode response from separate response tensor (should be same)
                    response_valid_mask_2 = (sample_responses != self.tokenizer.pad_token_id)
                    if response_valid_mask_2.any():
                        response_valid_ids_2 = sample_responses[response_valid_mask_2]
                        decoded_response_2 = self.tokenizer.decode(response_valid_ids_2, skip_special_tokens=False)
                        output_lines.append("")
                        output_lines.append(f"💬 RESPONSE from responses tensor ({response_valid_mask_2.sum().item()} attended, {(~response_valid_mask_2).sum().item()} masked/padding):")
                    output_lines.append(decoded_response_2)
                    
                    # Show attention pattern summary
                    total_attended = sample_attn_mask.sum().item()
                    total_masked = seq_len - total_attended
                    output_lines.append("")
                    output_lines.append(f"🎭 ATTENTION SUMMARY:")
                    output_lines.append(f"  Total sequence: {seq_len} tokens")
                    output_lines.append(f"  Attended (attn=1): {total_attended} tokens ({100*total_attended/seq_len:.1f}%)")
                    output_lines.append(f"  Masked (attn=0):   {total_masked} tokens ({100*total_masked/seq_len:.1f}%)")
                    output_lines.append(f"  Breakdown:")
                    output_lines.append(f"    - Prompt: {num_prompt_attended} attended, {num_prompt_masked} masked")
                    output_lines.append(f"    - Response: {num_response_attended} attended, {num_response_masked} masked")
                    
                    # Show first/last attended positions
                    attended_positions = sample_attn_mask.nonzero(as_tuple=True)[0]
                    if len(attended_positions) > 0:
                        first_attended = attended_positions[0].item()
                        last_attended = attended_positions[-1].item()
                        output_lines.append(f"  First attended position: {first_attended}")
                        output_lines.append(f"  Last attended position: {last_attended}")
                        output_lines.append(f"  Attended span: [{first_attended}:{last_attended}] ({last_attended - first_attended + 1} positions)")
                
                output_lines.append("")
                output_lines.append("="*100)
                
                # Write to JSON file
                # output_text = "\n".join(output_lines)
                # output_data = {
                #     "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                #     "rank": rank,
                #     "world_size": world_size,
                #     "batch_size": batch_size,
                #     "seq_len": seq_len,
                #     "output": output_text
                # }
                
                # Save to file
                # if hasattr(data.meta_info, 'get') and 'global_step' in data.meta_info:
                #     step = data.meta_info['global_step']
                # else:
                #     step = "unknown"
                
                # output_dir = "/tmp/mb0_decode_logs"
                # os.makedirs(output_dir, exist_ok=True)
                # output_file = os.path.join(output_dir, f"mb0_decode_rank{rank}_step{step}_{time.strftime('%Y%m%d_%H%M%S')}.json")
                
                # with open(output_file, 'w') as f:
                #     json.dump(output_data, f, indent=2)
                
                # Print as single block to avoid interleaving
                # print(f"\n💾 Saved MB0 decode to: {output_file}")
                # print(output_text)
            
            # Collect micro-batch input stats
            mb_input_ids_lst.append(micro_batch['input_ids'])
            mb_position_ids_lst.append(micro_batch['position_ids'])
            mb_attention_mask_lst.append(micro_batch['attention_mask'])
            mb_responses_lst.append(micro_batch['responses'])
            
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        # Micro-batch level stats (as arrays)
        print(f"\n📊 [MB_MIN] input_ids: {[mb.min().item() for mb in mb_input_ids_lst]}")
        print(f"📊 [MB_MEAN] input_ids: {[mb.float().mean().item() for mb in mb_input_ids_lst]}")
        print(f"📊 [MB_MAX] input_ids: {[mb.max().item() for mb in mb_input_ids_lst]}")
        
        print(f"📊 [MB_MIN] position_ids: {[mb.min().item() for mb in mb_position_ids_lst]}")
        print(f"📊 [MB_MEAN] position_ids: {[mb.float().mean().item() for mb in mb_position_ids_lst]}")
        print(f"📊 [MB_MAX] position_ids: {[mb.max().item() for mb in mb_position_ids_lst]}")
        
        print(f"📊 [MB_MIN] attention_mask: {[mb.min().item() for mb in mb_attention_mask_lst]}")
        print(f"📊 [MB_MEAN] attention_mask: {[mb.float().mean().item() for mb in mb_attention_mask_lst]}")
        print(f"📊 [MB_MAX] attention_mask: {[mb.max().item() for mb in mb_attention_mask_lst]}")
        print(f"📊 [MB_SUM] attention_mask: {[mb.sum().item() for mb in mb_attention_mask_lst]}")
        
        print(f"📊 [MB_MIN] responses: {[mb.min().item() for mb in mb_responses_lst]}")
        print(f"📊 [MB_MEAN] responses: {[mb.float().mean().item() for mb in mb_responses_lst]}")
        print(f"📊 [MB_MAX] responses: {[mb.max().item() for mb in mb_responses_lst]}")
        
        print(f"📊 [MB_MIN] log_probs: {[lp.min().item() for lp in log_probs_lst]}")
        print(f"📊 [MB MEAN] log_probs: {[lp.mean().item() for lp in log_probs_lst]}")
        print(f"📊 [MB_MAX] log_probs: {[lp.max().item() for lp in log_probs_lst]}")

        print(f"📊 [MB_MIN] entropy: {[e.min().item() for e in entropy_lst]}")
        print(f"📊 [MB_MEAN] entropy: {[e.mean().item() for e in entropy_lst]}")
        print(f"📊 [MB_MAX] entropy: {[e.max().item() for e in entropy_lst]}")

        print(f"📊 [GLOBAL] log_probs: min={log_probs.min().item():.8f}, mean={log_probs.mean().item():.8f}, max={log_probs.max().item():.8f}")
        print(f"📊 [GLOBAL] entropy: min={entropys.min().item():.8f}, mean={entropys.mean().item():.8f}, max={entropys.max().item():.8f}")

        first_param = next(self.actor_module.parameters())
        print(f"🔑 [WEIGHTS] mean={first_param.mean().item():.10f}, std={first_param.std().item():.10f}")
        
        # 2. Config check
        temperature = data.meta_info["temperature"]
        micro_batch_size = data.meta_info["micro_batch_size"]
        print(f"⚙️ [CONFIG] temp={temperature:.6f}, micro_bsz={micro_batch_size}, training={self.actor_module.training}")
        
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if 'traj_mask' in data.batch:
            select_keys.append('traj_mask')

            if 'is_pad_step' in data.non_tensor_batch:
                is_pad_step = data.non_tensor_batch["is_pad_step"]
                pad_step_indices = np.where(is_pad_step == True)[0]
                if len(pad_step_indices) > 0:
                    data.batch["advantages"][pad_step_indices] = 0

        print(f"[DEBUG] update_policy: Selecting batch keys: {select_keys}")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # 🎯 DECODE & VISUALIZE TOKENS (First 5 samples) - COMMON FOR ALL TRAINERS

        # 🎯 DECODE & VISUALIZE TOKENS (First 5 samples) - COMMON FOR ALL TRAINERS
        # if self.tokenizer is not None and batch["input_ids"].shape[0] > 0:
            
        #     import torch.distributed as dist
        #     print("Dist rank was: ", dist.get_rank())
        #     if not dist.is_initialized() or dist.get_rank() == 0:
                
        #         num_samples = min(1, batch["input_ids"].shape[0])
        #         output = ""
        #         output += f"\n{'#'*100}\n"
        #         output += f"{'#'*100}\n"
        #         output += f"🔍 DP_ACTOR RECEIVED - SHOWING {num_samples} SAMPLES\n"
        #         output += f"{'#'*100}\n"
        #         output += f"{'#'*100}\n\n"
                
        #         for i in range(num_samples):
        #             ids = batch["input_ids"][i]
        #             attn = batch["attention_mask"][i]
        #             resp = batch["responses"][i]
        #             advs = batch["advantages"][i]
        #             old_lp = batch["old_log_probs"][i]
                    
        #             # Decode to words
        #             words = [self.tokenizer.decode([tok], skip_special_tokens=False) for tok in ids]
        #             resp_words = self.tokenizer.decode(resp[resp != self.tokenizer.pad_token_id], skip_special_tokens=False)
                    
        #             # Build colored output
        #             colored_text = ""
        #             for w, a in zip(words, attn):
        #                 if a == 1:
        #                     colored_text += f"\033[92m{w}\033[0m"  # Green for attended
        #                 else:
        #                     colored_text += f"\033[90m{w}\033[0m"  # Gray for padding
                    
        #             output += f"\n{'='*100}\n"
        #             output += f"📋 SAMPLE [{i+1}/{num_samples}]\n"
        #             output += f"{'='*100}\n"
        #             output += f"🔍 FULL SEQUENCE (green=attended, gray=padding):\n"
        #             output += colored_text + "\n"
        #             output += f"\n📝 RESPONSE ONLY: {resp_words}\n"
        #             output += f"📊 ADVANTAGES: mean={advs.mean():.15f} std={advs.std():.15f} min={advs.min():.15f} max={advs.max():.15f}\n"
        #             output += f"📉 OLD_LOG_PROBS: mean={old_lp[old_lp!=0].mean():.15f} min={old_lp[old_lp!=0].min():.15f} max={old_lp[old_lp!=0].max():.15f}\n"
        #             output += f"{'='*100}\n\n"
                
        #         output += f"{'#'*100}\n"
        #         output += f"END OF {num_samples} SAMPLES IN DP_ACTOR\n"
        #         output += f"{'#'*100}\n\n"
                
        #         print(output)
        if self.config.use_dynamic_mini_batch:
            num_mini_batches = self.config.ppo_num_mini_batches
            self.config.ppo_mini_batch_size = math.ceil(data.batch.batch_size[0] / self.config.ppo_num_mini_batches)
            print(f"[DEBUG] update_policy: Dynamic mini batch enabled, ppo_mini_batch_size: {self.config.ppo_mini_batch_size}")
        else:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            print(f"[DEBUG] update_policy: Static mini batch, num_mini_batches: {num_mini_batches}, ppo_mini_batch_size: {self.config.ppo_mini_batch_size}")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        print(f"[DEBUG] update_policy: Creating dataloader")
        if has_multi_modal_inputs:
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    elif "traj_mask" in data:
                        response_mask = data['traj_mask']
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, debug_info = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )
                    if entropy_coeff == 0:
                        loss_agg_mode_entropy = 'token-mean'
                    else:
                        loss_agg_mode_entropy = loss_agg_mode
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode_entropy)
                    with torch.no_grad():
                        entropy_token_mean_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode='token-mean')

                    # compute policy loss
                    if entropy_coeff != 0:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_token_mean_loss': entropy_token_mean_loss.detach().item(),
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/debug_old_log_prob_mean": debug_info["old_log_prob_mean"],
                        "actor/debug_log_prob_mean": debug_info["log_prob_mean"],
                        "actor/debug_response_mask_ratio": debug_info["response_mask_ratio"],
                        "actor/debug_negative_approx_kl_mean": debug_info["negative_approx_kl_mean"],
                    }
                    append_to_dict(metrics, data)

                # 🔍 LOG: Capture weight mean BEFORE optimizer step
                first_param = next(self.actor_module.parameters())
                pre_mean = first_param.data.mean().item()

                grad_norm = self._optimizer_step()

                
                
                # 🔍 GRADIENT INSPECTION (print every 10 mini-batches)
                # This runs AFTER optimizer step, BEFORE zero_grad
                if not hasattr(self, '_grad_step_counter'):
                    self._grad_step_counter = 0
                self._grad_step_counter += 1
                
                if self._grad_step_counter % 2 == 0:
                    import torch.distributed as dist
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    world_size = dist.get_world_size() if dist.is_initialized() else 1
                    
                    # ALL workers print to compare gradients across ranks
                    # Build output as single string
                    output = []
                    output.append(f"\n{'='*80}")
                    output.append(f"🔍 GRADIENT INSPECTION (mini-batch {self._grad_step_counter}) [Rank {rank}/{world_size}]")
                    output.append(f"{'='*80}\n")
                    
                    # Print ALL parameters from named_parameters()
                    output.append(f"📦 ALL PARAMETERS (via named_parameters()):\n")
                    param_count = 0
                    params_with_grad = 0
                    
                    # Aggregate statistics across all parameters
                    all_grad_norms = []
                    all_grad_means = []
                    all_grad_stds = []
                    all_grad_mins = []
                    all_grad_maxs = []
                    
                    for name, param in self.actor_module.named_parameters():
                        param_count += 1
                        has_grad = param.grad is not None
                        
                        if has_grad:
                            params_with_grad += 1
                            g_norm = param.grad.norm().item()
                            g_mean = param.grad.mean().item()
                            g_std = param.grad.std().item()
                            g_min = param.grad.min().item()
                            g_max = param.grad.max().item()
                            
                            # Collect for aggregate stats
                            all_grad_norms.append(g_norm)
                            all_grad_means.append(g_mean)
                            all_grad_stds.append(g_std)
                            all_grad_mins.append(g_min)
                            all_grad_maxs.append(g_max)
                            
                            grad_str = f"GRAD: norm={g_norm:.6f} mean={g_mean:.10f} std={g_std:.6f} range=[{g_min:.6f}, {g_max:.6f}]"
                        else:
                            grad_str = "GRAD: None"
                        
                        output.append(f"[{param_count}] {name}")
                        output.append(f"    shape={tuple(param.shape)}, dtype={param.dtype}, requires_grad={param.requires_grad}")
                        output.append(f"    {grad_str}\n")
                    
                    output.append(f"{'─'*80}")
                    
                    import numpy as np
                    avg_norm = np.mean(all_grad_norms)
                    avg_mean = np.mean(all_grad_means)
                    avg_std = np.mean(all_grad_stds)
                    global_min = np.min(all_grad_mins)
                    global_max = np.max(all_grad_maxs)
                    output.append(f"SUMMARY [Rank {rank}]: {params_with_grad}/{param_count} params | AvgNorm={avg_norm:.8f} AvgMean={avg_mean:.10f} AvgStd={avg_std:.8f} GlobalMin={global_min:.8f} GlobalMax={global_max:.8f}")
                    
                    output.append(f"{'='*80}\n")
                    # Print all at once
                    print("\n".join(output))

                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        
        self.actor_optimizer.zero_grad()
        return metrics

    def update_policy_mini_batch(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if 'traj_mask' in data.batch:
            select_keys.append('traj_mask')

            if 'is_pad_step' in data.non_tensor_batch:
                is_pad_step = data.non_tensor_batch["is_pad_step"]
                pad_step_indices = np.where(is_pad_step == True)[0]
                if len(pad_step_indices) > 0:
                    data.batch["advantages"][pad_step_indices] = 0

        mini_batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
        
        metrics = {}

        if has_multi_modal_inputs:
            non_tensor_select_keys = ['multi_modal_inputs']
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
            micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
        else:
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            # split batch into micro_batches
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

        self.actor_optimizer.zero_grad()

        for data in micro_batches:
            data = data.cuda()  # actor device is cpu when using offload
            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            if "traj_mask" in data:
                response_mask = data['traj_mask']
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = self.config.clip_ratio
            clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
            clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
            clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
            entropy_coeff = self.config.entropy_coeff
            loss_agg_mode = self.config.loss_agg_mode

            # all return: (bsz, response_length)
            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode)
            # compute entropy loss from entropy
            if entropy_coeff !=0:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            else:
                entropy_loss = torch.tensor(0.0)

            # compute policy loss
            policy_loss = pg_loss - entropy_loss * entropy_coeff

            if self.config.use_kl_loss:
                ref_log_prob = data['ref_log_prob']
                # compute kl loss
                kld = kl_penalty(logprob=log_prob,
                                    ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld,
                                    loss_mask=response_mask,
                                    loss_agg_mode=self.config.loss_agg_mode)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics['actor/kl_loss'] = kl_loss.detach().item()
                metrics['actor/kl_coef'] = self.config.kl_loss_coef

            if self.config.use_dynamic_bsz:
                # relative to the dynamic bsz
                loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
            else:
                loss = policy_loss / self.gradient_accumulation
            loss.backward()

            data = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item(),
            }
            append_to_dict(metrics, data)

        grad_norm = self._optimizer_step()
        data = {'actor/grad_norm': grad_norm.detach().item()}
        append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
