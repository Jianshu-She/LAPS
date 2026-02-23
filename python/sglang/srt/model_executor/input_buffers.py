from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
    compute_local_num_token_non_padded,
)


@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor
    input_embeds: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    num_token_non_padded: torch.Tensor
    custom_mask: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    global_num_tokens_gpu: torch.Tensor
    global_num_tokens_for_logprob_gpu: torch.Tensor
    encoder_lens: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_num_token: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        dp_size: int,
        pp_size: int,
        is_encoder_decoder: bool,
        require_mlp_tp_gather: bool,
        seq_len_fill_value: int,
        encoder_len_fill_value: int,
        num_tokens_per_bs: int,
        cache_loc_dtype: torch.dtype,
        enable_mamba_track: bool,
    ) -> "GraphInputBuffers":
        with torch.device(device):
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = torch.ones(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
                dtype=torch.bool,
            )
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size),
                dtype=torch.float,
            )
            mamba_track_indices = (
                torch.zeros((max_bs,), dtype=torch.int64)
                if enable_mamba_track
                else None
            )
            mamba_track_mask = (
                torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
            )

            if pp_size > 1:
                pp_proxy_tensors = {
                    "hidden_states": torch.zeros((max_bs, hidden_size), dtype=dtype),
                    "residual": torch.zeros((max_bs, hidden_size), dtype=dtype),
                }
            else:
                pp_proxy_tensors = None

            if is_encoder_decoder:
                encoder_lens = torch.full(
                    (max_bs,), encoder_len_fill_value, dtype=torch.int32
                )
            else:
                encoder_lens = None

            if require_mlp_tp_gather:
                global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros(
                    (dp_size,), dtype=torch.int32
                )
            else:
                global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

        # Keep seq_lens_cpu as a true CPU tensor, like the old implementation.
        seq_lens_cpu = torch.full(
            (max_bs,),
            seq_len_fill_value,
            dtype=torch.int32,
            device="cpu",
        )

        return cls(
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            num_token_non_padded=num_token_non_padded,
            custom_mask=custom_mask,
            next_token_logits_buffer=next_token_logits_buffer,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            encoder_lens=encoder_lens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def populate_from_forward_batch(
        self,
        *,
        forward_batch: ForwardBatch,
        raw_bs: int,
        raw_num_token: int,
        bs: int,
        seq_len_fill_value: int,
        require_gathered_buffer: bool,
        num_tokens_per_bs: int,
        nsa_enable_prefill_cp: bool,
        enable_num_token_non_padded_flag: bool,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Optional[torch.Tensor]:
        if bs != raw_bs:
            self.seq_lens.fill_(seq_len_fill_value)
            self.out_cache_loc.zero_()
            if self.mamba_track_indices is not None:
                self.mamba_track_indices.zero_()
            if self.mamba_track_mask is not None:
                self.mamba_track_mask.fill_(False)

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        if (
            self.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            self.mamba_track_indices[:raw_bs].copy_(forward_batch.mamba_track_indices)
        if (
            self.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            self.mamba_track_mask[:raw_bs].copy_(forward_batch.mamba_track_mask)

        seq_lens_cpu: Optional[torch.Tensor] = None
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            seq_lens_cpu = self.seq_lens_cpu[:bs]

        if self.encoder_lens is not None and forward_batch.encoder_lens is not None:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)

        if require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * num_tokens_per_bs)

        if enable_num_token_non_padded_flag:
            if require_gathered_buffer and not nsa_enable_prefill_cp:
                num_tokens_per_dp = bs * num_tokens_per_bs
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=forward_batch.num_token_non_padded,
                    num_tokens_per_dp=num_tokens_per_dp,
                )
                self.num_token_non_padded.copy_(local)
            else:
                self.num_token_non_padded.copy_(forward_batch.num_token_non_padded)

        # Pipeline-parallel proxy tensors.
        if pp_proxy_tensors is not None and self.pp_proxy_tensors is not None:
            for key, buf in self.pp_proxy_tensors.items():
                src = pp_proxy_tensors.tensors[key]
                dim = src.shape[0]
                buf[:dim].copy_(src)

        return seq_lens_cpu


@dataclass
class BatchPrefillGraphInputBuffers:
    """
    Pre-allocated buffers for batch prefill CUDA graph capture.

    Unlike GraphInputBuffers (used for decode), this class supports batched prefill
    where multiple sequences with potentially different lengths are processed together
    in a single CUDA graph.

    Key differences from single-sequence prefill:
    - Supports multiple sequences (batch_size > 1) in extend/prefill mode
    - Uses uniform padded sequence length for graph capture
    - Handles padding for sequences shorter than the captured length
    """
    input_ids: torch.Tensor              # (max_bs * max_seq_len,)
    positions: torch.Tensor              # (max_bs * max_seq_len,)
    out_cache_loc: torch.Tensor          # (max_bs * max_seq_len,)
    req_pool_indices: torch.Tensor       # (max_bs,)
    seq_lens: torch.Tensor               # (max_bs,) - actual seq lens (for KV cache lookup)
    extend_seq_lens: torch.Tensor        # (max_bs,) - padded extend seq lens
    extend_prefix_lens: torch.Tensor     # (max_bs,)
    extend_start_loc: torch.Tensor       # (max_bs,)
    seq_lens_cpu: torch.Tensor           # (max_bs,) on CPU
    extend_seq_lens_cpu: torch.Tensor    # (max_bs,) on CPU
    extend_prefix_lens_cpu: torch.Tensor # (max_bs,) on CPU
    input_embeds: Optional[torch.Tensor] # (max_bs * max_seq_len, hidden_size)
    mrope_positions: Optional[torch.Tensor]  # (3, max_bs * max_seq_len)

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_seq_len: int,
        hidden_size: int,
        dtype: torch.dtype,
        cache_loc_dtype: torch.dtype,
        is_multimodal: bool = False,
    ) -> "BatchPrefillGraphInputBuffers":
        """
        Create pre-allocated buffers for batch prefill CUDA graph.

        Args:
            device: Target device for GPU tensors
            max_bs: Maximum batch size to capture
            max_seq_len: Maximum sequence length per request
            hidden_size: Model hidden size (for input_embeds)
            dtype: Data type for embeddings
            cache_loc_dtype: Data type for cache location indices
            is_multimodal: Whether model is multimodal (needs input_embeds buffer)
        """
        max_num_tokens = max_bs * max_seq_len

        with torch.device(device):
            input_ids = torch.zeros((max_num_tokens,), dtype=torch.int64)
            positions = torch.zeros((max_num_tokens,), dtype=torch.int64)
            out_cache_loc = torch.zeros((max_num_tokens,), dtype=cache_loc_dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = torch.zeros((max_bs,), dtype=torch.int32)
            extend_seq_lens = torch.zeros((max_bs,), dtype=torch.int32)
            extend_prefix_lens = torch.zeros((max_bs,), dtype=torch.int32)
            extend_start_loc = torch.zeros((max_bs,), dtype=torch.int32)

            if is_multimodal:
                input_embeds = torch.zeros((max_num_tokens, hidden_size), dtype=dtype)
                mrope_positions = torch.zeros((3, max_num_tokens), dtype=torch.int64)
            else:
                input_embeds = None
                mrope_positions = None

        # CPU tensors for attention metadata
        seq_lens_cpu = torch.zeros((max_bs,), dtype=torch.int32, device="cpu")
        extend_seq_lens_cpu = torch.zeros((max_bs,), dtype=torch.int32, device="cpu")
        extend_prefix_lens_cpu = torch.zeros((max_bs,), dtype=torch.int32, device="cpu")

        return cls(
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
        )

    def populate_from_forward_batch(
        self,
        forward_batch: ForwardBatch,
        target_bs: int,
        target_seq_len: int,
    ) -> Tuple[int, List[int]]:
        """
        Populate buffers from a forward batch with padding to target dimensions.

        This method copies data from the forward batch to the pre-allocated buffers,
        padding sequences to the target_seq_len. For padded positions:
        - input_ids: Set to 0 (ignored by model)
        - out_cache_loc: Set to 0 (no KV cache write for padded positions)
        - positions: Copied only for real positions (padded positions ignored by attention)

        Args:
            forward_batch: The incoming batch of requests
            target_bs: Target batch size for the graph (>= forward_batch.batch_size)
            target_seq_len: Target sequence length for uniform padding

        Returns:
            Tuple of (total_tokens, valid_token_indices) where:
            - total_tokens: Total number of tokens in the padded batch
            - valid_token_indices: List of indices for valid (non-padded) tokens
        """
        actual_bs = forward_batch.batch_size
        total_tokens = target_bs * target_seq_len

        # Zero out all buffers for padding
        self.input_ids[:total_tokens].zero_()
        self.out_cache_loc[:total_tokens].zero_()
        self.positions[:total_tokens].zero_()
        self.extend_seq_lens[:target_bs].fill_(target_seq_len)
        self.extend_prefix_lens[:target_bs].zero_()

        # Compute extend_start_loc for uniform target_seq_len
        self.extend_start_loc[:target_bs] = torch.arange(
            0, total_tokens, target_seq_len,
            device=self.extend_start_loc.device
        )[:target_bs]

        # Track valid token indices (for extracting outputs later)
        valid_token_indices = []

        # Copy real data per sequence
        src_offset = 0
        for i in range(actual_bs):
            _v = forward_batch.extend_seq_lens_cpu[i]
            actual_len = _v.item() if hasattr(_v, 'item') else int(_v)
            dst_offset = i * target_seq_len

            # Copy input_ids for real tokens
            self.input_ids[dst_offset:dst_offset + actual_len].copy_(
                forward_batch.input_ids[src_offset:src_offset + actual_len]
            )

            # Copy out_cache_loc for real tokens (padded positions stay 0)
            self.out_cache_loc[dst_offset:dst_offset + actual_len].copy_(
                forward_batch.out_cache_loc[src_offset:src_offset + actual_len]
            )

            # Copy positions for real tokens
            self.positions[dst_offset:dst_offset + actual_len].copy_(
                forward_batch.positions[src_offset:src_offset + actual_len]
            )

            # Track valid token indices (last token of each real sequence)
            # For logits, we typically only need the last token of each sequence
            valid_token_indices.append(dst_offset + actual_len - 1)

            src_offset += actual_len

        # Copy req_pool_indices
        self.req_pool_indices[:actual_bs].copy_(forward_batch.req_pool_indices)

        # Copy seq_lens (actual KV cache lengths for attention lookup)
        self.seq_lens[:actual_bs].copy_(forward_batch.seq_lens)

        # Fill remaining batch slots with dummy values (for padding)
        if actual_bs < target_bs:
            # Use first request's pool index as dummy (won't affect attention output)
            self.req_pool_indices[actual_bs:target_bs].fill_(
                forward_batch.req_pool_indices[0].item()
            )
            self.seq_lens[actual_bs:target_bs].fill_(1)  # Minimal seq len

        # CPU tensors for seq_lens
        self.seq_lens_cpu[:actual_bs].copy_(forward_batch.seq_lens_cpu)
        if actual_bs < target_bs:
            self.seq_lens_cpu[actual_bs:target_bs].fill_(1)

        # Set extend_seq_lens_cpu and extend_prefix_lens_cpu
        self.extend_seq_lens_cpu[:target_bs].fill_(target_seq_len)
        self.extend_prefix_lens_cpu[:target_bs].zero_()

        # Handle multimodal inputs if present
        if self.input_embeds is not None and forward_batch.input_embeds is not None:
            src_offset = 0
            for i in range(actual_bs):
                _v = forward_batch.extend_seq_lens_cpu[i]
                actual_len = _v.item() if hasattr(_v, 'item') else int(_v)
                dst_offset = i * target_seq_len
                self.input_embeds[dst_offset:dst_offset + actual_len].copy_(
                    forward_batch.input_embeds[src_offset:src_offset + actual_len]
                )
                src_offset += actual_len

        if self.mrope_positions is not None and forward_batch.mrope_positions is not None:
            src_offset = 0
            for i in range(actual_bs):
                _v = forward_batch.extend_seq_lens_cpu[i]
                actual_len = _v.item() if hasattr(_v, 'item') else int(_v)
                dst_offset = i * target_seq_len
                self.mrope_positions[:, dst_offset:dst_offset + actual_len].copy_(
                    forward_batch.mrope_positions[:, src_offset:src_offset + actual_len]
                )
                src_offset += actual_len

        return total_tokens, valid_token_indices
