# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import math
from functools import partial
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import preprocess_packed_seqs

logger = logging.getLogger(__name__)

_QWEN3_VL_PAD_TOKEN_ID = 151643  # TODO: hard-code here, need to get from tokenizer

def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Instead of raw tensors, expect a single 'visual_inputs' object in batch
    required_device_keys.add("visual_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    # Move all visual inputs contained tensors to CUDA
                    for k, v in val.__dict__.items():
                        _batch_required_keys[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        is_first_pp_stage: Whether the current stage is the first stage
        is_last_pp_stage: Whether the current stage is the last stage
    Returns:
        TODO: add description
    """
    # All PP stages load from iterator to get input_ids and visual grid info
    # This allows each stage to compute MRoPE position_ids locally without broadcasting
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first_pp_stage,
        is_last_pp_stage=is_last_pp_stage,
    )

    if "visual_inputs" in batch and batch.get("visual_inputs") is not None:
        # convert visual_inputs to multi_modal_inputs which is a dict contains "pixel_values" and "image_grid_thw"
        # TODO(jinliangl): add video support
        multi_modal_inputs = batch.get("visual_inputs").normalized_for_model()
    else:
        multi_modal_inputs = {}

    # return naive batch and don't do any padding or cp slicing
    return (
        batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        multi_modal_inputs,
    )

def pad_or_truncate_2d_attn_mask_to_len(mask: torch.Tensor | None, target_len: int, max_cap: int) -> torch.Tensor | None:
    """Pad or truncate a 2D attention mask to the target length with an upper cap.

    Expects input of shape (batch, seq_len). Pads the last dimension.
    """
    if mask is None:
        return None
    b, s = mask.shape
    pad_value = False if mask.dtype == torch.bool else 0
    if s < target_len:
        return F.pad(mask, (0, target_len - s), value=pad_value)
    if s > max_cap:
        return mask[:, :max_cap]
    return mask 

def pack_or_pad_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    this_pg_collection,
    use_fp8_padding: bool = False,
    force_to_pad_to_seq_len: bool = False,
    seq_length: int = None,
    pack_sequences_in_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackedSeqParams]:
    """
    Pad or truncate the batch sequences to the target length, and build packed sequences.
    If is_qwen3vl, return bshd tokens for be compatible with qwen3vl model.
    Otherwise, return thd tokens and packed sequences.
    """

    batch_size, cur_len = tokens.shape
    device = tokens.device

    tp_size = this_pg_collection.tp.size()
    cp_size = this_pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    divisible_by = math.lcm(divisible_by, 16) if use_fp8_padding else divisible_by

    packed_seq_params = None
    # for normal non-packing path, no change to existing logic: this place need handle cp compatiable padding before going to QWen3VLModel.forward
    if not pack_sequences_in_batch:
        # build bshd sequences with tiny padding to be compatible with qwen3vl model
        target_len = math.ceil(cur_len / divisible_by) * divisible_by
        if force_to_pad_to_seq_len:
            target_len = seq_length
        tokens = pad_or_truncate_2d_to_len(tokens, target_len=target_len, max_cap=target_len, pad_value=0)
        labels = pad_or_truncate_2d_to_len(labels, target_len=target_len, max_cap=target_len, pad_value=-100)
        loss_mask = pad_or_truncate_2d_to_len(loss_mask, target_len=target_len, max_cap=target_len, pad_value=0)
        attention_mask = pad_or_truncate_attn_to_len(attention_mask, target_len=target_len, max_cap=target_len)
        position_ids = pad_or_truncate_pos_to_len(position_ids, target_len=target_len, max_cap=target_len)
    else:
        # for packing path, we need to calculate the actual seq lengths info. For input data, no need padding, QWen3VLModel.forward will handle that and convert to thd format
        # Be aware: we skipped truncating as well, be sure you know the length of your dataset
        
        # Need original attention_mask to calculate the actual seq lengths before packing
        # QWen3VLModel.forward will repeatedly use this info when calling preprocess_packed_seqs
        # NOTE: This block of code calculating cu seqlens info is consistent with that in preprocess_packed_seqs of modeling_qwen3_vl/utils.py, which is called by QWen3VLModel.forward
        # Ideally, we should avoid two places of calling the same function and leave all into QWen3VLModel.forward, but to keep change minimal, we keep the current structure of calculating cu seqlens info in both places
        if attention_mask is None:
            raise ValueError("attention_mask is required when pack_sequences_in_batch is True, but we got None from dataset")
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        pad_size = (divisible_by - seqlens_in_batch % divisible_by) % divisible_by
        seqlens_in_batch_padded = seqlens_in_batch + pad_size

        cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
        max_seqlen_in_batch_padded = seqlens_in_batch_padded.max().item() # NOTE: here should use _padded

        # NOTE: all lens should be based on _padded
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            max_seqlen_q=max_seqlen_in_batch_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_kv=max_seqlen_in_batch_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def recover_2d_attention_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    # Build mask from trailing-padding convention: valid tokens are all positions
    # up to the last non-pad token in each sequence.
    is_pad = input_ids.eq(pad_token_id)
    trailing_pad_count = is_pad.flip(dims=(1,)).to(torch.int32).cumprod(dim=1).sum(dim=1)
    lengths = input_ids.shape[1] - trailing_pad_count

    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
    return position_ids.unsqueeze(0) < lengths.unsqueeze(1)

def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    this_pg_collection = get_pg_collection(model)
    is_first = is_pp_first_stage(this_pg_collection.pp)
    is_last = is_pp_last_stage(this_pg_collection.pp)

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            multi_modal_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, is_first_pp_stage=is_first, is_last_pp_stage=is_last)
    timers("batch-generator").stop()

    # To be compatible with qwen3vl, we move the sequence padding and packing to forward_step function.
    # Qwen3VL model need the original input and do cp and sp split in model.forward.
    pack_sequences_in_batch = getattr(state.cfg.dataset, "pack_sequences_in_batch", False)

    use_fp8_padding = True
    if pack_sequences_in_batch:
        # find the last non-padding token index for each sequence in the batch to get accurate attention mask when pack_sequences_in_batch is True
        attention_mask = recover_2d_attention_mask_from_input_ids(tokens, pad_token_id=_QWEN3_VL_PAD_TOKEN_ID)

    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = pack_or_pad_batch_sequences(
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        this_pg_collection,
        use_fp8_padding=use_fp8_padding,
        force_to_pad_to_seq_len=this_pg_collection.pp.size() > 1,
        seq_length=config.seq_length,
        pack_sequences_in_batch=pack_sequences_in_batch,
    )
    if not pack_sequences_in_batch: 
        # normal path: 
        # - labbels, loss_mask and attention_mask (actually None) go through cp slice
        # - input_ids set as original tokens
        # - position_ids set as None and calculate in QWen3VLModel.forward
        forward_args = {
            "input_ids": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        original_tokens = tokens.clone()
        forward_args = get_batch_on_this_cp_rank(forward_args, cp_group=this_pg_collection.cp)
        forward_args["packed_seq_params"] = None
        forward_args["input_ids"] = original_tokens
        forward_args["position_ids"] = None
        
    else:
        # packing path:
        # - labels, loss_mask go through cp slice and keep here for later loss 
        # - input_ids set as original tokens
        # - attention mask: passed in forward args for later calculating cu seqlens info in QWen3VLModel.forward
        # - position_ids set as None and calculate in QWen3VLModel.forward
        # - later attention_mask set as None in loss / scheduler
        packed_sliced_labels = None
        if labels is not None:
            packed_sliced_labels, _ = preprocess_packed_seqs(input_ids=labels, attention_mask=attention_mask, pre_process=True, pg_collection=this_pg_collection, use_fp8_padding=use_fp8_padding)
        packed_sliced_loss_mask = None
        if loss_mask is not None:
            packed_sliced_loss_mask, _ = preprocess_packed_seqs(input_ids=loss_mask, attention_mask=attention_mask, pre_process=True, pg_collection=this_pg_collection, use_fp8_padding=use_fp8_padding)

        forward_args = {
            "input_ids": tokens,
            "labels": packed_sliced_labels,
            "loss_mask": packed_sliced_loss_mask,
            "attention_mask": attention_mask,
            "position_ids": None,
            "packed_seq_params": packed_seq_params,
            "use_fp8_padding": use_fp8_padding,
        }
    
    # it is a bit messy here but labels and loss_mask are required here to calculate loss, so we have to convert the thd format here in-advance and pass to QWen3VLModel.forward
    # use cp split loss mask for calculate loss
    loss_mask = forward_args["loss_mask"]
    # follow the design of verl, we put the multi-modal inputs in the forward args
    if "pixel_values" in multi_modal_inputs:
        forward_args["pixel_values"] = multi_modal_inputs["pixel_values"]
    if "image_grid_thw" in multi_modal_inputs:
        forward_args["image_grid_thw"] = multi_modal_inputs["image_grid_thw"]
    if "pixel_values_videos" in multi_modal_inputs:
        forward_args["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"]
    if "video_grid_thw" in multi_modal_inputs:
        forward_args["video_grid_thw"] = multi_modal_inputs["video_grid_thw"]

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            # TODO: Here PP suppose broken on packing path, tokens, position_ids, attention_mask are not at the good shape.
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        else:
            output_tensor = model(**forward_args)

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
