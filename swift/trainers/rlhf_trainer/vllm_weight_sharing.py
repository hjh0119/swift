# Copyright (c) Alibaba, Inc. and its affiliates.
"""
vLLM Weight Sharing Module for Swift GRPO Training

This module implements zero-copy weight sharing between training models and vLLM inference models,
inspired by Unsloth's VLLM_STANDBY mechanism. It patches vLLM's memory allocator to keep model
weights resident in GPU memory during sleep/wake cycles, while offloading other memory (KV cache).

Key Components:
1. CuMemAllocator patching: Override sleep/wake_up to skip 'weights' tag
2. Worker.load_model patching: Ensure weights are allocated with proper tag
3. Environment setup: Configure PyTorch CUDA allocator for compatibility
4. Weight reference management: Link training model parameters to vLLM's weight tensors
"""

import os
import re
from contextlib import contextmanager
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch

from swift.utils import get_logger

logger = get_logger()

# Global flag to track if patches have been applied
_PATCHES_APPLIED = False
_WEIGHT_SHARING_ENABLED = False


def patch_cumem_allocator():
    """
    Patch vLLM's CuMemAllocator to skip weight memory during sleep/wake_up cycles.

    This is the core mechanism for weight sharing. By preventing weights from being
    offloaded during sleep(), both the training model and vLLM model can access
    the same GPU memory without data transfers.
    """
    try:
        from vllm.device_allocator.cumem import (
            create_and_map,
            CuMemAllocator,
            libcudart,
            unmap_and_release,
        )
        from vllm.utils import is_pin_memory_available
        import ctypes
    except ImportError:
        logger.warning('Failed to import CuMemAllocator from vllm. Weight sharing may not work.')
        return

    allocator = CuMemAllocator.get_instance()

    def sleep(self, offload_tags: Optional[Tuple[str, ...]] = None):
        """
        Modified sleep that preserves weights in GPU memory.

        Args:
            offload_tags: Tags of memory allocations to offload to CPU.
                         vLLM v1 passes ("weights",) for level=1, () for level=2
                         We ignore this and always preserve weights in GPU.
        """
        import gc
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag, )
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags, )

        assert isinstance(offload_tags, tuple)

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            if data.tag in offload_tags:
                if data.tag == 'weights':
                    continue
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes, dtype=torch.uint8, device='cpu', pin_memory=is_pin_memory_available())
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: Optional[List[str]] = None):
        """
        Modified wake_up that restores non-weight memory.

        Args:
            tags: Optional list of tags to selectively wake up (e.g., ['kv_cache'])
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                if data.tag == 'weights':
                    continue
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    # Apply patches
    allocator.sleep = MethodType(sleep, allocator)
    allocator.wake_up = MethodType(wake_up, allocator)
    logger.info('[Weight Sharing] Successfully patched CuMemAllocator')


def patch_worker_load_model():
    """
    Patch vLLM Worker's load_model to ensure weights are tagged correctly.

    This ensures that when vLLM loads model weights, they are allocated within
    a use_memory_pool(tag="weights") context so CuMemAllocator can identify them.
    """
    try:
        from vllm.v1.worker.gpu_worker import Worker
        from vllm.device_allocator.cumem import CuMemAllocator
    except ImportError:
        logger.warning('Failed to import Worker or CuMemAllocator from vllm')
        return

    original_load_model = Worker.load_model

    def patched_load_model(self):
        """Load model with proper weight tagging"""
        logger.info('[Weight Sharing] Loading vLLM model with weight tagging')

        # Get the CuMemAllocator instance
        allocator = CuMemAllocator.get_instance()

        # Load model within weights context
        with allocator.use_memory_pool(tag='weights'):
            result = original_load_model(self)

        logger.info('[Weight Sharing] vLLM model loaded, weights are tagged')
        return result

    Worker.load_model = patched_load_model
    logger.info('[Weight Sharing] Successfully patched Worker.load_model')


def setup_weight_sharing_environment():
    """
    Configure environment variables required for weight sharing.

    Key settings:
    1. Disable expandable_segments to prevent conflicts with vLLM's memory management
    2. Disable vLLM multiprocessing (weight sharing requires single-process)
    """
    # Get current PYTORCH_CUDA_ALLOC_CONF
    cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')

    # Remove expandable_segments if present
    conf_parts = [part.strip() for part in cuda_alloc_conf.split(',') if part.strip()]
    conf_parts = [part for part in conf_parts if not part.startswith('expandable_segments')]

    # Add expandable_segments:False
    conf_parts.append('expandable_segments:False')

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(conf_parts)
    logger.info(f'[Weight Sharing] Set PYTORCH_CUDA_ALLOC_CONF={os.environ["PYTORCH_CUDA_ALLOC_CONF"]}')

    # Disable vLLM multiprocessing
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    logger.info('[Weight Sharing] Disabled vLLM multiprocessing for weight sharing')


def patch_vllm_for_weight_sharing():
    """
    Main entry point: Apply all vLLM patches for weight sharing.

    This function should be called before vLLM engine initialization.
    It's idempotent - calling multiple times is safe.
    """
    global _PATCHES_APPLIED, _WEIGHT_SHARING_ENABLED

    if _PATCHES_APPLIED:
        logger.info('[Weight Sharing] Patches already applied, skipping')
        return

    logger.info('[Weight Sharing] Applying vLLM patches for weight sharing...')

    # Setup environment
    setup_weight_sharing_environment()

    # Apply patches
    patch_cumem_allocator()
    patch_worker_load_model()

    _PATCHES_APPLIED = True
    _WEIGHT_SHARING_ENABLED = True

    logger.info('[Weight Sharing] All patches applied successfully')


def get_vllm_model_weights(vllm_engine) -> Dict[str, torch.Tensor]:
    """
    Extract weight references from vLLM model.

    Args:
        vllm_engine: GRPOVllmEngine instance

    Returns:
        Dictionary mapping parameter names to tensors
    """
    try:
        llm_model = vllm_engine.inner_model
        weights = {}

        # Get state dict (this doesn't copy data, just references)
        for name, param in llm_model.named_parameters():
            weights[name] = param

        logger.info(f'[Weight Sharing] Retrieved {len(weights)} weight references from vLLM')
        return weights
    except Exception as e:
        logger.error(f'[Weight Sharing] Failed to get vLLM weights: {e}')
        return {}


def init_training_model_with_vllm_weights(training_model: torch.nn.Module,
                                          vllm_weights: Dict[str, torch.Tensor],
                                          is_lora: bool = False) -> Tuple[int, int]:
    """
    Initialize training model parameters to reference vLLM's weight tensors.

    This achieves true zero-copy weight sharing: the training model's parameters
    directly point to the same GPU memory as vLLM's weights.

    Args:
        training_model: The training model (may be wrapped by PEFT)
        vllm_weights: Weight references from vLLM
        is_lora: Whether using LoRA training

    Returns:
        Tuple of (shared_params_count, total_params_count)
    """
    shared_count = 0
    total_count = 0

    # Build mapping from base parameter names to vLLM weights
    # Handle potential prefix differences (e.g., "base_model.model." for PEFT)
    vllm_weight_map = {}
    for vllm_name, vllm_tensor in vllm_weights.items():
        # Store both with and without common prefixes
        vllm_weight_map[vllm_name] = vllm_tensor
        # Remove model. prefix if present
        if vllm_name.startswith('model.'):
            vllm_weight_map[vllm_name[6:]] = vllm_tensor

    for name, param in training_model.named_parameters():
        total_count += 1

        # Skip LoRA parameters in LoRA mode
        if is_lora and ('lora_' in name or 'modules_to_save' in name):
            continue

        # Find corresponding vLLM weight
        # Try multiple name variations
        vllm_name_candidates = [
            name,
            re.sub(r'^base_model\.model\.', '', name),  # Remove PEFT prefix
            re.sub(r'^_model\.', '', name),  # Remove Swift prefix
            re.sub(r'\.base_layer', '', name),  # Remove LoRA base_layer suffix
        ]

        vllm_tensor = None
        matched_name = None
        for candidate in vllm_name_candidates:
            if candidate in vllm_weight_map:
                vllm_tensor = vllm_weight_map[candidate]
                matched_name = candidate
                break

        if vllm_tensor is None:
            # Can't find matching weight, skip
            continue

        # Verify shape compatibility
        if param.shape != vllm_tensor.shape:
            logger.warning(f'[Weight Sharing] Shape mismatch for {name}: '
                           f'training={param.shape}, vllm={vllm_tensor.shape}')
            continue

        # CRITICAL: Make training parameter reference vLLM's tensor
        # This is zero-copy - both models now share the same GPU memory
        param.data = vllm_tensor
        shared_count += 1

        logger.debug(f'[Weight Sharing] Shared {name} -> {matched_name}')

    logger.info(f'[Weight Sharing] Weight sharing complete: {shared_count}/{total_count} parameters shared')
    return shared_count, total_count


def verify_weight_sharing(training_model: torch.nn.Module, vllm_engine, check_count: int = 5) -> bool:
    """
    Verify that weight sharing is working correctly by checking memory pointers.

    Args:
        training_model: Training model
        vllm_engine: vLLM engine
        check_count: Number of parameters to check

    Returns:
        True if weight sharing is verified
    """
    vllm_weights = get_vllm_model_weights(vllm_engine)

    checked = 0
    matched = 0

    for name, param in training_model.named_parameters():
        if 'lora_' in name or 'modules_to_save' in name:
            continue

        # Find corresponding vLLM weight
        clean_name = re.sub(r'^base_model\.model\.', '', name)
        clean_name = re.sub(r'^_model\.', '', clean_name)
        clean_name = re.sub(r'\.base_layer', '', clean_name)

        if clean_name not in vllm_weights:
            continue

        vllm_tensor = vllm_weights[clean_name]

        # Check if they share the same memory
        if param.data.data_ptr() == vllm_tensor.data_ptr():
            matched += 1
            logger.info(f'[Weight Sharing] ✓ Verified shared memory for {name}')
        else:
            logger.warning(f'[Weight Sharing] ✗ Memory mismatch for {name}: '
                           f'training_ptr={param.data.data_ptr()}, vllm_ptr={vllm_tensor.data_ptr()}')

        checked += 1
        if checked >= check_count:
            break

    success = matched == checked and checked > 0
    if success:
        logger.info(f'[Weight Sharing] Verification passed: {matched}/{checked} parameters share memory')
    else:
        logger.error(f'[Weight Sharing] Verification failed: only {matched}/{checked} parameters share memory')

    return success


@contextmanager
def temporary_expandable_segments(enabled: bool):
    """
    Temporarily enable/disable expandable_segments for a code block.

    This is useful for operations that require expandable_segments to be off
    (like vLLM operations) but need to restore the original setting afterward.
    """
    cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')

    # Parse current setting
    conf_parts = [part.strip() for part in cuda_alloc_conf.split(',') if part.strip()]
    original_expandable = any('expandable_segments:True' in part for part in conf_parts)

    # Set new value
    conf_parts = [part for part in conf_parts if not part.startswith('expandable_segments')]
    conf_parts.append(f'expandable_segments:{enabled}')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(conf_parts)

    try:
        yield
    finally:
        # Restore original value
        conf_parts = [part for part in conf_parts if not part.startswith('expandable_segments')]
        conf_parts.append(f'expandable_segments:{original_expandable}')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(conf_parts)


def is_weight_sharing_enabled() -> bool:
    """Check if weight sharing is currently enabled"""
    return _WEIGHT_SHARING_ENABLED
