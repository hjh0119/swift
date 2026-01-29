# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for fetching teacher model logprobs from swift deploy or vLLM server.

This module provides a client for communicating with OpenAI-compatible endpoints
(e.g., swift deploy with vLLM backend, standalone vLLM server) to obtain teacher
model logprobs for knowledge distillation (GKD) training.
"""
import asyncio
import logging
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar

import aiohttp
import torch

logger = logging.getLogger(__name__)

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a sync context, handling nested event loops.

    This utility function handles the complexity of running async code from
    synchronous contexts, including when an event loop is already running.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Event loop already running (e.g., in Jupyter or nested async)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists
        return asyncio.run(coro)


class TeacherAPIClient:
    """Client for fetching teacher logprobs from swift deploy or vLLM server.

    This client supports two API formats:

    1. Swift Deploy / Chat Completions API (preferred):
       - Uses /v1/chat/completions endpoint
       - Sets prompt_logprobs parameter to get logprobs for input tokens
       - Works with swift deploy using vLLM backend

    2. vLLM Native Completions API (fallback):
       - Uses /v1/completions endpoint
       - Works with standalone vLLM server

    The client auto-detects which API format the server supports.

    Args:
        base_url: The base URL of the teacher model server (e.g., 'http://localhost:8000').
        top_logprobs: Number of top log probabilities to request per token.
        timeout: Request timeout in seconds.
        api_key: Optional API key for authentication.
        model_name: Optional model name for the API request. If None, auto-detects.
        tokenizer: Optional tokenizer for converting text prompts. If provided,
            can decode response tokens.
    """

    def __init__(
        self,
        base_url: str,
        top_logprobs: int = 20,
        timeout: float = 300.0,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.top_logprobs = top_logprobs
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key
        self.model_name = model_name
        self.tokenizer = tokenizer
        self._api_format = None  # 'swift_deploy' or 'vllm_native', detected on first request

        if top_logprobs <= 0:
            raise ValueError(f'top_logprobs must be positive, got {top_logprobs}')

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    async def _get_model_name(self, session: aiohttp.ClientSession) -> str:
        """Get model name from server if not provided."""
        if self.model_name:
            return self.model_name

        try:
            async with session.get(
                    f'{self.base_url}/v1/models', headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data') and len(data['data']) > 0:
                        self.model_name = data['data'][0]['id']
                        return self.model_name
        except Exception as e:
            logger.warning(f'Failed to get model name: {e}')

        self.model_name = 'default'
        return self.model_name

    async def _detect_api_format(self, session: aiohttp.ClientSession, model_name: str) -> str:
        """Detect which API format the server supports.

        Returns:
            'swift_deploy' if server supports prompt_logprobs in chat/completions
            'vllm_native' if server supports vLLM native completions API
        """
        if self._api_format is not None:
            return self._api_format

        # Try swift deploy format first (chat/completions with prompt_logprobs)
        url = f'{self.base_url}/v1/chat/completions'
        test_payload = {
            'model': model_name,
            'messages': [{
                'role': 'user',
                'content': 'Hi'
            }],
            'max_tokens': 1,
            'temperature': 0,
            'prompt_logprobs': 5,
        }

        try:
            async with session.post(
                    url, json=test_payload, headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check if prompt_logprobs is returned
                    choices = data.get('choices', [])
                    if choices and choices[0].get('prompt_logprobs') is not None:
                        self._api_format = 'swift_deploy'
                        logger.info('Detected swift deploy API format with prompt_logprobs support')
                        return self._api_format
        except Exception as e:
            logger.debug(f'Swift deploy API detection failed: {e}')

        # Try vLLM native format
        url = f'{self.base_url}/v1/completions'
        test_payload = {
            'model': model_name,
            'prompt': [1, 2, 3],  # Token IDs
            'max_tokens': 0,
            'temperature': 0,
            'logprobs': 5,
        }

        try:
            async with session.post(
                    url, json=test_payload, headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    self._api_format = 'vllm_native'
                    logger.info('Detected vLLM native API format')
                    return self._api_format
        except Exception as e:
            logger.debug(f'vLLM native API detection failed: {e}')

        # Default to swift deploy and hope for the best
        self._api_format = 'swift_deploy'
        logger.warning('Could not detect API format, defaulting to swift deploy')
        return self._api_format

    async def get_logprobs_batch(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch logprobs for a batch of sequences.

        Args:
            input_ids: List of token ID sequences.
            top_logprobs: Override the default top_logprobs if provided.

        Returns:
            List of dictionaries, each containing:
            - 'indices': List of token indices per position [seq_len, topk]
            - 'values': List of logprob values per position [seq_len, topk]
        """
        topk = top_logprobs or self.top_logprobs

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            model_name = await self._get_model_name(session)
            api_format = await self._detect_api_format(session, model_name)

            # Create tasks for concurrent requests
            if api_format == 'swift_deploy':
                tasks = [self._fetch_swift_deploy(session, model_name, ids, topk) for ids in input_ids]
            else:
                tasks = [self._fetch_vllm_native(session, model_name, ids, topk) for ids in input_ids]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions that occurred
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f'Request {i} failed with exception: {result}')
                    processed_results.append(self._empty_result(len(input_ids[i]), topk))
                else:
                    processed_results.append(result)
            return processed_results

    async def _fetch_swift_deploy(
        self,
        session: aiohttp.ClientSession,
        model_name: str,
        ids: List[int],
        topk: int,
    ) -> Dict[str, Any]:
        """Fetch logprobs using swift deploy's chat/completions API with prompt_logprobs.

        This converts token IDs to text using the tokenizer (if available) or
        sends as a raw text prompt.
        """
        # Convert token IDs to text for chat completions API
        if self.tokenizer is not None:
            prompt_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        else:
            # Fallback: try to use the server's tokenizer by sending a special request
            # For now, just convert to string representation
            prompt_text = ''.join(chr(i) if 32 <= i < 127 else f'<{i}>' for i in ids[:100])
            logger.warning_once('No tokenizer provided to TeacherAPIClient. '
                                'Prompt may not be decoded correctly. Pass tokenizer for accurate results.')

        url = f'{self.base_url}/v1/chat/completions'
        payload = {
            'model': model_name,
            'messages': [{
                'role': 'user',
                'content': prompt_text
            }],
            'max_tokens': 1,  # Minimum required by swift deploy, we only need prompt_logprobs
            'temperature': 0,
            'prompt_logprobs': topk,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=self._get_headers()) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        logger.warning(f'API error after {max_retries} retries: {resp.status} - {error_text}')
                        return self._empty_result(len(ids), topk)

                    data = await resp.json()
                    return self._parse_swift_deploy_response(data, len(ids), topk)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                logger.warning(f'Failed to get logprobs after {max_retries} retries: {e}')
                return self._empty_result(len(ids), topk)

        return self._empty_result(len(ids), topk)

    async def _fetch_vllm_native(
        self,
        session: aiohttp.ClientSession,
        model_name: str,
        ids: List[int],
        topk: int,
    ) -> Dict[str, Any]:
        """Fetch logprobs using vLLM native completions API"""
        url = f'{self.base_url}/v1/completions'
        payload = {
            'model': model_name,
            'prompt': ids,  # Token IDs directly
            'max_tokens': 0,
            'temperature': 0,
            'logprobs': topk,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=self._get_headers()) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        logger.warning(f'API error after {max_retries} retries: {resp.status} - {error_text}')
                        return self._empty_result(len(ids), topk)

                    data = await resp.json()
                    return self._parse_vllm_native_response(data, len(ids), topk)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                logger.warning(f'Failed to get logprobs after {max_retries} retries: {e}')
                return self._empty_result(len(ids), topk)

        return self._empty_result(len(ids), topk)

    def _parse_swift_deploy_response(self, response: Dict[str, Any], seq_len: int, topk: int) -> Dict[str, Any]:
        """Parse swift deploy chat/completions response with prompt_logprobs.

        The response format is:
        {
            "choices": [{
                "prompt_logprobs": [
                    {"token_id": int, "token": str, "logprob": float, "top_logprobs": [...]},
                    ...
                ]
            }]
        }
        """
        result = {'indices': [], 'values': []}

        try:
            if 'choices' not in response or len(response['choices']) == 0:
                return self._empty_result(seq_len, topk)

            choice = response['choices'][0]
            prompt_logprobs = choice.get('prompt_logprobs')

            if prompt_logprobs is None:
                logger.warning('prompt_logprobs not found in response')
                return self._empty_result(seq_len, topk)

            for pos_entry in prompt_logprobs:
                pos_indices = []
                pos_values = []

                if pos_entry is not None:
                    top_logprobs_list = pos_entry.get('top_logprobs', [])

                    for item in top_logprobs_list[:topk]:
                        token_id = item.get('token_id')
                        logprob = item.get('logprob')
                        if token_id is not None and logprob is not None:
                            pos_indices.append(token_id)
                            pos_values.append(float(logprob))

                # Pad if needed
                while len(pos_indices) < topk:
                    pos_indices.append(0)
                    pos_values.append(float('-inf'))

                result['indices'].append(pos_indices)
                result['values'].append(pos_values)

            # Pad to seq_len if needed
            while len(result['indices']) < seq_len:
                result['indices'].append([0] * topk)
                result['values'].append([float('-inf')] * topk)

        except Exception as e:
            logger.warning(f'Failed to parse swift deploy response: {e}')
            return self._empty_result(seq_len, topk)

        return result

    def _parse_vllm_native_response(self, response: Dict[str, Any], seq_len: int, topk: int) -> Dict[str, Any]:
        """Parse vLLM native completions API response.

        vLLM returns logprobs as:
        {
            "choices": [{
                "logprobs": {
                    "top_logprobs": [{token_str: logprob, ...}, ...]
                }
            }]
        }

        Or with prompt_logprobs (if using newer vLLM):
        {
            "choices": [{
                "prompt_logprobs": [{token_id_str: {"logprob": float}, ...}, ...]
            }]
        }
        """
        result = {'indices': [], 'values': []}

        try:
            if 'choices' not in response or len(response['choices']) == 0:
                return self._empty_result(seq_len, topk)

            choice = response['choices'][0]

            # Try prompt_logprobs first (vLLM native format with token IDs as keys)
            prompt_logprobs = choice.get('prompt_logprobs')
            if prompt_logprobs is not None:
                for pos_idx, pos_logprobs in enumerate(prompt_logprobs):
                    pos_indices = []
                    pos_values = []

                    if pos_logprobs is not None:
                        # vLLM format: {token_id_str: {logprob: float, ...}, ...}
                        sorted_items = sorted(pos_logprobs.items(), key=lambda x: -self._get_logprob_value(x[1]))[:topk]

                        for token_id_str, logprob_data in sorted_items:
                            try:
                                token_id = int(token_id_str)
                                pos_indices.append(token_id)
                                pos_values.append(self._get_logprob_value(logprob_data))
                            except (ValueError, TypeError):
                                continue

                    # Pad if needed
                    while len(pos_indices) < topk:
                        pos_indices.append(0)
                        pos_values.append(float('-inf'))

                    result['indices'].append(pos_indices)
                    result['values'].append(pos_values)

                # Pad to seq_len if needed
                while len(result['indices']) < seq_len:
                    result['indices'].append([0] * topk)
                    result['values'].append([float('-inf')] * topk)

                return result

            # Fallback to logprobs.top_logprobs (OpenAI format, keys are token text)
            logprobs_data = choice.get('logprobs', {})
            if logprobs_data is None:
                return self._empty_result(seq_len, topk)

            top_logprobs_list = logprobs_data.get('top_logprobs', [])

            for pos_idx, pos_logprobs in enumerate(top_logprobs_list):
                pos_indices = []
                pos_values = []

                if pos_logprobs is not None:
                    sorted_items = sorted(pos_logprobs.items(), key=lambda x: -self._get_logprob_value(x[1]))[:topk]

                    for token_str, logprob in sorted_items:
                        try:
                            token_id = int(token_str)
                            pos_indices.append(token_id)
                            pos_values.append(self._get_logprob_value(logprob))
                        except (ValueError, TypeError):
                            # Token is text, not ID - skip (can't use without tokenizer)
                            continue

                # Pad if needed
                while len(pos_indices) < topk:
                    pos_indices.append(0)
                    pos_values.append(float('-inf'))

                result['indices'].append(pos_indices)
                result['values'].append(pos_values)

            # Pad to seq_len if needed
            while len(result['indices']) < seq_len:
                result['indices'].append([0] * topk)
                result['values'].append([float('-inf')] * topk)

        except Exception as e:
            logger.warning(f'Failed to parse vLLM native response: {e}')
            return self._empty_result(seq_len, topk)

        return result

    @staticmethod
    def _get_logprob_value(logprob) -> float:
        """Extract logprob value from response (handles both float and dict)."""
        if isinstance(logprob, (int, float)):
            return float(logprob)
        elif hasattr(logprob, 'logprob'):
            return float(logprob.logprob)
        elif isinstance(logprob, dict) and 'logprob' in logprob:
            return float(logprob['logprob'])
        return float('-inf')

    def _empty_result(self, seq_len: int, topk: int) -> Dict[str, Any]:
        """Return empty result for failed requests."""
        return {
            'indices': [[0] * topk for _ in range(seq_len)],
            'values': [[float('-inf')] * topk for _ in range(seq_len)],
        }

    def check_server_health(self, timeout: float = 60.0) -> bool:
        """Check if the teacher model server is healthy."""
        import requests
        try:
            for endpoint in ['/health', '/v1/models']:
                try:
                    response = requests.get(f'{self.base_url}{endpoint}', timeout=timeout)
                    if response.status_code == 200:
                        return True
                except requests.RequestException:
                    continue
            return False
        except Exception as e:
            logger.warning(f'Health check failed: {e}')
            return False

    def get_logprobs_sync(
        self,
        input_ids: List[List[int]],
        top_logprobs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronous wrapper for get_logprobs_batch.

        Args:
            input_ids: List of token ID sequences
            top_logprobs: Number of top logprobs to fetch

        Returns:
            Tuple of (logprobs_tensor, indices_tensor) with shapes [batch, seq_len, topk]
        """
        results = run_async(self.get_logprobs_batch(input_ids, top_logprobs))

        # Convert to tensors
        topk = top_logprobs or self.top_logprobs
        batch_size = len(input_ids)
        max_seq_len = max(len(ids) for ids in input_ids)

        logprobs_tensor = torch.full((batch_size, max_seq_len, topk), float('-inf'), dtype=torch.float32)
        indices_tensor = torch.zeros((batch_size, max_seq_len, topk), dtype=torch.long)

        for batch_idx, result in enumerate(results):
            indices = result.get('indices', [])
            values = result.get('values', [])
            for pos_idx, (pos_indices, pos_values) in enumerate(zip(indices, values)):
                if pos_idx >= max_seq_len:
                    break
                for k_idx in range(min(len(pos_indices), topk)):
                    indices_tensor[batch_idx, pos_idx, k_idx] = pos_indices[k_idx]
                    logprobs_tensor[batch_idx, pos_idx, k_idx] = pos_values[k_idx]

        return logprobs_tensor, indices_tensor


def fetch_teacher_logprobs_from_api(
    teacher_api_client: TeacherAPIClient,
    input_ids: torch.Tensor,
    topk: int,
    device: torch.device,
    is_master_rank: bool = True,
    broadcast_src: int = 0,
    group: Optional['torch.distributed.ProcessGroup'] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fetch teacher logprobs from external API service.

    This is a shared utility function used by both swift RLHF and Megatron GKD trainers.
    Only the designated rank (master/last) makes API calls, then broadcasts results.

    Note on off-by-one alignment:
        The first token's logprob may be None because there's no preceding context
        to predict it. The returned tensors will have zeros/negative infinity at
        position 0. In GKD training, this is acceptable since the loss mask
        (labels=-100) typically excludes prompt tokens.

    Args:
        teacher_api_client: The TeacherAPIClient instance (may be None on non-API ranks)
        input_ids: Input token IDs tensor [batch_size, seq_len]
        topk: Number of top-k logprobs to fetch
        device: Device for output tensors
        is_master_rank: Whether this rank should make API calls
        broadcast_src: Source rank for broadcasting results (rank within the group)
        group: Optional process group for broadcasting. If None, uses the default
            global process group. For Megatron, pass the model parallel group
            (TP×PP×CP) so that ranks processing the same data share results.

    Returns:
        Tuple of (teacher_logprobs, teacher_indices) tensors with shapes [batch, seq_len, topk]
    """
    import torch.distributed as dist

    batch_size, seq_len = input_ids.shape

    # Initialize tensors
    teacher_logprobs = torch.zeros(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    teacher_indices = torch.zeros(batch_size, seq_len, topk, device=device, dtype=torch.long)

    # Only designated rank fetches from API
    if is_master_rank and teacher_api_client is not None:
        # Fetch logprobs from API
        api_results = run_async(
            teacher_api_client.get_logprobs_batch(
                input_ids=input_ids.tolist(),
                top_logprobs=topk,
            ))

        # Parse API results into tensors
        # api_results is list of dicts with 'values' (logprobs) and 'indices' for each sample
        for batch_idx, result in enumerate(api_results):
            indices_list = result.get('indices', [])
            values_list = result.get('values', [])
            for pos_idx, (pos_indices, pos_values) in enumerate(zip(indices_list, values_list)):
                if pos_idx >= seq_len:
                    break
                for k_idx in range(min(len(pos_indices), topk)):
                    teacher_indices[batch_idx, pos_idx, k_idx] = pos_indices[k_idx]
                    teacher_logprobs[batch_idx, pos_idx, k_idx] = pos_values[k_idx]

    # Broadcast results within the process group
    if dist.is_initialized():
        # Get group size to determine if broadcast is needed
        group_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
        if group_size > 1:
            dist.broadcast(teacher_logprobs, src=broadcast_src, group=group)
            dist.broadcast(teacher_indices, src=broadcast_src, group=group)

    return teacher_logprobs, teacher_indices
