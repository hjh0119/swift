# Copyright (c) Alibaba, Inc. and its affiliates.

# code borrowed from openr1
# https://github.com/huggingface/open-r1/blob/main/src/open_r1/utils/code_providers.py
import abc
import asyncio
from typing import List, Optional

from ..utils import is_e2b_available, is_morph_available


if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox
    from e2b_code_interpreter.models import Execution

    from .routed_sandbox import RoutedSandbox
else:
    AsyncSandbox = None
    Execution = None
    RoutedSandbox = None

if is_morph_available():
    from morphcloud.api import MorphCloudClient
    from morphcloud.sandbox import Sandbox

    from .routed_morph import RoutedMorphSandbox
else:
    MorphCloudClient = None
    Sandbox = None
    RoutedMorphSandbox = None


class CodeExecutionProvider(abc.ABC):
    """Abstract base class for code execution providers."""

    @abc.abstractmethod
    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute multiple scripts and return their reward values.

        Args:
            scripts: List of code scripts to execute
            language: The programming language of the scripts

        Returns:
            List of float rewards (one per script)
        """
        pass

class E2BProvider(CodeExecutionProvider):
    """Provider that executes code using E2B sandboxes."""

    def __init__(self, num_parallel: int = 2, e2b_router_url: Optional[str] = None):
        """Initialize the E2B provider.

        Args:
            num_parallel: Number of parallel sandboxes to use
            e2b_router_url: URL for the E2B router (if using router mode)
        """
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this provider. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )

        self.num_parallel = num_parallel
        self.e2b_router_url = e2b_router_url

    def execute_scripts(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Execute scripts using E2B sandboxes.

        If e2b_router_url is provided, uses the RoutedSandbox for batch processing.
        Otherwise, uses direct AsyncSandbox with parallelization.
        """
        if self.e2b_router_url is not None:
            routed_sandbox = RoutedSandbox(router_url=self.e2b_router_url)

            executions = routed_sandbox.run_code(
                scripts=scripts,
                languages=languages,
                timeout=30,
                request_timeout=28,
            )

            rewards = []
            for execution in executions:
                try:
                    reward = float(execution.text)
                    rewards.append(reward)
                except Exception:
                    rewards.append(None)
            return rewards

        try:
            rewards = self._run_async_from_sync(scripts, languages, self.num_parallel)
        except Exception as e:
            print(f"Error from E2B executor: {e}")
            rewards = [0.0] * len(scripts)

        return rewards

    def _run_async_from_sync(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        """Function wrapping the `_run_async` function."""
        try:
            rewards = asyncio.run(self._run_async(scripts, languages, num_parallel))
        except Exception as e:
            print(f"Error from E2B executor async: {e}")
            raise e

        return rewards

    async def _run_async(self, scripts: List[str], languages: List[str], num_parallel: int) -> List[float]:
        semaphore = asyncio.Semaphore(num_parallel)

        tasks = [self._run_script(script, languages, semaphore) for script in scripts]

        results = await asyncio.gather(*tasks)
        rewards = list(results)

        return rewards

    async def _run_script(self, script: str, languages: List[str], semaphore: asyncio.Semaphore) -> float:
        # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
        # These values are based on running 256 examples with the gold solution
        # from open-r1/verifiable-coding-problems-python_decontaminated
        # see scripts/benchmark_e2b.py

        SANDBOX_TIMEOUT = 30
        MARGIN = 2
        REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

        async with semaphore:
            try:
                sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
                execution = await asyncio.wait_for(
                    sandbox.run_code(script, languages=languages),
                    timeout=ASYNCIO_TIMEOUT,
                )
                return float(execution.text)
            except (TypeError, ValueError):
                return 0.0
            except asyncio.TimeoutError:
                print("Operation timed out")
                return 0.0
            except Exception as e:
                print(f"Error in `_run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
                return 0.0
            finally:
                try:
                    await sandbox.kill()
                except Exception as e:
                    print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")
