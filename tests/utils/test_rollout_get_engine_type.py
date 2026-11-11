"""get_engine_type reports the truncation config from args.get_template_kwargs.

The engine template lives in the worker process, so the main process reuses the
args-side kwargs (which normalize ``delete`` to ``raise``) instead of a template.
"""
import asyncio
from types import SimpleNamespace

from swift.arguments import RolloutArguments
from swift.pipelines.infer.rollout import SwiftRolloutDeploy


def test_get_engine_type_normalizes_delete_to_raise():
    args = RolloutArguments(model='Qwen/Qwen2.5-0.5B-Instruct', truncation_strategy='delete')
    fake = SimpleNamespace(args=args, use_async_engine=False, use_gym_env=False)

    engine_type = asyncio.run(SwiftRolloutDeploy.get_engine_type(fake))

    assert args.truncation_strategy == 'delete'  # raw args keep the user value
    assert engine_type['truncation_strategy'] == 'raise'  # normalized like the training template
    assert engine_type['max_length'] == args.max_length
