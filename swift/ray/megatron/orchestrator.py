# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI entry point for Ray-based Megatron distributed training.

Routes to the appropriate Driver class based on ``rlhf_type``.

Usage::

    python -m swift.ray.megatron.orchestrator \
        --config examples/megatron/rlhf/dpo/ray.yaml
"""
from typing import Any, Dict, List, Optional, Tuple, Type

from swift.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Registry: training method -> (driver class, required group names)
# ---------------------------------------------------------------------------

_TRAINING_REGISTRY: Dict[str, Tuple[str, List[str]]] = {
    'dpo': ('swift.ray.megatron.driver.DPODriver', ['train', 'ref']),
    'kto': ('swift.ray.megatron.driver.DPODriver', ['train', 'ref']),
    # Future:
    # 'grpo': ('swift.ray.megatron.driver.GRPODriver', ['train', 'rollout']),
    # 'gkd': ('swift.ray.megatron.driver.GKDDriver', ['train', 'rollout']),
}


def get_driver_cls(rlhf_type: str) -> Type:
    """Resolve driver class from the registry."""
    entry = _TRAINING_REGISTRY.get(rlhf_type)
    if entry is None:
        raise ValueError('No driver registered for rlhf_type=%r. '
                         'Available: %s' % (rlhf_type, list(_TRAINING_REGISTRY.keys())))
    cls_path = entry[0]
    module_path, cls_name = cls_path.rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def get_required_groups(rlhf_type: str) -> List[str]:
    """Return the required group names for a training method."""
    entry = _TRAINING_REGISTRY.get(rlhf_type)
    if entry is None:
        raise ValueError('Unknown rlhf_type: %r' % rlhf_type)
    return list(entry[1])


def load_ray_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return raw


def _build_group_argv(
    shared_config: Dict[str, Any],
    group_config: Dict[str, Any],
) -> List[str]:
    """Build argv list for a single group by merging shared and group-specific config.

    Group-specific keys take precedence. The 'gpus' key is excluded
    since it is consumed by the driver, not by MegatronArguments.
    """
    merged = dict(shared_config)
    merged.update(group_config)

    skip_keys = {'gpus', 'colocate_groups'}
    argv = []
    for key, value in merged.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            if value:
                argv.append('--%s' % key)
                argv.append('true')
            else:
                argv.append('--%s' % key)
                argv.append('false')
        elif isinstance(value, (list, tuple)):
            argv.append('--%s' % key)
            argv.extend(str(v) for v in value)
        elif isinstance(value, dict):
            import json
            argv.append('--%s' % key)
            argv.append(json.dumps(value))
        elif value is None:
            continue
        else:
            argv.append('--%s' % key)
            argv.append(str(value))
    return argv


def parse_ray_config(config_path: str) -> Dict[str, Any]:
    """Parse a Ray YAML config into driver constructor kwargs.

    Returns a dict with:
        - ``rlhf_type``: str
        - ``group_argv``: dict mapping group name -> argv list
        - ``group_gpus``: dict mapping group name -> int
        - ``colocate_groups``: list of lists (groups sharing GPUs)
    """
    raw = load_ray_config(config_path)

    rlhf_type = raw.pop('rlhf_type', 'dpo')
    required_groups = get_required_groups(rlhf_type)
    colocate_groups = raw.pop('colocate_groups', [])

    group_sections = {}
    for group_name in required_groups:
        if group_name in raw:
            group_sections[group_name] = raw.pop(group_name)
        else:
            group_sections[group_name] = {}

    # Remaining top-level keys are shared config
    shared_config = dict(raw)

    group_argv = {}
    group_gpus = {}
    for group_name, group_cfg in group_sections.items():
        group_gpus[group_name] = group_cfg.pop('gpus', 0)
        group_argv[group_name] = _build_group_argv(shared_config, group_cfg)

    return {
        'rlhf_type': rlhf_type,
        'group_argv': group_argv,
        'group_gpus': group_gpus,
        'colocate_groups': colocate_groups,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main():
    """CLI entry point.

    Reads ``--config path.yaml`` and launches the appropriate driver.
    """
    import sys
    argv = sys.argv[1:]
    config_path = None
    for i, arg in enumerate(argv):
        if arg == '--config' and i + 1 < len(argv):
            config_path = argv[i + 1]
            break

    if config_path is None:
        raise ValueError('Usage: python -m swift.ray.megatron.orchestrator '
                         '--config <path.yaml>')

    parsed = parse_ray_config(config_path)
    rlhf_type = parsed['rlhf_type']

    driver_cls = get_driver_cls(rlhf_type)

    driver = driver_cls(
        group_argv=parsed['group_argv'],
        group_gpus=parsed['group_gpus'],
        colocate_groups=parsed['colocate_groups'],
    )
    return driver.run()


if __name__ == '__main__':
    main()
