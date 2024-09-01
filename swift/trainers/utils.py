# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import heapq
import inspect
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset as HfDataset
from torch.nn import Module
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                                        SchedulerType)

from swift.llm.utils.template import Context, History, Template
from swift.utils import get_logger

try:
    # https://github.com/huggingface/transformers/pull/25702
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    ShardedDDPOption = None

logger = get_logger()


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function


def is_instance_of_ms_model(model: Module) -> bool:
    """avoid import modelscope: circular dependency problem"""
    for m_cls in model.__class__.__mro__:
        cls_name = m_cls.__name__
        cls_module = m_cls.__module__
        if cls_name == 'Model' and cls_module.startswith('modelscope'):
            return True
    return False


def concat_template(feature: Dict, template: Template):
    query: Optional[str] = feature.get('query', None)
    system: Optional[str] = feature.get('system', None)
    history: Optional[History] = feature.get('history', None)
    if history is None:
        history = []
    if system is None:
        if template.use_default_system:
            system = template.default_system
    else:
        assert template.system_prefix is not None, 'not support `system`'
    res_context_list: List[Context] = []
    compute_loss_idx: List[float] = []
    if system is None:
        assert template.prefix != template.system_prefix, f'template.prefix: {template.prefix}'
        prefix = template.prefix
    else:
        prefix = template.system_prefix
    template._concat_context_list(prefix, res_context_list, compute_loss_idx, system=system)
    for i, (q, r) in enumerate(history):
        template._concat_context_list(
            [
                *template.prompt,
                '{{RESPONSE}}',
                *template.chat_sep  # noqa
            ],
            res_context_list,
            compute_loss_idx,
            query=q,
            response=r,
            round0=i)  # noqa
    template._concat_context_list(template.prompt, res_context_list, compute_loss_idx, query=query, round0=len(history))
    res_context_list, compute_loss_idx = template._simplify_context_list(res_context_list, compute_loss_idx)

    return res_context_list, feature['response'], feature['rejected_response'], compute_loss_idx


def build_tokenized_answer(answer, template: Template):
    tgt_input_ids = template._encode_context_list([answer], [1.0])[0]
    tgt_input_ids += template._encode_context_list(template.suffix, [1.0])[0]
    return dict(
        input_ids=tgt_input_ids,
        attention_mask=[1] * len(tgt_input_ids),
    )


def sort_by_max_length(dataset: HfDataset, num_dataset: int, is_encoder_decoder: bool = False) -> HfDataset:
    logger.info('sort by max length...')
    if not is_encoder_decoder:
        dataset_chosen_len = [len(d['chosen_input_ids']) for d in dataset]
        dataset_rejected_len = [len(d['rejected_input_ids']) for d in dataset]
        idx = heapq.nlargest(
            num_dataset,
            range(len(dataset_chosen_len)),
            key=lambda i: max(dataset_chosen_len[i], dataset_rejected_len[i]))
    else:
        dataset_len = [len(d['prompt_input_ids']) for d in dataset]
        idx = heapq.nlargest(num_dataset, range(len(dataset_len)), key=lambda i: dataset_len[i])
    return dataset.select(idx)


def patch_trl(is_vision_model: bool = False):
    from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
    from transformers import trainer

    trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
    trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
    trainer.PrinterCallback = PrinterCallbackNew

    # fix encoder-decoder error
    if is_vision_model:
        patch_datacollator()
        patch_dataset_map()

    patch_itds_map()


def patch_datacollator():
    import torch
    from typing import Any, Dict, List
    from trl.trainer.utils import DPODataCollatorWithPadding, pad
    if not hasattr(DPODataCollatorWithPadding, '_old_call'):  # Avoid double patching
        from torch.nn.utils.rnn import pad_sequence
        from functools import wraps

        old_call = DPODataCollatorWithPadding.__call__

        @wraps(old_call)
        def new_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            padded_batch = {}
            for k in features[0].keys():
                if k.endswith(('_input_ids', '_attention_mask', '_labels', '_pixel_values', '_images')):
                    if self.is_encoder_decoder:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]

                        if (k.startswith('prompt')) and (k.endswith('input_ids')):
                            if self.pad_token_id is None:
                                raise ValueError(
                                    'Padding is enabled, but the tokenizer is not configured with a padding token.'
                                    ' Explicitly set `tokenizer.pad_token`'
                                    ' (e.g. `tokenizer.pad_token = tokenizer.eos_token`)'
                                    ' before calling the trainer.')
                            padding_value = self.pad_token_id
                        elif k.endswith('_attention_mask'):
                            padding_value = 0
                        elif k.startswith(('chosen', 'rejected', 'completion')) or ('decoder' in k):
                            padding_value = self.label_pad_token_id
                        # patch here
                        elif k.endswith('_pixel_values'):
                            padding_value = 0
                        else:
                            raise ValueError(f"Unexpected key in batch '{k}'")
                        padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    else:
                        # Set padding value based on the key
                        if k.endswith('_input_ids'):
                            if self.pad_token_id is None:
                                raise ValueError(
                                    'Padding is enabled, but the tokenizer is not configured with a padding token.'
                                    ' Explicitly set `tokenizer.pad_token`'
                                    ' (e.g. `tokenizer.pad_token = tokenizer.eos_token`)'
                                    ' before calling the trainer.')
                            padding_value = self.pad_token_id
                        elif k.endswith('_labels'):
                            padding_value = self.label_pad_token_id
                        elif k.endswith('_attention_mask'):
                            padding_value = 0
                        elif k.endswith(('_pixel_values', '_images')):
                            padding_value = 0
                        else:
                            raise ValueError(f"Unexpected key in batch '{k}'")

                        # Set padding side based on the key
                        if k in ['prompt_input_ids', 'prompt_attention_mask']:
                            padding_side = 'left'
                        else:
                            padding_side = 'right'

                        # Set the dtype
                        if k.endswith(('_pixel_values', '_images')):
                            dtype = torch.float32  # will be downcasted if necessary by the Trainer
                        else:
                            dtype = torch.int64

                        # Convert to tensor and pad
                        to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                        padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
                elif k.endswith('_logps'):
                    # the cached reference model logprobs
                    padded_batch[k] = torch.tensor([ex[k] for ex in features])
                else:
                    padded_batch[k] = [ex[k] for ex in features]

            return padded_batch

        DPODataCollatorWithPadding.__call__ = new_call
        DPODataCollatorWithPadding._old_call = old_call


def patch_itds_map():
    # resolve conflict with `num_proc` in iterable_dataset map func
    from datasets import IterableDataset
    from functools import wraps

    if not hasattr(IterableDataset, '_old_map'):  # Avoid double patching
        old_map = IterableDataset.map

        @wraps(old_map)
        def new_map(self, *args, **kwargs):
            kwargs.pop('num_proc', None)
            kwargs.pop('writer_batch_size', None)
            return old_map(self, *args, **kwargs)

        IterableDataset.map = new_map
        IterableDataset._old_map = old_map


def patch_dataset_map():
    original_map = HfDataset.map
    if not hasattr(HfDataset, '_old_map'):

        def patched_map(self, function, **kwargs):
            if 'writer_batch_size' not in kwargs:
                kwargs['writer_batch_size'] = 10
            return original_map(self, function, **kwargs)

        HfDataset.map = patched_map
        HfDataset._old_map = original_map


def tokenize_row(feature: dict[str, List[Any]], template: Template, **kwargs) -> Dict:
    is_encoder_decoder = kwargs.get('is_encoder_decoder', False)
    # TODO: get _data_keys
    _data_keys = kwargs.get('_data_keys', None)
    max_length = kwargs.get('max_length', 2048)
    truncation_mode = kwargs.get('truncation_mode', 'keep_start')
    max_prompt_length = kwargs.get('max_prompt_length', 512)
    label_pad_token_id = kwargs.get('label_pad_token_id', -100)
    s
    batch = {}
    if not is_encoder_decoder:
        # encode without response
        prompt = feature.copy()
        prompt['response'] = None
        prompt_tokens = template.encode(prompt)[0]

        if 'input_ids' not in prompt_tokens:
            raise Exception('Detect too lengthy prompt, please consider set larger max_length ')

        # for MLLM, pop vision related data to process after
        if prompt_tokens.get('_data', None) is not None:
            for key in prompt_tokens['_data'].keys():
                if key not in prompt_tokens:
                    prompt_tokens[key] = prompt_tokens['_data'][key]
            prompt_tokens.pop('_data')

        prompt_tokens.pop('labels', None)

        # convert bfloat16 to float32 to avoid conflict in mapping
        if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
            prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)

        if 'images' in prompt_tokens and prompt_tokens['images'].dtype == torch.bfloat16:
            prompt_tokens['images'] = prompt_tokens['images'].to(torch.float32)

        if 'attention_mask' not in prompt_tokens:
            prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

        prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

        # encode with response
        chosen_tokens = build_tokenized_answer(feature['response'], template)
        chosen_tokens.update(prompt_tokens)

        rejected_tokens = build_tokenized_answer(feature['rejected_response'], template)
        rejected_tokens.update(prompt_tokens)

        longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens['prompt_input_ids']) + longer_response_length > max_length:
                if truncation_mode == 'keep_start':
                    for k in ['prompt_input_ids', 'prompt_attention_mask']:
                        answer_tokens[k] = answer_tokens[k][:max_prompt_length]
                elif truncation_mode == 'keep_end':
                    for k in ['prompt_input_ids', 'prompt_attention_mask']:
                        answer_tokens[k] = answer_tokens[k][-max_prompt_length:]
                else:
                    raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens['prompt_input_ids']) + longer_response_length > max_length:
                for k in ['input_ids', 'attention_mask']:
                    answer_tokens[k] = answer_tokens[k][:max_length - max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f'prompt_{k}'] + chosen_tokens[k]
            for k in ['input_ids', 'attention_mask']
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f'prompt_{k}'] + rejected_tokens[k]
            for k in ['input_ids', 'attention_mask']
        }

        chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
        _paddings = [label_pad_token_id] * len(chosen_tokens['prompt_input_ids'])
        chosen_sequence_tokens['labels'][:len(chosen_tokens['prompt_input_ids'])] = _paddings
        rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
        _paddings = [label_pad_token_id] * len(rejected_tokens['prompt_input_ids'])
        rejected_sequence_tokens['labels'][:len(rejected_tokens['prompt_input_ids'])] = _paddings

        for k, toks in {
                'chosen_': chosen_sequence_tokens,
                'rejected_': rejected_sequence_tokens,
                '': prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == 'token_type_ids':
                    continue
                batch[f'{k}{type_key}'] = tokens

    else:
        # encoder-decoder
        prompt = feature.copy()
        prompt['response'] = None
        prompt_tokens = template.encode(prompt)[0]

        if prompt_tokens.get('_data', None) is not None:
            for key in prompt_tokens['_data'].keys():
                if key not in prompt_tokens:
                    prompt_tokens[key] = prompt_tokens['_data'][key]
            prompt_tokens.pop('_data')

        prompt_tokens.pop('labels', None)

        if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
            # datasets do not accept bfloat16; convert to float32.
            prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)
        if 'attention_mask' not in prompt_tokens:
            prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

        prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

        # encode with response
        chosen_tokens = build_tokenized_answer(feature['response'], template)
        rejected_tokens = build_tokenized_answer(feature['rejected_response'], template)

        batch['chosen_labels'] = chosen_tokens['input_ids']
        batch['rejected_labels'] = rejected_tokens['input_ids']

        if model is not None and hasattr(model, 'prepare_decoder_input_ids_from_labels'):
            batch['rejected_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch['rejected_labels']))
            batch['chosen_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch['chosen_labels']))

        batch.update(prompt_tokens)
    return batch
