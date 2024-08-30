from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from trl import RewardTrainer as HFRewardTrainer
from trl.trainer.utils import pad_to_length

from swift.utils import get_logger
from .mixin import PushToMsHubMixin, SwiftMixin
from .utils import build_tokenized_answer, patch_trl, sort_by_max_length, patch_valuehead_model
from swift.llm.utils.template import Template
from accelerate import PartialState

logger = get_logger()
from trl import AutoModelForCausalLMWithValueHead

class RewardTrainer(PushToMsHubMixin, SwiftMixin, HFRewardTrainer):

    def __init__(self, *args, template: Template, test_oom_error=False, **kwargs):
        kwargs['model'] = patch_valuehead_model(AutoModelForCausalLMWithValueHead.from_pretrained(kwargs['model']))
        
        
        self.template = template
        template._is_training = True
        self.streaming = kwargs.pop('streaming')
        is_vision = kwargs.pop('is_vision')
        patch_trl(is_vision)
        self.processed_keys = []  # keys after tokenize_row mapiing
        self.column_names = list(next(iter(kwargs.get('train_dataset'))).keys())
        self._data_keys = []  # vision related key in _data
        self.need_filter: bool = False
        with PartialState().local_main_process_first():
            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc, writer_batch_size=10)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row, num_proc=self.dataset_num_proc, writer_batch_size=10
                )
        super().__init__(*args, **kwargs)
        # remove origin columns to resolve conflit in streaming mode
        self.train_dataset = self.train_dataset.remove_columns(self.column_names)
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.remove_columns(self.column_names)

        if self.need_filter:
            self.train_dataset = self.train_dataset.filter(lambda x: x['prompt_input_ids'] is not None)
            if self.eval_dataset is not None:
                self.eval_dataset = self.eval_dataset.filter(lambda x: x['prompt_input_ids'] is not None)
        if not self.streaming:
            train_ds_info = self.stat_dataset(self.train_dataset, self.is_encoder_decoder)

            if self.eval_dataset is not None:
                val_ds_info = self.stat_dataset(self.eval_dataset, self.is_encoder_decoder)
                self.dataset_info = {'train_dataset': train_ds_info, 'val_dataset': val_ds_info}
            else:
                self.dataset_info = {'train_dataset': train_ds_info}
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000, self.is_encoder_decoder)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time': 0.,
            'gen_len': 0,
            'memory': {},
            'model': self.model.get_trainable_parameters() if hasattr(self.model, 'get_trainable_parameters') else None,
        }
        # modify after init
        self.is_vision_model = is_vision
        self.model.config.model_type = self.model.config.model_type[:-1]  # remove suffix
        
    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:

        batch = {}
        if not self.is_encoder_decoder:
            # encode without response
            prompt = feature.copy()
            prompt['response'] = None
            prompt_tokens = self.template.encode(prompt)[0]
            prompt_tokens.pop('labels', None)
            # Skip examples that have too lengthy prompt to avoid conflict in following processing
            if 'input_ids' not in prompt_tokens:
                self.need_filter = True
                return {k: None for k in self.processed_keys}

            # for MLLM, pop vision related data to process after
            if '_data' in prompt_tokens:
                if not self._data_keys:
                    self._data_keys = prompt_tokens['_data'].keys()
                for key in prompt_tokens['_data'].keys():
                    if key not in prompt_tokens:
                        prompt_tokens[key] = prompt_tokens['_data'][key]
                prompt_tokens.pop('_data')

            # convert bfloat16 to float32 to avoid conflict in mapping
            if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
                prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)

            if 'attention_mask' not in prompt_tokens:
                prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

            prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

            # encode with response
            chosen_tokens = build_tokenized_answer(feature['response'], self.template)
            chosen_tokens.update(prompt_tokens)

            rejected_tokens = build_tokenized_answer(feature['rejected_response'], self.template)
            rejected_tokens.update(prompt_tokens)

            longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens['prompt_input_ids']) + longer_response_length > self.max_length:
                    if self.truncation_mode == 'keep_start':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[k][:self.max_prompt_length]
                    elif self.truncation_mode == 'keep_end':
                        for k in ['prompt_input_ids', 'prompt_attention_mask']:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f'Unknown truncation mode: {self.truncation_mode}')

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens['prompt_input_ids']) + longer_response_length > self.max_length:
                    for k in ['input_ids', 'attention_mask']:
                        answer_tokens[k] = answer_tokens[k][:self.max_length - self.max_prompt_length]

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
            _paddings = [self.label_pad_token_id] * len(chosen_tokens['prompt_input_ids'])
            chosen_sequence_tokens['labels'][:len(chosen_tokens['prompt_input_ids'])] = _paddings
            rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
            _paddings = [self.label_pad_token_id] * len(rejected_tokens['prompt_input_ids'])
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
            prompt_tokens = self.template.encode(prompt)[0]
            prompt_tokens.pop('labels', None)

            if '_data' in prompt_tokens:
                if not self._data_keys:
                    self._data_keys = prompt_tokens['_data'].keys()
                for key in prompt_tokens['_data'].keys():
                    if key not in prompt_tokens:
                        prompt_tokens[key] = prompt_tokens['_data'][key]
                prompt_tokens.pop('_data')

            if 'pixel_values' in prompt_tokens and prompt_tokens['pixel_values'].dtype == torch.bfloat16:
                # datasets do not accept bfloat16; convert to float32.
                prompt_tokens['pixel_values'] = prompt_tokens['pixel_values'].to(torch.float32)
            if 'attention_mask' not in prompt_tokens:
                prompt_tokens['attention_mask'] = [1] * len(prompt_tokens['input_ids'])

            prompt_tokens = {f'prompt_{k}': v for k, v in prompt_tokens.items()}

            # encode with response
            chosen_tokens = build_tokenized_answer(feature['response'], self.template)
            rejected_tokens = build_tokenized_answer(feature['rejected_response'], self.template)

            batch['chosen_labels'] = chosen_tokens['input_ids']
            batch['rejected_labels'] = rejected_tokens['input_ids']

            if model is not None and hasattr(model, 'prepare_decoder_input_ids_from_labels'):
                batch['rejected_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch['rejected_labels']))
                batch['chosen_decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch['chosen_labels']))

            batch.update(prompt_tokens)
        if not self.processed_keys:
            self.processed_keys = (list(batch.keys()))
        return batch
        """
        input_ids_chosen
        input_ids_rejected
        attention_mask_chosen
        attention_mask_rejected
        """
    
    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # code from trl, we patch here to make compatible with mllm
        model_kwargs = {}
        if self.is_vision_model:
            # Here, we restore the _data, processing image information within the forward hook of the model.
            batch_size = inputs["input_ids_chosen"].shape[0]
            if self._data_keys:
                _data = [dict() for _ in range(batch_size)]
                for k in self._data_keys:
                    if k == 'input_ids':
                        _data = [{**d, k: inputs['concatenated_input_ids'][i]} for i, d in enumerate(_data)]
                    elif k == 'pixel_values':
                        # convert the dtype of the pixel values that may be converted to float32 in tokenize_row
                        model_dtype = self.accelerator.unwrap_model(model).dtype
                        # for vision related data, paired response share the same one
                        _data = [{**d, k: inputs[k][i // 2].to(model_dtype)} for i, d in enumerate(_data)]
                    else:
                        _data = [{**d, k: inputs[k][i // 2]} for i, d in enumerate(_data)]
                    model_kwargs['_data'] = _data

            if 'images' in inputs:
                model_kwargs['images'] = inputs['images']

        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
            **model_kwargs,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
            **model_kwargs,
        )["logits"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss
    @staticmethod
    def stat_dataset(llm_dataset, is_encoder_decoder: bool = False) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            if is_encoder_decoder:
                prompt = llm_dataset['prompt_input_ids']
                chosen = llm_dataset['chosen_labels']
                rejected = llm_dataset['chosen_labels']
                for p, cc, rr in zip(prompt, chosen, rejected):
                    _token_len.append(max(len(cc), len(rr)) + len(p))
            else:
                chosen = llm_dataset['chosen_input_ids']
                rejected = llm_dataset['rejected_input_ids']
                for cc, rr in zip(chosen, rejected):
                    _token_len.append(max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                if is_encoder_decoder:
                    _token_len.append(
                        max(len(d['chosen_labels']), len(d['chosen_labels'])) + len(d['prompt_input_ids']))
                else:
                    _token_len.append(max(len(d['chosen_input_ids']), len(d['rejected_input_ids'])))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str
