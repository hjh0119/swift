# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch

from swift.utils import upper_bound
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall

@dataclass
class KimiVLTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])


class KimiVLTemplate(Template):
    boi_token_id = 255999
    placeholder_tokens = ['<start_of_image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<start_of_image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        encoded = super()._encode(inputs)
        if inputs.images:
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self.tokenizer.encode(self.processor.full_image_sequence)
            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, lambda _: img_tokens)

            # TODO: customize
            processor_kwargs = Gemma3ProcessorKwargs._defaults['images_kwargs']
            image_inputs = self.processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(np.array(image_inputs['pixel_values']))
            image_inputs.pop('num_crops')

            array_ids = np.array(input_ids)
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[array_ids == self.processor.image_token_id] = 1
            encoded['token_type_ids'] = mm_token_type_ids.tolist()
            encoded['input_ids'] = input_ids
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['labels'] = labels
        return encoded


register_template(KimiVLTemplateMeta(MLLMTemplateType.gemma3_vision, template_cls=KimiVLTemplate))

register_template(
    TemplateMeta(
        MLLMTemplateType.kimi_vl,
        prefix=[],
        system_prefix=['<|im_system|>system<|im_middle|>{{SYSTEM}}<|im_end|>'],
        prompt=['<|im_user|>user<|im_middle|>{{QUERY}}<|im_end|><|im_assistant|>assistant<|im_middle|>'],
        chat_sep=['<|im_end|>'],
        suffix=['<|im_end|>'],
        default_system='You are a helpful assistant',
        template_cls=KimiVLTemplate,
    ))
