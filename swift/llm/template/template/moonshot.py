# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from transformers.utils import strtobool

from swift.utils import get_current_device
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class MoonlightTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda:
                           ['<|im_user|>user<|im_middle|>{{QUERY}}<|im_end|><|im_assistant|>assistant<|im_middle|>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|im_system|>system<|im_middle|>{{SYSTEM}}<|im_end|>'])
    default_system: str = 'You are a helpful assistant'


register_template(MoonlightTemplateMeta(LLMTemplateType.moonlight))


class KimiVLTemplate(Template):
    placeholder_tokens = ['<|media_pad|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|media_start|>image<|media_content|><|media_pad|><|media_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        media_token = self._tokenize('<|media_pad|>')[0]
        idx_list = findall(input_ids, media_token)
        if inputs.images:
            image_processor = self.processor.image_processor
            image_inputs = image_processor(inputs.images, return_tensors='pt')
            image_grid_hws = image_inputs['image_grid_hws']
            merge_length = image_processor.merge_kernel_size[0] * image_processor.merge_kernel_size[1]

            def _get_new_tokens(i):
                token_len = (image_grid_hws[i].prod() // merge_length)
                return [media_token] * token_len

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded.update(image_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        image_grid_hws = self.concat_tensor(batch, 'image_grid_hws', 0)
        if image_grid_hws is not None:
            res['image_grid_hws'] = image_grid_hws
        return res


register_template(MoonlightTemplateMeta(MLLMTemplateType.kimi_vl, template_cls=KimiVLTemplate))


class KimiATemplate(Template):
    kimia_text_audiodelaytokens = 6
    kimia_user_msg_start = 151670
    kimia_text_blank = '<|im_kimia_text_blank|>'  # 151666
    kimia_assistant_msg_start = 151671
    media_begin = 151661
    media_end = 151663
    msg_end = 151645
    kimia_speech_ct_id = '<|im_kimia_speech_ct_id|>'

    # text encode 时，audio 部分增加同长的kimia_text_blank
    # text audio 时， text 部分增加speech_tokens同长（+2） kimia_text_blank
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from kimia_infer.api.prompt_manager import KimiAPromptManager
        use_hf = strtobool(os.environ.get('USE_HF', '0'))
        from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
        if use_hf:
            from huggingface_hub import snapshot_download
            audio_tokenizer_model = 'THUDM/glm-4-voice-tokenizer'
        else:
            from modelscope import snapshot_download
            audio_tokenizer_model = 'ZhipuAI/glm-4-voice-tokenizer'
        origin_init = KimiAPromptManager.__init__

        def __init__(self, *args, **kwargs):
            self.audio_tokenizer = Glm4Tokenizer(audio_tokenizer_model)
            self.audio_tokenizer = self.audio_tokenizer.to(get_current_device())

        self.kimia_token_offset = model_config.kimia_token_offset

        model_info = self.model_info._name_or_path
        self.prompt_manager = KimiAPromptManager(model_info._name_or_path, kimia_token_offset=152064)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        audios = inputs.audios
        if audios:
            for audio in audios:
                speech_tokens = self.audio_tokenizer.tokenize(audio_path=audio)
                speech_tokens = speech_tokens + self.kimia_token_offset
                speech_tokens = speech_tokens.squeeze(0).cpu().numpy().tolist()
                audio_features = 0
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(MoonlightTemplateMeta(MLLMTemplateType.kimi_vl, template_cls=KimiVLTemplate))
