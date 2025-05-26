# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from transformers.utils import strtobool

from swift.llm import TemplateType
from swift.utils import get_device
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ModelInfo, git_clone_github

register_model(
    ModelMeta(
        LLMModelType.moonlight,
        [
            ModelGroup([
                Model('moonshotai/Moonlight-16B-A3B', 'moonshotai/Moonlight-16B-A3B'),
                Model('moonshotai/Moonlight-16B-A3B-Instruct', 'moonshotai/Moonlight-16B-A3B-Instruct'),
            ]),
        ],
        TemplateType.moonlight,
        get_model_tokenizer_with_flash_attn,
        architectures=['DeepseekV3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers<4.49'],
    ))


def get_model_tokenizer_kimi_vl(*args, **kwargs):
    model, processor = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        patch_output_clone(model.language_model.model.embed_tokens)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.kimi_vl,
        [
            ModelGroup([
                Model('moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Instruct'),
                Model('moonshotai/Kimi-VL-A3B-Thinking', 'moonshotai/Kimi-VL-A3B-Thinking'),
            ])
        ],
        TemplateType.kimi_vl,
        get_model_tokenizer_kimi_vl,
        architectures=['KimiVLForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers<4.49'],
    ))


def get_model_tokenizer_kimia(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):

    model, text_tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model,
                                                                **kwargs)

    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/MoonshotAI/Kimi-Audio')
    sys.path.append(local_repo_path)
    from kimia_infer.api.kimia import KimiAudio
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.speech_encoder = os.path.join(model_dir, 'large-v3.pt')
    if not os.path.exists(model_config.speech_encoder):
        whisper.load_model('large-v3', download_root=model_dir)
    kwargs['automodel_class'] = KimiAudio
    kwargs['model_config'] = model_config
    for key in ['forward', 'generate']:
        try:
            delattr(OmniSpeech2SLlamaForCausalLM, key)
            delattr(OmniSpeechLlamaForCausalLM, key)
        except AttributeError:
            pass
    # not support device_map='auto'
    device_map = model_kwargs['device_map']
    model_kwargs['device_map'] = None
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model:
        model.to(get_device() if device_map == 'auto' else device_map)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.kimi_audio,
        [
            ModelGroup([
                Model('moonshotai/Kimi-Audio-7B', 'moonshotai/Kimi-Audio-7B'),
                Model('moonshotai/Kimi-Audio-7B-Instruct', 'moonshotai/Kimi-Audio-7B-Instruct'),
            ])
        ],
        TemplateType.kimi_vl,
        get_model_tokenizer_kimia,
        architectures=['MoonshotKimiaForCausalLM'],
        model_arch=ModelArch.llava_hf,
        requires=['sox', 'conformer', 'openai-whisper', 'librosa', 'diffusers', 'hyperpyyaml', 'torchdyn'],
    ))
