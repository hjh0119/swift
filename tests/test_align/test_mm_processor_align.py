import copy

import torch
from swift.model import get_processor
from swift.template import get_template

try:
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import nested_tensors_equal
except ImportError:
    ModelConfig = None
    MULTIMODAL_REGISTRY = None
    nested_tensors_equal = None

CAT_IMAGE = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'
BABY_VIDEO = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'
WEATHER_AUDIO = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'

_SKIP_TRAIN_KEYS = frozenset({'input_ids', 'labels', 'loss_scale'})


def _image_sample():
    return {
        'messages': [{'role': 'user', 'content': '<image>describe the image.'}],
        'images': [CAT_IMAGE],
    }


def _video_sample():
    return {
        'messages': [{'role': 'user', 'content': '<video>describe the video.'}],
        'videos': [BABY_VIDEO],
    }


def _audio_sample():
    return {
        'messages': [{'role': 'user', 'content': 'describe the audio.'}],
        'audios': [WEATHER_AUDIO],
    }


def _as_list_ids(x):
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).tolist()
    return list(x)


def _build_mm_data(vllm_encoded):
    mm_data = {}
    for plural, singular in [('images', 'image'), ('videos', 'video'), ('audios', 'audio')]:
        data = vllm_encoded.get(plural)
        if not data:
            continue
        if len(data) == 1 and not isinstance(data[0], tuple):
            mm_data[singular] = data[0]
        else:
            mm_data[singular] = data
    return mm_data


def _swift_train_kwargs(template, sample):
    train_template = copy.deepcopy(template)
    train_template.set_mode('train')
    return train_template.encode(sample)


def _vllm_forward_kwargs(model_id, template, sample):
    vllm_template = copy.deepcopy(template)
    vllm_template.set_mode('vllm')
    encoded = vllm_template.encode(sample)
    mm_data = _build_mm_data(encoded)
    if not mm_data:
        return {'input_ids': encoded['input_ids'], 'mm_tensors': {}}

    model_config = ModelConfig(model_id, trust_remote_code=True, dtype='auto', seed=0)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    mm_items = processor.info.parse_mm_data(mm_data)
    result = processor(
        encoded['input_ids'],
        mm_items=mm_items,
        hf_processor_mm_kwargs=encoded.get('mm_processor_kwargs') or {},
    )
    return {
        'input_ids': result['prompt_token_ids'],
        'mm_tensors': result['mm_kwargs'].get_data(),
    }


def _tensors_aligned(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.shape != b.shape:
            return False
        if a.dtype.is_floating_point or b.dtype.is_floating_point:
            a = a.to(torch.bfloat16).float()
            b = b.to(torch.bfloat16).float()
            return torch.allclose(a, b, rtol=0, atol=0)
        return torch.equal(a, b)
    return nested_tensors_equal(a, b)


def _assert_mm_align(model_id, sample, *, tensor_key_aliases=None):
    tensor_key_aliases = tensor_key_aliases or {}
    processor = get_processor(model_id)
    template = get_template(processor)
    train = _swift_train_kwargs(template, sample)
    vllm = _vllm_forward_kwargs(model_id, template, sample)

    train_ids = _as_list_ids(train['input_ids'])
    vllm_ids = _as_list_ids(vllm['input_ids'])
    assert train_ids == vllm_ids, (
        f'[{model_id}] input_ids mismatch: train_len={len(train_ids)}, vllm_len={len(vllm_ids)}')

    vllm_tensors = dict(vllm['mm_tensors'])
    train_keys = sorted(k for k, v in train.items() if v is not None and k not in _SKIP_TRAIN_KEYS)
    vllm_keys = sorted(vllm_tensors.keys())
    compared = sorted(
        (train_key, tensor_key_aliases.get(train_key, train_key))
        for train_key in train_keys
        if tensor_key_aliases.get(train_key, train_key) in vllm_tensors)
    print(f'[{model_id}] train keys: {train_keys}')
    print(f'[{model_id}] vllm keys: {vllm_keys}')
    print(f'[{model_id}] compared keys: {[f"{tk}->{vk}" for tk, vk in compared]}')

    for train_key, vllm_key in compared:
        assert _tensors_aligned(train[train_key], vllm_tensors[vllm_key]), (
            f'[{model_id}] tensor mismatch: {train_key}!={vllm_key}')


def test_qwen3_5_image():
    _assert_mm_align('Qwen/Qwen3.5-0.8B', _image_sample())


def test_qwen3_5_video():
    _assert_mm_align('Qwen/Qwen3.5-0.8B', _video_sample())


def test_qwen3_vl_image():
    _assert_mm_align('Qwen/Qwen3-VL-2B-Instruct', _image_sample())


def test_qwen3_vl_video():
    _assert_mm_align('Qwen/Qwen3-VL-2B-Instruct', _video_sample())


def test_qwen3_omni_image():
    _assert_mm_align('Qwen/Qwen3-Omni-30B-A3B-Instruct', _image_sample())


def test_qwen3_omni_video():
    _assert_mm_align('Qwen/Qwen3-Omni-30B-A3B-Instruct', _video_sample())


def test_qwen3_omni_audio():
    _assert_mm_align(
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        _audio_sample(),
        tensor_key_aliases={
            'input_features': 'input_audio_features',
            'feature_attention_mask': 'feature_attention_mask',
        },
    )


def test_gemma4_image():
    _assert_mm_align(
        'google/gemma-4-E2B-it',
        _image_sample(),
        tensor_key_aliases={'image_position_ids': 'pixel_position_ids'},
    )


def test_gemma4_video():
    _assert_mm_align(
        'google/gemma-4-E2B-it',
        _video_sample(),
        tensor_key_aliases={
            'image_position_ids': 'pixel_position_ids',
            'video_position_ids': 'video_position_ids',
        },
    )


def test_gemma4_audio():
    _assert_mm_align(
        'google/gemma-4-E2B-it',
        _audio_sample(),
        tensor_key_aliases={
            'image_position_ids': 'pixel_position_ids',
            'video_position_ids': 'video_position_ids',
            'input_features': 'input_features_padded',
            'input_features_mask': 'input_features_mask',
        },
    )


if __name__ == '__main__':
    # test_qwen3_5_image()
    # test_qwen3_5_video()
    # test_qwen3_vl_image()
    test_qwen3_vl_video()
    # test_qwen3_omni_image()
    # test_qwen3_omni_video()
    # test_qwen3_omni_audio()
    # test_gemma4_image()
    # test_gemma4_video()
    # test_gemma4_audio()
