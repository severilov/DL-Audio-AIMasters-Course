from dataclasses import dataclass, field
import hashlib
import os
import random
import time
from urllib import request

import torch
import torch.nn.functional as F
try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None

from tortoise.models.arch_util import TorchMelSpectrogram
from tortoise.models.autoregressive import UnifiedVoice
from tortoise.utils.audio import denormalize_tacotron_mel
from tortoise.utils.diffusion import SpacedDiffusion, get_named_beta_schedule, space_timesteps


pbar = None

ARTIFACT_SCHEMA_VERSION = 1
AUTOREGRESSIVE_TEXT_LOSS_WEIGHT = 0.01
AUTOREGRESSIVE_MEL_LOSS_WEIGHT = 1.0
DIFFUSION_CODE_PREDICTION_LOSS_ENABLED_BY_DEFAULT = False
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'tortoise', 'models')
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', DEFAULT_MODELS_DIR)
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
STAGE_FILENAMES = {
    'autoregressive': 'stage1_autoregressive.pt',
    'clvp': 'stage2_clvp.pt',
    'diffusion_vocoder': 'stage3_diffusion_vocoder.pt',
}
MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
}
VQ_VAE_OBJECTIVE_SPEC = {
    'reconstruction': 'mse',
    'commitment_loss_present': True,
    'commitment_weight': 'unspecified_in_public_repo_or_paper',
    'codebook_loss': 'unspecified_in_public_repo_or_paper',
    'executable_integration_supported': False,
}


@dataclass
class Config:
    models_dir: str = MODELS_DIR
    enable_redaction: bool = True
    use_deepspeed: bool = False
    device: str | None = None
    autoregressive_batch_size: int | None = None
    jit: bool = False
    text: str = (
        "Joining two modalities results in a surprising increase in generalization! "
        "What would happen if we combined them all?"
    )
    voice: str = "tom"
    preset: str | None = "ultra_fast"
    k: int = 1
    verbose: bool = True
    use_deterministic_seed: int | None = None
    return_deterministic_state: bool = False
    num_autoregressive_samples: int | None = None
    temperature: float | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    max_mel_tokens: int | None = None
    cvvp_amount: float | None = None
    diffusion_iterations: int | None = None
    cond_free: bool | None = None
    cond_free_k: float | None = None
    diffusion_temperature: float | None = None
    trained_diffusion_steps: int = 4000
    hf_generate_kwargs: dict = field(default_factory=dict)

    


def load_autoregressive_model(models_dir, use_deepspeed=False):
    autoregressive = UnifiedVoice(
        max_mel_tokens=604,
        max_text_tokens=402,
        max_conditioning_inputs=2,
        layers=30,
        model_dim=1024,
        heads=16,
        number_text_tokens=255,
        start_text_token=255,
        checkpointing=False,
        train_solo_embeddings=False,
    ).cpu().eval()
    autoregressive_state = torch.load(
        get_model_path('autoregressive.pth', models_dir),
        map_location=torch.device('cpu'),
    )
    for key in list(autoregressive_state.keys()):
        key_parts = key.split('.')
        if key.startswith('gpt.h.') and key_parts[-2:] in (['attn', 'bias'], ['attn', 'masked_bias']):
            autoregressive_state.pop(key)
    load_result = autoregressive.load_state_dict(autoregressive_state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            'Unexpected autoregressive checkpoint mismatch after compatibility cleanup: '
            f'missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}'
        )
    autoregressive.post_init_gpt2_config(use_deepspeed=use_deepspeed)
    return autoregressive


def get_autoregressive_conditioning_latents(voice_samples, autoregressive, device):
    with torch.no_grad():
        if isinstance(voice_samples, tuple):
            voice_samples = list(voice_samples)
        elif not isinstance(voice_samples, list):
            voice_samples = [voice_samples]
        voice_samples = [sample.to(device) for sample in voice_samples]

        auto_conds = []
        for sample in voice_samples:
            auto_conds.append(format_conditioning(sample, device=device))
        auto_conds = torch.stack(auto_conds, dim=1)

        autoregressive = autoregressive.to(device)
        auto_latent = autoregressive.get_conditioning(auto_conds)
        autoregressive = autoregressive.cpu()

    return auto_latent, auto_conds


def download_models(specific_models=None, models_dir=MODELS_DIR):
    """
    Call to download all the models that Tortoise uses.
    """
    os.makedirs(models_dir, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        global pbar
        if progressbar is None:
            return
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None

    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, model_path, show_progress)
        print('Done.')


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f'Model {model_name} not found in available models.')
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        download_models([model_name], models_dir=models_dir)
    return model_path


def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    if t.shape[-1] < length:
        return F.pad(t, (0, length - t.shape[-1]))
    return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True, cond_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(
        use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
        model_mean_type='epsilon',
        model_var_type='learned_range',
        loss_type='mse',
        betas=get_named_beta_schedule('linear', trained_diffusion_steps),
        conditioning_free=cond_free,
        conditioning_free_k=cond_free_k,
    )


def format_conditioning(clip, cond_length=132300, device='cuda'):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)


def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    """
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        if complain:
            print(
                "No stop tokens found in one of the generated voice clips. This typically means the spoken audio is "
                "too long. In some cases, the output will still be good, though. Listen to it and if it is missing "
                "words, try breaking up your input text."
            )
        return codes
    codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248
    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[1] * 4 * 24000 // 22050
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(
            latents, conditioning_latents, output_seq_len, False
        )

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(
            diffusion_model,
            output_shape,
            noise=noise,
            model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
            progress=verbose,
        )
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


def pick_best_batch_size_for_gpu():
    """
    Tries to pick a batch size that will fit in your GPU. These sizes aren't guaranteed to work, but they should give
    you a good shot.
    """
    if torch.cuda.is_available():
        _, available = torch.cuda.mem_get_info()
        available_gb = available / (1024 ** 3)
        if available_gb > 14:
            return 16
        if available_gb > 10:
            return 8
        if available_gb > 7:
            return 4
    return 1


def set_deterministic_seed(seed=None):
    seed = int(time.time()) if seed is None else seed
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


def resolve_generation_settings(config):
    settings = {
        'num_autoregressive_samples': 512,
        'temperature': 0.8,
        'length_penalty': 1.0,
        'repetition_penalty': 2.0,
        'top_p': 0.8,
        'max_mel_tokens': 500,
        'cvvp_amount': 0.0,
        'diffusion_iterations': 100,
        'cond_free': True,
        'typical_mass': 0.9,
        'typical_sampling': False,
        'cond_free_k': 2.0,
        'k': 1,
        'input_tokens': None,
        'diffusion_temperature': 1.0,
        'trained_diffusion_steps': config.trained_diffusion_steps,
        'hf_generate_kwargs': config.hf_generate_kwargs,
    }
    presets = {
        'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
        'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
        'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
        'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
    }
    if config.preset is not None:
        settings.update(presets[config.preset])

    overrides = {
        'num_autoregressive_samples': config.num_autoregressive_samples,
        'temperature': config.temperature,
        'length_penalty': config.length_penalty,
        'repetition_penalty': config.repetition_penalty,
        'top_p': config.top_p,
        'max_mel_tokens': config.max_mel_tokens,
        'cvvp_amount': config.cvvp_amount,
        'diffusion_iterations': config.diffusion_iterations,
        'cond_free': config.cond_free,
        'cond_free_k': config.cond_free_k,
        'diffusion_temperature': config.diffusion_temperature,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def resolve_conditioning_source(voice_samples, conditioning_latents):
    has_voice_samples = voice_samples is not None
    has_conditioning_latents = conditioning_latents is not None
    if has_voice_samples and has_conditioning_latents:
        raise ValueError("Provide either voice_samples or conditioning_latents, not both.")
    if has_voice_samples:
        return 'voice_samples'
    if has_conditioning_latents:
        return 'conditioning_latents'
    return 'random'


def require_conditioning_latents(conditioning_latents, *, require_auto=False, require_diffusion=False):
    if conditioning_latents is None:
        raise ValueError("conditioning_latents must be provided for latent-based conditioning.")
    if not isinstance(conditioning_latents, (tuple, list)) or len(conditioning_latents) != 2:
        raise ValueError("conditioning_latents must be a 2-item tuple/list of (auto_latent, diffusion_latent).")

    auto_conditioning, diffusion_conditioning = conditioning_latents
    if require_auto and auto_conditioning is None:
        raise ValueError(
            "conditioning_latents[0] is None, so the autoregressive path has no speaker/style conditioning latent."
        )
    if require_diffusion and diffusion_conditioning is None:
        raise ValueError(
            "conditioning_latents[1] is None, so the diffusion path has no conditioning latent."
        )
    return auto_conditioning, diffusion_conditioning


def combine_autoregressive_losses(
    loss_text_raw,
    loss_mel_raw,
    text_weight=AUTOREGRESSIVE_TEXT_LOSS_WEIGHT,
    mel_weight=AUTOREGRESSIVE_MEL_LOSS_WEIGHT,
):
    return loss_text_raw * text_weight + loss_mel_raw * mel_weight


def _require_analysis_keys(batch, required_keys, batch_name):
    if not isinstance(batch, dict):
        raise TypeError(f"{batch_name} must be a dict, got {type(batch).__name__}")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise ValueError(f"{batch_name} is missing required keys: {missing}")


def _require_tensor(value, name, dims=None, dtype=None):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if dims is not None and value.ndim != dims:
        raise ValueError(f"{name} must have rank {dims}, got {value.ndim}")
    if dtype == 'floating' and not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if dtype == 'integral' and value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must be an integer tensor, got {value.dtype}")
    return value


def _require_batch_vector(lengths, name, batch_size, allow_zero=False):
    _require_tensor(lengths, name, dims=1, dtype='integral')
    if lengths.shape[0] != batch_size:
        raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {lengths.shape[0]}")
    min_allowed = 0 if allow_zero else 1
    if torch.any(lengths < min_allowed):
        comparator = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must contain {comparator} values")


def tensor_metadata(tensor):
    _require_tensor(tensor, 'tensor')
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
    }


def summarize_tensor_mapping(mapping):
    return {
        key: tensor_metadata(value)
        for key, value in mapping.items()
        if torch.is_tensor(value)
    }


def _validate_autoregressive_analysis_batch(model, batch):
    _require_analysis_keys(
        batch,
        ['speech_conditioning_latent', 'text_inputs', 'text_lengths', 'mel_codes', 'wav_lengths'],
        'analysis_batch',
    )
    if batch.get('raw_mels') is not None:
        raise ValueError("analysis_batch['raw_mels'] is not supported by this checkpoint configuration")
    if batch.get('types') not in (None, 0):
        raise ValueError("analysis_batch['types'] is not supported by this checkpoint configuration")

    speech_conditioning_latent = _require_tensor(batch['speech_conditioning_latent'], 'analysis_batch.speech_conditioning_latent')
    text_inputs = _require_tensor(batch['text_inputs'], 'analysis_batch.text_inputs', dims=2, dtype='integral')
    mel_codes = _require_tensor(batch['mel_codes'], 'analysis_batch.mel_codes', dims=2, dtype='integral')
    batch_size = speech_conditioning_latent.shape[0]
    if text_inputs.shape[0] != batch_size or mel_codes.shape[0] != batch_size:
        raise ValueError("analysis_batch tensors must share the same batch dimension")

    text_lengths = batch['text_lengths']
    wav_lengths = batch['wav_lengths']
    _require_batch_vector(text_lengths, 'analysis_batch.text_lengths', batch_size)
    _require_batch_vector(wav_lengths, 'analysis_batch.wav_lengths', batch_size)

    if text_lengths.max().item() > text_inputs.shape[1]:
        raise ValueError("analysis_batch.text_lengths cannot exceed text_inputs width")
    if torch.any(text_inputs < 0) or torch.any(text_inputs >= model.text_embedding.num_embeddings):
        raise ValueError("analysis_batch.text_inputs contains token ids outside the text vocabulary")

    for batch_index in range(batch_size):
        text_length = int(text_lengths[batch_index].item())
        last_token = text_inputs[batch_index, text_length - 1].item()
        if last_token == model.stop_text_token:
            raise ValueError(
                "analysis_batch.text_inputs must be raw BPE ids without the stage-1 stop token appended"
            )

    if torch.any(mel_codes < 0) or torch.any(mel_codes >= model.start_mel_token):
        raise ValueError(
            "analysis_batch.mel_codes must contain only discrete mel ids and must not include reserved start/stop ids"
        )
    max_required_mel_len = torch.div(wav_lengths.max(), model.mel_length_compression, rounding_mode='trunc').item()
    if mel_codes.shape[1] < max_required_mel_len:
        raise ValueError(
            "analysis_batch.mel_codes is too short for the supplied wav_lengths and mel_length_compression"
        )

    return {
        'speech_conditioning_latent': speech_conditioning_latent,
        'text_inputs': text_inputs,
        'text_lengths': text_lengths,
        'mel_codes': mel_codes,
        'wav_lengths': wav_lengths,
        'text_first': batch.get('text_first', True),
        'clip_inputs': batch.get('clip_inputs', True),
    }


def run_autoregressive_loss_analysis(model, batch):
    validated = _validate_autoregressive_analysis_batch(model, batch)
    loss_text_raw, loss_mel_raw, mel_logits = model(
        validated['speech_conditioning_latent'],
        validated['text_inputs'],
        validated['text_lengths'],
        validated['mel_codes'],
        validated['wav_lengths'],
        text_first=validated['text_first'],
        clip_inputs=validated['clip_inputs'],
    )
    return {
        'loss_text_raw': loss_text_raw.detach(),
        'loss_mel_raw': loss_mel_raw.detach(),
        'loss_total_weighted': combine_autoregressive_losses(loss_text_raw, loss_mel_raw).detach(),
        'mel_logits': mel_logits.detach(),
        'input_summary': summarize_tensor_mapping(
            {
                'speech_conditioning_latent': validated['speech_conditioning_latent'],
                'text_inputs': validated['text_inputs'],
                'text_lengths': validated['text_lengths'],
                'mel_codes': validated['mel_codes'],
                'wav_lengths': validated['wav_lengths'],
            }
        ),
        'output_summary': summarize_tensor_mapping({'mel_logits': mel_logits}),
    }


def _validate_clvp_analysis_batch(model, batch):
    _require_analysis_keys(batch, ['text_tokens', 'speech_tokens'], 'analysis_batch')
    text_tokens = _require_tensor(batch['text_tokens'], 'analysis_batch.text_tokens', dims=2, dtype='integral')
    speech_tokens = _require_tensor(batch['speech_tokens'], 'analysis_batch.speech_tokens', dims=2, dtype='integral')
    if text_tokens.shape[0] != speech_tokens.shape[0]:
        raise ValueError("analysis_batch.text_tokens and analysis_batch.speech_tokens must share a batch dimension")
    if torch.any(text_tokens < 0) or torch.any(text_tokens >= model.text_emb.num_embeddings):
        raise ValueError("analysis_batch.text_tokens contains token ids outside the CLVP text vocabulary")
    if torch.any(speech_tokens < 0) or torch.any(speech_tokens >= model.speech_emb.num_embeddings):
        raise ValueError("analysis_batch.speech_tokens contains token ids outside the CLVP speech vocabulary")
    return text_tokens, speech_tokens


def run_clvp_loss_analysis(model, batch):
    text_tokens, speech_tokens = _validate_clvp_analysis_batch(model, batch)
    diagonal_similarity = model(text_tokens, speech_tokens, return_loss=False)
    loss = model(text_tokens, speech_tokens, return_loss=True)
    return {
        'loss': loss.detach(),
        'diagonal_similarity': diagonal_similarity.detach(),
        'temperature_exp': model.temperature.exp().detach(),
        'batch_size': text_tokens.shape[0],
        'batch_is_singleton': text_tokens.shape[0] == 1,
        'input_summary': summarize_tensor_mapping(
            {
                'text_tokens': text_tokens,
                'speech_tokens': speech_tokens,
            }
        ),
        'output_summary': summarize_tensor_mapping(
            {
                'diagonal_similarity': diagonal_similarity,
                'loss': loss,
            }
        ),
    }


def load_training_schedule_diffuser(trained_diffusion_steps=4000, cond_free=True, cond_free_k=1):
    return SpacedDiffusion(
        use_timesteps=list(range(trained_diffusion_steps)),
        model_mean_type='epsilon',
        model_var_type='learned_range',
        loss_type='mse',
        betas=get_named_beta_schedule('linear', trained_diffusion_steps),
        conditioning_free=cond_free,
        conditioning_free_k=cond_free_k,
    )


def _validate_diffusion_analysis_batch(batch, diffuser):
    _require_analysis_keys(
        batch,
        ['x_start', 'aligned_conditioning', 'conditioning_latent'],
        'analysis_batch',
    )
    x_start = _require_tensor(batch['x_start'], 'analysis_batch.x_start', dims=3, dtype='floating')
    aligned_conditioning = _require_tensor(batch['aligned_conditioning'], 'analysis_batch.aligned_conditioning')
    conditioning_latent = _require_tensor(batch['conditioning_latent'], 'analysis_batch.conditioning_latent', dims=2, dtype='floating')
    batch_size = x_start.shape[0]
    if conditioning_latent.shape[0] != batch_size or aligned_conditioning.shape[0] != batch_size:
        raise ValueError("analysis_batch diffusion tensors must share a batch dimension")

    conditioning_free = batch.get('conditioning_free', False)
    return_code_pred = batch.get('return_code_pred', False)
    if conditioning_free and return_code_pred:
        raise ValueError(
            "analysis_batch cannot set conditioning_free=True together with return_code_pred=True for DiffusionTts"
        )

    if aligned_conditioning.dtype == torch.long:
        conditioning_mode = 'tokens'
    elif aligned_conditioning.dtype == torch.float32:
        conditioning_mode = 'latents'
    else:
        raise TypeError(
            "analysis_batch.aligned_conditioning must be torch.long tokens or torch.float32 latents"
        )

    timesteps = batch.get('timesteps')
    if timesteps is None:
        timesteps = torch.randint(0, diffuser.num_timesteps, (batch_size,), device=x_start.device)
    else:
        _require_batch_vector(timesteps, 'analysis_batch.timesteps', batch_size, allow_zero=True)

    noise = batch.get('noise')
    if noise is not None:
        _require_tensor(noise, 'analysis_batch.noise', dims=x_start.ndim, dtype='floating')
        if noise.shape != x_start.shape:
            raise ValueError("analysis_batch.noise must match x_start shape")

    return {
        'x_start': x_start,
        'aligned_conditioning': aligned_conditioning,
        'conditioning_latent': conditioning_latent,
        'timesteps': timesteps,
        'noise': noise,
        'conditioning_free': conditioning_free,
        'return_code_pred': return_code_pred,
        'conditioning_mode': conditioning_mode,
    }


def run_diffusion_loss_analysis(model, diffuser, batch):
    validated = _validate_diffusion_analysis_batch(batch, diffuser)
    terms = diffuser.training_losses(
        model,
        validated['x_start'],
        validated['timesteps'],
        model_kwargs={
            'aligned_conditioning': validated['aligned_conditioning'],
            'conditioning_latent': validated['conditioning_latent'],
            'conditioning_free': validated['conditioning_free'],
            'return_code_pred': validated['return_code_pred'],
        },
        noise=validated['noise'],
    )
    analysis = {
        'loss': terms['loss'].detach(),
        'mse': terms['mse'].detach(),
        'x_start_predicted': terms['x_start_predicted'].detach(),
        'timesteps': validated['timesteps'].detach(),
        'conditioning_free': validated['conditioning_free'],
        'conditioning_mode': validated['conditioning_mode'],
        'input_summary': summarize_tensor_mapping(
            {
                'x_start': validated['x_start'],
                'aligned_conditioning': validated['aligned_conditioning'],
                'conditioning_latent': validated['conditioning_latent'],
                'timesteps': validated['timesteps'],
            }
        ),
        'output_summary': summarize_tensor_mapping(
            {
                'loss': terms['loss'],
                'mse': terms['mse'],
                'x_start_predicted': terms['x_start_predicted'],
            }
        ),
    }
    if 'vb' in terms:
        analysis['vb'] = terms['vb'].detach()
        analysis['output_summary']['vb'] = tensor_metadata(terms['vb'])
    extra_outputs = terms.get('extra_outputs')
    if extra_outputs:
        analysis['mel_pred'] = extra_outputs[0].detach()
        analysis['output_summary']['mel_pred'] = tensor_metadata(extra_outputs[0])
    return analysis


def default_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_run_dir(artifacts_dir=ARTIFACTS_DIR, run_id=None):
    artifacts_dir = os.path.abspath(artifacts_dir)
    if run_id is None:
        os.makedirs(artifacts_dir, exist_ok=True)
        return None, artifacts_dir
    run_dir = os.path.join(artifacts_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def resolve_run_dir(artifacts_dir=ARTIFACTS_DIR, run_id=None, run_dir=None):
    if run_dir is not None:
        resolved_run_dir = os.path.abspath(run_dir)
        resolved_run_id = os.path.basename(os.path.normpath(resolved_run_dir))
    else:
        artifacts_dir = os.path.abspath(artifacts_dir)
        if run_id is None:
            resolved_run_id = None
            resolved_run_dir = artifacts_dir
        else:
            resolved_run_id = run_id
            resolved_run_dir = os.path.join(artifacts_dir, run_id)
    if not os.path.isdir(resolved_run_dir):
        raise FileNotFoundError(f"Run directory not found: {resolved_run_dir}")
    return resolved_run_id, resolved_run_dir


def stage_artifact_path(run_dir, stage_name):
    if stage_name not in STAGE_FILENAMES:
        raise ValueError(f"Unknown stage name: {stage_name}")
    return os.path.join(run_dir, STAGE_FILENAMES[stage_name])


def capture_rng_state():
    return {
        'python_random_state': random.getstate(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state):
    if rng_state is None:
        return
    random.setstate(rng_state['python_random_state'])
    torch.set_rng_state(rng_state['torch_rng_state'])
    if rng_state.get('cuda_rng_state_all') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['cuda_rng_state_all'])


def move_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    return value


def summarize_value(value):
    if torch.is_tensor(value):
        return {
            'kind': 'tensor',
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if isinstance(value, list):
        if value and all(torch.is_tensor(item) for item in value):
            return {
                'kind': 'list[tensor]',
                'length': len(value),
                'items': [summarize_value(item) for item in value],
            }
        return {
            'kind': 'list',
            'length': len(value),
            'items': [summarize_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            'kind': 'tuple',
            'length': len(value),
            'items': [summarize_value(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            'kind': 'dict',
            'items': {key: summarize_value(item) for key, item in value.items()},
        }
    return {
        'kind': type(value).__name__,
    }


def summarize_payload(payload):
    return {key: summarize_value(value) for key, value in payload.items()}


def validate_value_summary(value, summary, path):
    kind = summary['kind']
    if kind == 'tensor':
        if not torch.is_tensor(value):
            raise ValueError(f"{path} is not a tensor")
        expected_shape = summary['shape']
        expected_dtype = summary['dtype']
        if list(value.shape) != expected_shape:
            raise ValueError(f"{path} shape mismatch: expected {expected_shape}, got {list(value.shape)}")
        if str(value.dtype) != expected_dtype:
            raise ValueError(f"{path} dtype mismatch: expected {expected_dtype}, got {value.dtype}")
        return
    if kind == 'list[tensor]':
        if not isinstance(value, list):
            raise ValueError(f"{path} is not a list")
        if len(value) != summary['length']:
            raise ValueError(f"{path} length mismatch: expected {summary['length']}, got {len(value)}")
        for index, item in enumerate(value):
            validate_value_summary(item, summary['items'][index], f"{path}[{index}]")
        return
    if kind == 'list':
        if not isinstance(value, list):
            raise ValueError(f"{path} is not a list")
        if len(value) != summary['length']:
            raise ValueError(f"{path} length mismatch: expected {summary['length']}, got {len(value)}")
        for index, item in enumerate(value):
            validate_value_summary(item, summary['items'][index], f"{path}[{index}]")
        return
    if kind == 'tuple':
        if not isinstance(value, tuple):
            raise ValueError(f"{path} is not a tuple")
        if len(value) != summary['length']:
            raise ValueError(f"{path} length mismatch: expected {summary['length']}, got {len(value)}")
        for index, item in enumerate(value):
            validate_value_summary(item, summary['items'][index], f"{path}[{index}]")
        return
    if kind == 'dict':
        if not isinstance(value, dict):
            raise ValueError(f"{path} is not a dict")
        expected_keys = set(summary['items'].keys())
        actual_keys = set(value.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"{path} keys mismatch: expected {sorted(expected_keys)}, got {sorted(actual_keys)}")
        for key, item_summary in summary['items'].items():
            validate_value_summary(value[key], item_summary, f"{path}.{key}")


def text_hash(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def save_stage_artifact(
    run_dir,
    stage_name,
    payload,
    config_snapshot,
    generation_settings_snapshot,
    conditioning_source,
    text,
    upstream_artifact_path=None,
):
    payload = move_to_cpu(payload)
    manifest = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'run_id': os.path.basename(os.path.normpath(run_dir)),
        'stage_name': stage_name,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'conditioning_source': conditioning_source,
        'text_hash': text_hash(text),
        'config_snapshot': config_snapshot,
        'generation_settings_snapshot': generation_settings_snapshot,
        'upstream_artifact_path': upstream_artifact_path,
        'required_keys': sorted(payload.keys()),
        'payload_summary': summarize_payload(payload),
    }
    path = stage_artifact_path(run_dir, stage_name)
    torch.save({'manifest': manifest, 'payload': payload}, path)
    return path


def load_stage_artifact(path, expected_stage_name=None, required_keys=None):
    artifact = torch.load(path, map_location=torch.device('cpu'))
    if not isinstance(artifact, dict) or 'manifest' not in artifact or 'payload' not in artifact:
        raise ValueError(f"Invalid artifact file: {path}")
    manifest = artifact['manifest']
    payload = artifact['payload']
    if manifest.get('schema_version') != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema mismatch for {path}: expected {ARTIFACT_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    if expected_stage_name is not None and manifest.get('stage_name') != expected_stage_name:
        raise ValueError(
            f"Artifact stage mismatch for {path}: expected {expected_stage_name}, got {manifest.get('stage_name')}"
        )
    if required_keys is not None:
        missing = [key for key in required_keys if key not in payload]
        if missing:
            raise ValueError(f"Artifact payload missing keys in {path}: {missing}")
    for key, summary in manifest['payload_summary'].items():
        validate_value_summary(payload[key], summary, key)
    return artifact


def list_saved_runs(artifacts_dir=ARTIFACTS_DIR):
    if not os.path.isdir(artifacts_dir):
        return []
    runs = []
    root_stage_manifests = {}
    for stage_name in STAGE_FILENAMES:
        path = stage_artifact_path(artifacts_dir, stage_name)
        if os.path.exists(path):
            try:
                root_stage_manifests[stage_name] = torch.load(path, map_location=torch.device('cpu'))['manifest']
            except Exception:
                root_stage_manifests[stage_name] = None
    if root_stage_manifests:
        runs.append({'run_id': None, 'run_dir': artifacts_dir, 'stages': root_stage_manifests})
    for run_id in sorted(os.listdir(artifacts_dir)):
        run_dir = os.path.join(artifacts_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        stage_manifests = {}
        for stage_name in STAGE_FILENAMES:
            path = stage_artifact_path(run_dir, stage_name)
            if os.path.exists(path):
                try:
                    stage_manifests[stage_name] = torch.load(path, map_location=torch.device('cpu'))['manifest']
                except Exception:
                    stage_manifests[stage_name] = None
        runs.append({'run_id': run_id, 'run_dir': run_dir, 'stages': stage_manifests})
    return runs
