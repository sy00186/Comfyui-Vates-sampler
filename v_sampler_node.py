"""
V-Sampler：Rust (`vates_sampler_core`) 与 Comfy `sample_custom` 组合出的手写采样入口。

- σ：Rust 纯实现 **karras / exponential / polyexponential**（与 Comfy `k_diffusion/sampling.py` + `BasicScheduler` 截取一致）；其它调度降级为 `comfy.samplers.calculate_sigmas`。
- **噪声**：Rust **PCG** 生成的 **对齐 B×C×H×W f32**，经 NumPy→`torch.from_numpy` 映射至设备；
- **显存哨兵**：每步预览回调链路中耦合 `memory_monitor_tick`（可选 NVML），并配合 `cuda.empty_cache` 做 batch 碎片化平滑。
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
from pathlib import Path

import latent_preview

import comfy.model_management as model_management

import comfy.sample as comfy_sample
import comfy.samplers as comfy_samplers
import comfy.utils as comfy_utils
import numpy as np
import torch

logger = logging.getLogger(__name__)

_SAMPLER_ROOT = Path(__file__).resolve().parent
_DCT_CORE = _SAMPLER_ROOT / "dct-core"
if _DCT_CORE.is_dir() and str(_DCT_CORE) not in sys.path:
    sys.path.insert(0, str(_DCT_CORE))

_SCHED_RUST = frozenset({"karras", "exponential", "polyexponential"})
_MAX_BATCH_HINT = 9


def _pick_sampler_core():  # noqa: ANN401
    """优先加载 `dct-core` 对齐的 ``vates_sampler_core`` 扩展模块。"""
    for cand in ("vates_sampler_core.pyd", "vates_sampler_core.so"):
        p = _DCT_CORE / cand
        if not p.is_file():
            continue
        spec = importlib.util.spec_from_file_location("vates_sampler_core_ext_shim", p)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    try:
        import vates_sampler_core as vsc  # type: ignore
    except ImportError:
        raise ImportError(
            "未找到 `vates_sampler_core`。请在 `ComfyUI-Vates-Sampler/dct-core/` 目录执行：`python install.py`。"
        ) from None
    return vsc


def _sampler_core_or_none():  # noqa: ANN401
    try:
        return _pick_sampler_core()
    except ImportError as e:
        logger.warning("%s", e)
        return None


def _batch_hint(b: int) -> None:
    if b > _MAX_BATCH_HINT:
        logger.warning(
            "V-Sampler：batch=%s 超过与批量 Loader（9 槽）软性对齐的上限提示；仍可运行，OOM 时请降 batch。",
            b,
        )


def _sigma_tensor_from_python(model, scheduler: str, steps: int, denoise: float) -> torch.Tensor:
    """与 `BasicScheduler.execute`：`total_steps = int(steps/denoise)`、`sigmas[-(steps+1):]` 对齐。"""
    if denoise < 1.0:
        if denoise <= 0.0:
            raise ValueError("denoise 须为正；Comfy BasicScheduler 在 denoise≤0 时返回空 sigma，本节点同样拒绝该配置。")
        total_steps = max(1, int(steps / denoise))
    else:
        total_steps = steps

    ms = model.get_model_object("model_sampling")
    sigmas = comfy_samplers.calculate_sigmas(ms, scheduler, total_steps).cpu().float()
    if sigmas.shape[-1] > steps + 1:
        sigmas = sigmas[-(steps + 1) :]
    return sigmas


def _sigma_tensor_pref_rust(core, model, scheduler: str, steps: int, denoise: float) -> torch.Tensor:
    ls = scheduler.lower().strip()
    if ls not in _SCHED_RUST:
        return _sigma_tensor_from_python(model, scheduler, steps, denoise)
    ms = model.get_model_object("model_sampling")
    arr = core.build_sigma_schedule(
        ls,
        float(ms.sigma_min),
        float(ms.sigma_max),
        int(steps),
        float(denoise),
        7.0,
        1.0,
    )
    flat = np.ascontiguousarray(np.asarray(arr), dtype=np.float32)
    return torch.from_numpy(flat)


def _rust_latent_noise(
    core,
    shape_bchw: tuple[int, int, int, int],
    seed: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    b, c, h, w = shape_bchw
    arr = core.latent_gaussian_noise_f32(b, c, h, w, int(seed) & 0xFFFFFFFFFFFFFFFF, 0)
    t = torch.from_numpy(np.asarray(arr))
    blk = getattr(model_management, "device_supports_non_blocking", None)
    nb = blk(device) if callable(blk) else False
    return t.to(device=device, dtype=dtype, non_blocking=nb)


class VatesAdvancedSampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "model": ("MODEL",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (comfy_samplers.SAMPLER_NAMES, {}),
                "scheduler": (comfy_samplers.SCHEDULER_NAMES, {}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "Vates/Sampling"

    def sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image: dict,
        denoise: float,
    ) -> tuple[dict]:
        core = _sampler_core_or_none()

        latent = latent_image.copy()
        latent_image_ts = latent["samples"]
        if getattr(latent_image_ts, "is_nested", False):
            raise RuntimeError("V-Sampler：暂不支持 NestedTensor latent — 请使用内置采样节点。")

        latent_image_ts = comfy_sample.fix_empty_latent_channels(
            model,
            latent_image_ts,
            latent.get("downscale_ratio_spacial"),
        )
        latent["samples"] = latent_image_ts

        _batch_hint(latent_image_ts.shape[0])

        try:
            sampler = comfy_samplers.sampler_object(sampler_name)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"无效的 sampler_name={sampler_name!r}") from e

        if core is not None:
            sigmas = _sigma_tensor_pref_rust(core, model, scheduler, steps, denoise)
        else:
            sigmas = _sigma_tensor_from_python(model, scheduler, steps, denoise)

        if sigmas.numel() < 2:
            raise RuntimeError("Sigma 序列过短；请增大 denoise/steps。")

        sigmas_device = sigmas.to(model.load_device)

        dtype = latent_image_ts.dtype
        device = latent_image_ts.device
        if core is None:
            noise_t = comfy_sample.prepare_noise(latent_image_ts, int(seed))
        else:
            b, c, h, w = (
                latent_image_ts.shape[0],
                latent_image_ts.shape[1],
                latent_image_ts.shape[2],
                latent_image_ts.shape[3],
            )
            noise_t = _rust_latent_noise(core, (b, c, h, w), int(seed), dtype=dtype, device=device)

        noise_mask = latent.get("noise_mask")

        x0_payload: dict = {}
        preview_cb = latent_preview.prepare_callback(
            model,
            sigmas_device.shape[-1] - 1,
            x0_payload,
        )

        def vacuum_callback(step_idx: int, latent_denoised, x_lat, total_steps):
            if core is not None:
                try:
                    s_np = np.ascontiguousarray(sigmas.detach().cpu().numpy(), dtype=np.float64)
                    if 0 <= int(step_idx) < s_np.size - 1:
                        _ = core.sigma_pair_instruction(s_np, int(step_idx))
                except Exception:  # noqa: BLE001
                    pass

                try:
                    st = core.memory_monitor_tick(0.90)
                    if isinstance(st, dict) and st.get("vacuum_hot"):
                        gc.collect()
                except Exception:  # noqa: BLE001
                    gc.collect()
            else:
                gc.collect()

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

            if preview_cb is not None:
                preview_cb(step_idx, latent_denoised, x_lat, total_steps)

        disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED
        sampled = comfy_sample.sample_custom(
            model,
            noise_t,
            float(cfg),
            sampler,
            sigmas_device,
            positive,
            negative,
            latent_image_ts,
            noise_mask=noise_mask,
            callback=vacuum_callback,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )

        latent_out = latent.copy()
        latent_out.pop("downscale_ratio_spacial", None)
        latent_out["samples"] = sampled.to(
            dtype=model_management.intermediate_dtype(),
            device=model_management.intermediate_device(),
        )
        logger.debug("V-Sampler 完成 sampler=%s scheduler=%s steps=%s", sampler_name, scheduler, steps)
        return (latent_out,)
