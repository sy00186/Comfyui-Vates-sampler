# ComfyUI-Vates-Sampler

**V-Sampler** (`VatesAdvancedSampler`) — Vates 品牌专属高性能采样链路。

Rust 组件位于本包 `dct-core/`，导出 PyO3 模块 **`vates_sampler_core`**（与 `ComfyUI-Vates-BatchLoader` 的 `vates_core` **互不冲突**，可共存）。

## 能力摘要

| 层级 | 行为 |
|------|------|
| **σ 调度** | Rust 高精度实现 **Karras / Exponential / Polyexponential**，其余调度自动降级至 `comfy.samplers.calculate_sigmas` |
| **显存真空 (Memory Vacuum)** | `memory_monitor_tick`：可选编译 **NVML**（`VATES_SAMPLER_BUILD_NVML=1 python install.py`）读取 GPU index0 占用比例；与高周期 `cuda.empty_cache` 组合，在多 batch（如 9 槽 Batch Loader）下平滑碎片峰值 |
| **噪声** | **PCG** 对齐 **B×C×H×W** float32 NumPy——`torch.from_numpy` **零拷贝**映射 |
| **循环** | 使用 `comfy.sample.sample_custom` + 回调注入（与内置 `SamplerCustom` 拓扑一致）；**不显式导入**整块 KSampler |

## 安装

1. 安装 [Rust toolchain](https://rustup.rs/)。
2. 在 **`ComfyUI-Vates-Sampler/dct-core/`** 目录执行：**`python install.py`**
   - （可选启用 NVML 监控：`set VATES_SAMPLER_BUILD_NVML=1` Windows / `export VATES_SAMPLER_BUILD_NVML=1` POSIX 后执行）。
3. 重启 ComfyUI。

## 输入

对标经典 KSampler Advanced：`model` / `seed` / `steps` / `cfg` / `sampler_name` / `scheduler` / `positive` / `negative` / `latent_image` / `denoise`。

## 输出

`LATENT`（拓扑与原生采样链路一致）。

## 兼容性

需要一个**完整安装**的官方 ComfyUI 树（内置 `comfy.sample`、`latent_preview` 与 `sampler_object`）。
