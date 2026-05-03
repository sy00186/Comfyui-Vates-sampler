//! V-Sampler Rust 核心：**σ 调度**（对齐 Comfy `k_diffusion/sampling.py`）、**PCG64 噪声**、**可选 NVML 刻度**。
//!
//! 与同机 `vates_core`（DCT）独立：导出 PyO3 模块 `vates_sampler_core`。

#[cfg(feature = "nvml")]
use once_cell::sync::Lazy;

/// `denoise ∈ (0,1)` 时类似 Comfy `BasicScheduler`：先离散整段调度再截取末尾 `steps + 1` 个 σ。
#[inline]
pub fn comfy_scheduler_total_steps(visible_steps: u32, denoise: f64) -> usize {
    if !(denoise > 0.0) || visible_steps < 1 {
        return visible_steps.max(1) as usize;
    }
    if denoise >= 1.0 - f64::EPSILON {
        return visible_steps as usize;
    }
    let t = (f64::from(visible_steps) / denoise).floor() as i64;
    t.max(1) as usize
}

fn append_zero(mut v: Vec<f64>) -> Vec<f64> {
    v.push(0.0);
    v
}

/// `get_sigmas_karras` + `append_zero`。
pub fn sigmas_karras(n: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    debug_assert!(n >= 1, "n must be >= 1");
    let rho = rho.max(1e-6);
    let min_inv_rho = sigma_min.powf(1.0 / rho);
    let max_inv_rho = sigma_max.powf(1.0 / rho);
    let mut ramp = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n <= 1 {
            0.0_f64
        } else {
            i as f64 / (n - 1) as f64
        };
        ramp.push((max_inv_rho + t * (min_inv_rho - max_inv_rho)).powf(rho));
    }
    append_zero(ramp)
}

/// `get_sigmas_exponential` + `append_zero`。
pub fn sigmas_exponential(n: usize, sigma_min: f64, sigma_max: f64) -> Vec<f64> {
    debug_assert!(n >= 1, "n must be >= 1");
    let log_min = sigma_min.ln();
    let log_max = sigma_max.ln();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n <= 1 {
            log_max
        } else {
            log_max + (log_min - log_max) * (i as f64 / (n - 1) as f64)
        };
        out.push(t.exp());
    }
    append_zero(out)
}

/// Comfy：`ramp = linspace(1, 0, n)**rho`、`exp(ramp * (log max - log min) + log min)`。
pub fn sigmas_polyexponential(n: usize, sigma_min: f64, sigma_max: f64, rho: f64) -> Vec<f64> {
    debug_assert!(n >= 1, "n must be >= 1");
    let log_min = sigma_min.ln();
    let lr = sigma_max.ln() - log_min;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lin = if n <= 1 {
            1.0_f64
        } else {
            1.0 - (i as f64 / (n - 1) as f64)
        };
        let ramp = lin.powf(rho);
        out.push((ramp * lr + log_min).exp());
    }
    append_zero(out)
}

/// BasicScheduler：`sigmas[-(steps+1):]`。
pub fn denoise_sigma_tail(full: &[f64], visible_steps: usize) -> Vec<f64> {
    let want = visible_steps.saturating_add(1);
    if full.len() <= want {
        return full.to_vec();
    }
    let start = full.len() - want;
    full[start..].to_vec()
}

pub fn sigma_pair_at(sigmas: &[f64], step: usize) -> Option<(f64, f64)> {
    if step + 1 < sigmas.len() {
        Some((sigmas[step], sigmas[step + 1]))
    } else {
        None
    }
}

#[cfg(not(feature = "nvml"))]
#[inline]
pub fn cuda_memory_stats_bytes() -> (i64, i64, i64) {
    (-1, -1, -1)
}

#[cfg(feature = "nvml")]
static NVML_HANDLE: Lazy<Option<nvml_wrapper::Nvml>> =
    Lazy::new(|| nvml_wrapper::Nvml::init().ok());

/// 读取 **GPU index 0** 的 NVIDIA 驱动显存；失败返回 `(-1,-1,-1)`。
#[cfg(feature = "nvml")]
#[inline]
pub fn cuda_memory_stats_bytes() -> (i64, i64, i64) {
    let Some(ref nvml) = *NVML_HANDLE else {
        return (-1, -1, -1);
    };
    let Ok(dev) = nvml.device_by_index(0) else {
        return (-1, -1, -1);
    };
    let Ok(mi) = dev.memory_info() else {
        return (-1, -1, -1);
    };
    (
        i64::try_from(mi.used).unwrap_or(-1),
        i64::try_from(mi.free).unwrap_or(-1),
        i64::try_from(mi.total).unwrap_or(-1),
    )
}
