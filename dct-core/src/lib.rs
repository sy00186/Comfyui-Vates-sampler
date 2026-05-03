//! Vates **V-Sampler**：PyO3 扩展 `vates_sampler_core`（σ 编排、RNG、可选 NVML）。

mod v_sampler_core;
mod python_binding;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModule;

#[cfg(feature = "python")]
#[pymodule]
fn vates_sampler_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_binding::register(m)
}
