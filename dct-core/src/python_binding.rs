use crate::v_sampler_core;
use numpy::{IntoPyArray, PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::{PyDict, PyModule};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64Mcg;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(build_sigma_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(sigma_pair_instruction, m)?)?;
    m.add_function(wrap_pyfunction!(memory_monitor_tick, m)?)?;
    m.add_function(wrap_pyfunction!(latent_gaussian_noise_f32, m)?)?;
    Ok(())
}

fn pyfun_catch_panic<T>(ctx: &'static str, f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(inner) => inner,
        Err(payload) => {
            let msg: String = match payload.downcast::<String>() {
                Ok(s) => *s,
                Err(p) => match p.downcast::<&'static str>() {
                    Ok(s) => (*s).to_string(),
                    Err(_) => "(opaque panic payload)".to_string(),
                },
            };
            Err(PyRuntimeError::new_err(format!(
                "[vates_sampler_core] {ctx}: Rust panic — {msg}"
            )))
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    scheduler,
    sigma_min,
    sigma_max,
    steps,
    denoise,
    karras_rho = 7.0,
    poly_rho = 1.0
))]
fn build_sigma_schedule<'py>(
    py: Python<'py>,
    scheduler: String,
    sigma_min: f64,
    sigma_max: f64,
    steps: u32,
    denoise: f64,
    karras_rho: f64,
    poly_rho: f64,
) -> PyResult<Bound<'py, PyArray<f64, ndarray::Ix1>>> {
    if steps < 1 {
        return Err(PyValueError::new_err("steps 须 >= 1"));
    }
    if !(sigma_min > 0.0 && sigma_max > 0.0 && sigma_max >= sigma_min) {
        return Err(PyValueError::new_err("sigma_min / sigma_max 须为正且 sigma_max ≥ sigma_min"));
    }
    if denoise <= 0.0 {
        return Err(PyValueError::new_err("denoise 须为正"));
    }

    pyfun_catch_panic("build_sigma_schedule", move || {
        let total_steps = if denoise >= 1.0 {
            steps as usize
        } else {
            v_sampler_core::comfy_scheduler_total_steps(steps, denoise)
        };
        let n = total_steps.max(1);

        let full_sigmas = match scheduler.to_ascii_lowercase().as_str() {
            "karras" => v_sampler_core::sigmas_karras(n, sigma_min, sigma_max, karras_rho),
            "exponential" => v_sampler_core::sigmas_exponential(n, sigma_min, sigma_max),
            "polyexponential" => {
                v_sampler_core::sigmas_polyexponential(n, sigma_min, sigma_max, poly_rho)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Rust σ 编排仅支持 karras / exponential / polyexponential；收到 `{other}`。"
                )));
            }
        };

        let sliced = if denoise < 1.0 {
            v_sampler_core::denoise_sigma_tail(&full_sigmas, steps as usize)
        } else {
            full_sigmas
        };

        if sliced.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "σ 序列长度异常：{} len={}",
                scheduler,
                sliced.len()
            )));
        }

        let vec = ndarray::Array1::from_vec(sliced);
        Ok(vec.into_pyarray_bound(py))
    })
}

#[pyfunction]
#[pyo3(signature = (sigmas, step))]
fn sigma_pair_instruction(sigmas: PyReadonlyArrayDyn<f64>, step: isize) -> PyResult<(f64, f64)> {
    let arr = sigmas.as_array();
    let flat = arr.flatten();
    let s: Vec<f64> = flat.iter().copied().collect();
    let len = s.len();
    if len < 2 {
        return Err(PyValueError::new_err("sigma 向量长度须 >= 2"));
    }
    let idx = if step < 0 {
        usize::try_from(len as isize + step).map_err(|_| PyValueError::new_err("step 索引溢出"))?
    } else {
        step as usize
    };
    v_sampler_core::sigma_pair_at(&s, idx).ok_or_else(|| {
        PyValueError::new_err(format!("无效的 step={step} — sigma 长度为 {len}"))
    })
}

#[pyfunction]
#[pyo3(signature = (pressure_threshold = None))]
fn memory_monitor_tick(py: Python<'_>, pressure_threshold: Option<f64>) -> PyResult<PyObject> {
    pyfun_catch_panic("memory_monitor_tick", || {
        py.allow_threads(|| {
            std::hint::spin_loop();
        });

        let (used, free, total) = v_sampler_core::cuda_memory_stats_bytes();
        let threshold = pressure_threshold.unwrap_or(0.92_f64).clamp(0.0_f64, 1.0_f64);

        let pressure = if total > 0 {
            (used as f64 / total as f64).clamp(0.0, 1.0)
        } else {
            -1.0
        };

        let hot = pressure >= threshold && threshold > 0.0 && pressure >= 0.0;

        let dict = PyDict::new_bound(py);
        dict.set_item("cuda_used_bytes", used)?;
        dict.set_item("cuda_free_bytes", free)?;
        dict.set_item("cuda_total_bytes", total)?;
        dict.set_item("pressure", pressure)?;
        dict.set_item("vacuum_hot", hot)?;
        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(signature = (batch, channels, height, width, seed, stream = 0u64))]
fn latent_gaussian_noise_f32<'py>(
    py: Python<'py>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    seed: u64,
    stream: u64,
) -> PyResult<Bound<'py, PyArray<f32, ndarray::IxDyn>>> {
    if batch < 1 || channels < 1 || height < 1 || width < 1 {
        return Err(PyValueError::new_err("batch、c、h、w 均须为正"));
    }
    let count = batch
        .checked_mul(channels)
        .and_then(|x| x.checked_mul(height))
        .and_then(|x| x.checked_mul(width))
        .ok_or_else(|| PyValueError::new_err("潜空间扁平元素溢出 usize"))?;

    pyfun_catch_panic("latent_gaussian_noise_f32", move || {
        let mut rng = Pcg64Mcg::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ stream);
        let distr = StandardNormal;
        let mut buf = Vec::with_capacity(count);
        for _ in 0..count {
            let x: f64 = distr.sample(&mut rng);
            buf.push(x as f32);
        }
        let shape = ndarray::IxDyn(&[batch, channels, height, width]);
        let arr = ndarray::Array::from_shape_vec(shape, buf).map_err(|e| {
            PyValueError::new_err(format!("noise 缓冲区形状不匹配：{}", e))
        })?;
        Ok(arr.into_pyarray_bound(py))
    })
}
