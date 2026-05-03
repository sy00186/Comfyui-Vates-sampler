#!/usr/bin/env python3
"""
编译并对齐 **V-Sampler** 原生扩展 `vates_sampler_core`，使命令行当前 Python 可 `import vates_sampler_core`。

在 **`dct-core`** 目录执行：`python install.py`（或通过 ComfyUI 使用的同一解释器）。
可选 NVML：**`cargo build --release --features python,nvml`**（或设置环境变量后由脚本追加参数，见源码）。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "target" / "release"

from v_sampler_repo_meta import expected_sampler_core_version


def _copy_native_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32" and dst.is_file():
        fd, tmp_name = tempfile.mkstemp(suffix=dst.suffix, dir=str(dst.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return
    shutil.copy2(src, dst)


def _post_copy_dest_dirs() -> list[Path]:
    raw = (os.environ.get("VATES_SAMPLER_POST_COPY_DIR") or os.environ.get("VATES_POST_COPY_DIR") or "").strip()
    if not raw:
        return []
    return [Path(p).strip().expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]


def _mirror_aligned_files_to_post_copy(aligned_files: list[str]) -> None:
    dirs = _post_copy_dest_dirs()
    for d in dirs:
        if not d.is_dir():
            print(f"[V-Sampler install] 警告: POST_COPY_DIR 不可用: {d}", flush=True)
            continue
        for fp in aligned_files:
            src = Path(fp)
            if not src.is_file():
                continue
            dst = d / src.name
            _copy_native_artifact(src, dst)
            print(f"[V-Sampler install] 已镜像: {dst}", flush=True)


def _pythonpath_for_verify() -> str:
    parts = [str(ROOT)] + [str(p) for p in _post_copy_dest_dirs()]
    prev = os.environ.get("PYTHONPATH", "")
    if prev:
        parts.append(prev)
    return os.pathsep.join(parts)


def _pick_windows_native_artifact(release_dir: Path) -> Path | None:
    if not release_dir.is_dir():
        return None
    tagged = list(release_dir.glob("vates_sampler_core.cp*.pyd"))
    if tagged:
        return max(tagged, key=lambda p: p.stat().st_mtime)
    for name in ("vates_sampler_core.dll", "libvates_sampler_core.dll"):
        p = release_dir / name
        if p.is_file():
            return p
    plain = release_dir / "vates_sampler_core.pyd"
    if plain.is_file():
        return plain
    return None


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _sampler_core_version_usable() -> bool:
    if not _has_module("vates_sampler_core"):
        return False
    try:
        import vates_sampler_core as vsc  # noqa: PLC0415
    except ImportError:
        return False
    return getattr(vsc, "__version__", None) == expected_sampler_core_version()


def _cargo_build(extra_features: str | None = None) -> bool:
    if shutil.which("cargo") is None or shutil.which("rustc") is None:
        print("[V-Sampler install] 需要 Rust：https://rustup.rs/", flush=True)
        return False
    feats = "python"
    if extra_features:
        feats = f"{feats},{extra_features.strip().strip(',')}"
    cmd = ["cargo", "build", "--release", "--no-default-features", "--features", feats]
    print(f"[V-Sampler install] {' '.join(cmd)}  cwd={ROOT}", flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
        return True
    except subprocess.CalledProcessError:
        return False


def _align_native_artifacts() -> list[str]:
    out: list[str] = []
    if not RELEASE.is_dir():
        return out
    if sys.platform == "win32":
        src = _pick_windows_native_artifact(RELEASE)
        if src is not None:
            dst = ROOT / "vates_sampler_core.pyd"
            _copy_native_artifact(src, dst)
            out.append(str(dst))
        return out
    if sys.platform == "darwin":
        for name in ("libvates_sampler_core.dylib", "libvates_sampler_core.so"):
            p = RELEASE / name
            if p.is_file():
                dst = ROOT / "vates_sampler_core.so"
                _copy_native_artifact(p, dst)
                stat_chmod_exe(dst)
                out.append(str(dst))
                break
        return out
    for name in ("libvates_sampler_core.so", "vates_sampler_core.so"):
        p = RELEASE / name
        if p.is_file():
            dst = ROOT / "vates_sampler_core.so"
            _copy_native_artifact(p, dst)
            stat_chmod_exe(dst)
            out.append(str(dst))
            break
    return out


def stat_chmod_exe(dst: Path) -> None:
    try:
        os.chmod(
            dst,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
    except OSError:
        pass


def _try_maturin() -> bool:
    if shutil.which("maturin") is None:
        return False
    cmd = ["maturin", "develop", "--release"]
    print(f"[V-Sampler install] {' '.join(cmd)}", flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
        return _sampler_core_version_usable()
    except subprocess.CalledProcessError:
        return False


def _verify_import_subprocess() -> bool:
    exp = expected_sampler_core_version()
    script = (
        "import vates_sampler_core as v; "
        f"exp={exp!r}; "
        "assert getattr(v,'__version__',None)==exp; "
        "assert callable(v.build_sigma_schedule); "
        "assert callable(v.memory_monitor_tick); "
        "assert callable(v.latent_gaussian_noise_f32)"
    )
    try:
        subprocess.check_call(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": _pythonpath_for_verify()},
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    print(
        f"[V-Sampler install] Python {sys.version.split()[0]}, platform={sys.platform}, cwd={os.getcwd()}",
        flush=True,
    )
    if _sampler_core_version_usable():
        print("[V-Sampler install] vates_sampler_core 已就绪。", flush=True)
        return 0

    nvml_extra = os.environ.get("VATES_SAMPLER_BUILD_NVML", "").lower() in ("1", "true", "yes")
    extra_feats = "nvml" if nvml_extra else None

    if _cargo_build(extra_feats):
        copied = _align_native_artifacts()
        if copied:
            for p in copied:
                print(f"[V-Sampler install] aligned: {p}", flush=True)
            _mirror_aligned_files_to_post_copy(copied)
        if _verify_import_subprocess():
            print("[V-Sampler install] 构建并通过 import 校验。", flush=True)
            return 0
        print("[V-Sampler install] cargo 后对齐仍未通过 import。", flush=True)
    elif nvml_extra:
        print("[V-Sampler install] NVML 构建失败；回退为基础 python …", flush=True)
        if _cargo_build(None):
            copied = _align_native_artifacts()
            if copied:
                _mirror_aligned_files_to_post_copy(copied)
            if _verify_import_subprocess():
                return 0

    if _try_maturin():
        print("[V-Sampler install] maturin develop OK。", flush=True)
        return 0

    print(
        "[V-Sampler install] 失败：请在本目录手动执行 cargo build/maturin，并重启 ComfyUI。\n",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
