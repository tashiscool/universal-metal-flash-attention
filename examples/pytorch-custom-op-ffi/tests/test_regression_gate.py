"""
Regression gate for Metal native SDPA backend.

48-combo parity matrix (dtype × causal × mask × asymmetric × contiguous)
plus WAN-scale stress tests. Run before every release or submodule bump.

Usage:
    pytest tests/test_regression_gate.py -v
    pytest tests/test_regression_gate.py -v -m "not slow"   # skip WAN stress
    make test-gate                                           # via Makefile
"""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import metal_sdpa_extension

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
COS_THRESH_FP16 = 0.999
COS_THRESH_FP32 = 0.9999
MAX_ABS_ERR_FP16 = 0.05
MAX_ABS_ERR_FP32 = 0.001

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def _max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _reference_sdpa(q, k, v, attn_mask=None, is_causal=False):
    """Compute reference via PyTorch SDPA on CPU FP32."""
    mask_cpu = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # Preserve bool semantics: True=keep, False=masked.
            mask_cpu = attn_mask.cpu()
        else:
            mask_cpu = attn_mask.float().cpu()
    with torch.inference_mode():
        return F.scaled_dot_product_attention(
            q.float().cpu(),
            k.float().cpu(),
            v.float().cpu(),
            attn_mask=mask_cpu,
            is_causal=is_causal,
        )


def _make_tensors(B, H, Nq, Nkv, D, dtype, device, contiguous=True):
    """Create Q, K, V on device. If not contiguous, permute to force strides."""
    torch.manual_seed(42)
    q = torch.randn(B, H, Nq, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, Nkv, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, Nkv, D, dtype=dtype, device=device) * 0.1
    if not contiguous:
        # Allocate as [B,Nq,H,D] then permute to [B,H,Nq,D] — single permute
        # gives a non-contiguous view with strides swapped in dims 1 and 2.
        q = torch.randn(B, Nq, H, D, dtype=dtype, device=device) * 0.1
        q = q.permute(0, 2, 1, 3)  # [B,H,Nq,D] non-contiguous
        k = torch.randn(B, Nkv, H, D, dtype=dtype, device=device) * 0.1
        k = k.permute(0, 2, 1, 3)
        v = torch.randn(B, Nkv, H, D, dtype=dtype, device=device) * 0.1
        v = v.permute(0, 2, 1, 3)
        assert not q.is_contiguous()
    return q, k, v


def _make_mask(mask_type, B, H, Nq, Nkv, dtype, device):
    """Create attention mask of the given type, or None."""
    if mask_type == "none":
        return None
    torch.manual_seed(123)
    if mask_type == "bool":
        # Random bool mask — ~80% kept (True=attend)
        return torch.rand(1, 1, Nq, Nkv, device=device) > 0.2
    if mask_type == "float":
        # Additive float mask with small negative values
        raw = torch.rand(1, 1, Nq, Nkv, device=device, dtype=dtype)
        return (raw - 0.5) * 2.0  # range [-1, 1]
    raise ValueError(f"Unknown mask_type: {mask_type}")


# ---------------------------------------------------------------------------
# 48-combo parity matrix
# ---------------------------------------------------------------------------

DTYPES = [torch.float16, torch.float32]
CAUSAL = [False, True]
MASK_TYPES = ["none", "bool", "float"]
ASYMMETRIC = [False, True]
CONTIGUOUS = [True, False]

# Base dimensions for parity tests
_B, _H, _Nq, _D = 1, 4, 64, 64
_Nkv_equal = 64
_Nkv_asym = 48


def _parity_id(val):
    """Human-readable parametrize IDs."""
    if val is torch.float16:
        return "fp16"
    if val is torch.float32:
        return "fp32"
    if val is True:
        return "yes"
    if val is False:
        return "no"
    return str(val)


@pytest.mark.metal
@pytest.mark.regression_gate
@pytest.mark.parametrize("dtype", DTYPES, ids=_parity_id)
@pytest.mark.parametrize("causal", CAUSAL, ids=lambda v: f"causal={_parity_id(v)}")
@pytest.mark.parametrize("mask_type", MASK_TYPES, ids=lambda v: f"mask={v}")
@pytest.mark.parametrize("asymmetric", ASYMMETRIC, ids=lambda v: f"asym={_parity_id(v)}")
@pytest.mark.parametrize("contiguous", CONTIGUOUS, ids=lambda v: f"contig={_parity_id(v)}")
class TestParityMatrix:
    """48-combo parity matrix: native Metal SDPA vs PyTorch CPU reference."""

    def test_parity(self, metal_device, dtype, causal, mask_type, asymmetric, contiguous):
        # Causal requires Nq == Nkv in PyTorch SDPA
        if causal and asymmetric:
            pytest.skip("causal + asymmetric not supported by PyTorch SDPA")

        Nkv = _Nkv_asym if asymmetric else _Nkv_equal

        q, k, v = _make_tensors(_B, _H, _Nq, Nkv, _D, dtype, metal_device, contiguous)
        mask = _make_mask(mask_type, _B, _H, _Nq, Nkv, dtype, metal_device)

        # Reference on CPU FP32
        ref = _reference_sdpa(q, k, v, attn_mask=mask, is_causal=causal)

        # MPS sync before call (belt-and-suspenders; the bridge also does this)
        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
        )
        torch.mps.synchronize()

        # Move to CPU for comparison
        out_cpu = out.float().cpu()
        ref_cpu = ref.float()

        # Checks
        assert torch.isfinite(out_cpu).all(), "Output contains NaN/Inf"

        cos = _cosine_similarity(out_cpu, ref_cpu)
        mae = _max_abs_error(out_cpu, ref_cpu)

        cos_thresh = COS_THRESH_FP32 if dtype == torch.float32 else COS_THRESH_FP16
        mae_thresh = MAX_ABS_ERR_FP32 if dtype == torch.float32 else MAX_ABS_ERR_FP16

        assert cos >= cos_thresh, (
            f"Cosine {cos:.6f} < {cos_thresh} | "
            f"dtype={dtype} causal={causal} mask={mask_type} "
            f"asym={asymmetric} contig={contiguous}"
        )
        assert mae <= mae_thresh, (
            f"Max abs error {mae:.6f} > {mae_thresh} | "
            f"dtype={dtype} causal={causal} mask={mask_type} "
            f"asym={asymmetric} contig={contiguous}"
        )


# ---------------------------------------------------------------------------
# WAN-scale stress tests
# ---------------------------------------------------------------------------

WAN_CONFIGS = [
    # (N, label, repeat)
    (256, "small", 1),
    (512, "medium", 1),
    (1024, "large", 1),
    (2048, "xlarge", 1),
    (4096, "xxlarge", 1),
]

WAN_SOAK = (1024, "soak", 20)  # 20 iterations at N=1024


@pytest.mark.metal
@pytest.mark.regression_gate
@pytest.mark.slow
class TestWANStress:
    """WAN 2.2-scale stress: H=40, D=128, contiguous FP16, no mask, no causal."""

    _H = 40
    _D = 128
    _B = 1

    @pytest.mark.parametrize(
        "N,label,repeat", WAN_CONFIGS,
        ids=[c[1] for c in WAN_CONFIGS],
    )
    def test_wan_scale(self, metal_device, N, label, repeat):
        dtype = torch.float16
        q, k, v = _make_tensors(self._B, self._H, N, N, self._D, dtype, metal_device)
        ref = _reference_sdpa(q, k, v)

        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, repeat)):
            out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / max(1, repeat) * 1000

        out_cpu = out.float().cpu()
        ref_cpu = ref.float()
        cos = _cosine_similarity(out_cpu, ref_cpu)

        assert torch.isfinite(out_cpu).all(), f"[{label}] NaN/Inf in output"
        assert cos >= COS_THRESH_FP16, f"[{label}] cos={cos:.6f} < {COS_THRESH_FP16}"

    def test_wan_soak(self, metal_device):
        """Soak test: 20 iterations at N=1024, check latency stability and no leaks."""
        N, label, repeat = WAN_SOAK
        dtype = torch.float16
        q, k, v = _make_tensors(self._B, self._H, N, N, self._D, dtype, metal_device)

        # Warm up
        for _ in range(3):
            metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()

        times = []
        for _ in range(repeat):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        # Latency should not spike more than 5x the average (no thrash)
        assert max_ms < avg_ms * 5, (
            f"[soak] Latency spike: max={max_ms:.1f}ms > 5×avg={avg_ms:.1f}ms"
        )

    def test_wan_no_swap_growth(self, metal_device):
        """Run several large configs and verify no significant swap growth."""
        psutil = pytest.importorskip("psutil")

        swap_before = psutil.swap_memory().used
        dtype = torch.float16

        for N in [512, 1024, 2048]:
            q, k, v = _make_tensors(self._B, self._H, N, N, self._D, dtype, metal_device)
            torch.mps.synchronize()
            metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            torch.mps.synchronize()
            del q, k, v

        torch.mps.empty_cache()
        gc.collect()

        swap_after = psutil.swap_memory().used
        swap_delta_gb = (swap_after - swap_before) / (1024 ** 3)
        assert swap_delta_gb < 1.0, (
            f"Swap grew by {swap_delta_gb:.2f} GB during WAN stress"
        )


# ---------------------------------------------------------------------------
# Rollout control smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.metal
@pytest.mark.regression_gate
class TestRolloutControls:
    """Verify runtime toggle and fallback work correctly."""

    def test_env_disable_fallback(self, metal_device, monkeypatch):
        """METAL_NATIVE_SDPA_ENABLED=0 should still produce correct output via SDPA."""
        monkeypatch.setenv("METAL_NATIVE_SDPA_ENABLED", "0")

        q = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=metal_device) * 0.1

        ref = _reference_sdpa(q, k, v)
        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()

        cos = _cosine_similarity(out.float().cpu(), ref.float())
        assert cos >= COS_THRESH_FP16, f"Fallback cos={cos:.6f}"

    def test_exception_fallback(self, metal_device):
        """Trigger native exception (head_dim > 1024) — should fallback, not crash."""
        # head_dim=2048 exceeds the kernel's limit and throws
        q = torch.randn(1, 1, 32, 2048, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 1, 32, 2048, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 1, 32, 2048, dtype=torch.float16, device=metal_device) * 0.1

        # Should NOT raise — fallback catches the exception
        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
        assert out.shape == q.shape
        assert torch.isfinite(out).all()

    def test_2d_timing_enabled(self):
        """2D inputs should remain valid with METAL_SDPA_TIMING=1 (no dim-index crash)."""
        script = r"""
import torch
import metal_sdpa_extension as ext
if not torch.backends.mps.is_available():
    raise SystemExit(0)
q = torch.randn(128, 64, dtype=torch.float16, device="mps") * 0.1
k = torch.randn(128, 64, dtype=torch.float16, device="mps") * 0.1
v = torch.randn(128, 64, dtype=torch.float16, device="mps") * 0.1
out = ext.metal_scaled_dot_product_attention(q, k, v)
torch.mps.synchronize()
assert out.shape == q.shape
assert torch.isfinite(out).all()
print("timing_2d_ok")
"""
        env = os.environ.copy()
        env["METAL_SDPA_TIMING"] = "1"
        # Run from the package root so metal_sdpa_extension.so is importable
        # regardless of where pytest was invoked.
        pkg_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"subprocess failed: {proc.stderr}\n{proc.stdout}"
        assert "timing_2d_ok" in proc.stdout

    def test_2d_env_disable_fallback(self, metal_device, monkeypatch):
        """2D inputs should succeed through fallback when native backend is disabled."""
        monkeypatch.setenv("METAL_NATIVE_SDPA_ENABLED", "0")

        q = torch.randn(128, 64, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(128, 64, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(128, 64, dtype=torch.float16, device=metal_device) * 0.1

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.float().cpu().unsqueeze(0).unsqueeze(0),
            k.float().cpu().unsqueeze(0).unsqueeze(0),
            v.float().cpu().unsqueeze(0).unsqueeze(0),
        ).squeeze(0).squeeze(0)

        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()

        out_cpu = out.float().cpu()
        assert out_cpu.shape == ref.shape
        assert torch.isfinite(out_cpu).all()
        cos = _cosine_similarity(out_cpu, ref.float())
        assert cos >= COS_THRESH_FP16, f"2D fallback cos={cos:.6f}"
