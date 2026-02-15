"""
Buffer lifetime and RAII cleanup test for native Metal SDPA bridge.

Validates that MFA buffer wrappers (mfa_buffer_t) are properly destroyed
after each call, including on exception paths. The RAII _BufferGuard in
metal_sdpa_backend.cpp (aa1b7ce) ensures cleanup.

This test exercises:
1. Repeated calls to check for handle accumulation (leak detection)
2. Mixed dtype/size sequences that stress the buffer allocation path
3. Error paths that should trigger RAII cleanup

Usage:
    pytest tests/test_buffer_lifetime.py -v
"""

import gc
import sys

import pytest
import torch

import metal_sdpa_extension


@pytest.fixture
def metal_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    return torch.device("mps")


def _get_mps_allocated():
    """Get current MPS allocated memory in bytes."""
    if hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory()
    return 0


def _run_sdpa(q, k, v, mask=None):
    """Run native bridge SDPA with sync."""
    torch.mps.synchronize()
    out = metal_sdpa_extension.metal_scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False,
    )
    torch.mps.synchronize()
    return out


class TestBufferLifetime:
    """Verify MFA buffer wrappers are properly cleaned up."""

    def test_repeated_calls_no_leak(self, metal_device):
        """Many repeated calls should not accumulate leaked buffers.

        If mfa_buffer_t handles leak (missing mfa_destroy_buffer), each call
        would retain ~4 buffer wrappers. After 100 calls we'd have 400 leaked
        handles, which would show as growing MPS allocation.
        """
        B, H, N, D = 1, 4, 256, 64
        dtype = torch.float16
        q = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)

        # Warmup
        for _ in range(3):
            _run_sdpa(q, k, v)

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        baseline = _get_mps_allocated()

        # Run many iterations
        ITERS = 100
        for _ in range(ITERS):
            out = _run_sdpa(q, k, v)
            del out

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        after = _get_mps_allocated()

        # Allow 10 MB growth for MPS allocator overhead (pool granularity),
        # but not the ~100+ MB that 400 leaked buffer wrappers would cause.
        growth_mb = (after - baseline) / (1024 * 1024)
        assert growth_mb < 10.0, (
            f"Possible buffer leak: {growth_mb:.1f} MB growth after {ITERS} calls "
            f"(baseline={baseline / (1024**2):.1f} MB, after={after / (1024**2):.1f} MB)"
        )

    def test_mixed_dtype_sizes_no_leak(self, metal_device):
        """Mixed dtype and size calls should properly destroy all buffers."""
        configs = [
            (1, 2, 64, 64, torch.float16),
            (1, 4, 128, 128, torch.float32),
            (1, 8, 256, 64, torch.float16),
            (1, 1, 32, 64, torch.float32),
            (2, 4, 64, 64, torch.float16),
        ]

        # Warmup
        for B, H, N, D, dt in configs:
            q = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
            k = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
            v = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
            _run_sdpa(q, k, v)
            del q, k, v

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        baseline = _get_mps_allocated()

        # Run mixed sequence 20 times
        for _ in range(20):
            for B, H, N, D, dt in configs:
                q = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
                k = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
                v = torch.randn(B, H, N, D, device=metal_device, dtype=dt)
                out = _run_sdpa(q, k, v)
                del q, k, v, out

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        after = _get_mps_allocated()

        growth_mb = (after - baseline) / (1024 * 1024)
        assert growth_mb < 20.0, (
            f"Possible buffer leak with mixed configs: {growth_mb:.1f} MB growth "
            f"after 100 calls across 5 configs"
        )

    def test_masked_calls_cleanup(self, metal_device):
        """Calls with masks (which create additional temp buffers) should clean up."""
        B, H, N, D = 1, 4, 128, 64
        dtype = torch.float16
        q = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)

        masks = [
            torch.rand(1, 1, N, N, device=metal_device) > 0.2,  # bool
            torch.randn(1, 1, N, N, device=metal_device, dtype=dtype),  # float
        ]

        # Warmup
        for mask in masks:
            _run_sdpa(q, k, v, mask=mask)

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        baseline = _get_mps_allocated()

        ITERS = 50
        for i in range(ITERS):
            mask = masks[i % len(masks)]
            out = _run_sdpa(q, k, v, mask=mask)
            del out

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        after = _get_mps_allocated()

        growth_mb = (after - baseline) / (1024 * 1024)
        assert growth_mb < 10.0, (
            f"Possible mask buffer leak: {growth_mb:.1f} MB growth after {ITERS} "
            f"masked calls"
        )

    def test_error_path_cleanup(self, metal_device):
        """Buffer cleanup should happen even when the bridge raises an error.

        We provoke an error by passing an unsupported dtype (bfloat16 if the
        bridge doesn't support it) or invalid shapes.
        """
        B, H, N, D = 1, 4, 64, 64

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        baseline = _get_mps_allocated()

        # Try to provoke errors 20 times
        error_count = 0
        for _ in range(20):
            try:
                # BF16 may or may not be supported; if it errors, RAII should clean up
                q = torch.randn(B, H, N, D, device=metal_device, dtype=torch.bfloat16)
                k = torch.randn(B, H, N, D, device=metal_device, dtype=torch.bfloat16)
                v = torch.randn(B, H, N, D, device=metal_device, dtype=torch.bfloat16)
                _run_sdpa(q, k, v)
                del q, k, v
            except (RuntimeError, Exception):
                error_count += 1

        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        after = _get_mps_allocated()

        # If BF16 is supported (no errors), the test still validates no leak
        growth_mb = (after - baseline) / (1024 * 1024)
        assert growth_mb < 10.0, (
            f"Possible buffer leak on error path: {growth_mb:.1f} MB growth "
            f"after {error_count} errors out of 20 attempts"
        )
