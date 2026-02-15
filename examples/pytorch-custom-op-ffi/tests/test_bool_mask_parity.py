"""
Targeted bool mask parity test for native bridge.

Validates that the native bridge follows PyTorch SDPA bool mask semantics:
  - True = attend (keep)
  - False = masked out

This test specifically exercises the convention that ComfyUI's attention.py
relies on after removing the ~attn_mask inversion (c33cd6af).

Usage:
    pytest tests/test_bool_mask_parity.py -v
"""

import pytest
import torch
import torch.nn.functional as F

import metal_sdpa_extension


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


@pytest.fixture
def metal_device():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    return torch.device("mps")


# ---------------------------------------------------------------------------
# Bool mask convention tests
# ---------------------------------------------------------------------------


class TestBoolMaskConvention:
    """Verify native bridge matches PyTorch SDPA bool mask semantics exactly.

    PyTorch SDPA: mask[i,j]=True means Q[i] attends to K[j].
    ComfyUI (after c33cd6af): passes bool mask directly, no inversion.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_all_true_mask_matches_no_mask(self, metal_device, dtype):
        """All-True mask should produce identical output to no mask."""
        B, H, N, D = 1, 4, 64, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        mask = torch.ones(1, 1, N, N, device=metal_device, dtype=torch.bool)

        torch.mps.synchronize()
        out_no_mask = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False,
        )
        torch.mps.synchronize()
        out_all_true = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
        )
        torch.mps.synchronize()

        cos = _cosine_similarity(out_no_mask, out_all_true)
        assert cos >= 0.999, f"All-True mask should match no-mask: cos={cos:.6f}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_partial_bool_mask_matches_pytorch(self, metal_device, dtype):
        """Partial bool mask should produce same output as PyTorch SDPA CPU reference."""
        B, H, N, D = 1, 4, 64, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)

        # ~80% True (attend), 20% False (masked)
        torch.manual_seed(123)
        mask = torch.rand(1, 1, N, N, device=metal_device) > 0.2

        # Reference: PyTorch SDPA on CPU FP32
        ref = F.scaled_dot_product_attention(
            q.float().cpu(), k.float().cpu(), v.float().cpu(),
            attn_mask=mask.cpu(),
        )

        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
        )
        torch.mps.synchronize()

        cos = _cosine_similarity(out.float().cpu(), ref.float())
        thresh = 0.999 if dtype == torch.float16 else 0.9999
        assert cos >= thresh, (
            f"Bool mask parity failed: cos={cos:.6f} < {thresh} (dtype={dtype})"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_inverted_mask_differs(self, metal_device, dtype):
        """Passing ~mask should produce DIFFERENT output than mask.

        This proves the bridge actually uses the mask direction, and confirms
        that the old ComfyUI code (which inverted) was wrong.
        """
        B, H, N, D = 1, 4, 64, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=metal_device, dtype=dtype)

        torch.manual_seed(123)
        mask = torch.rand(1, 1, N, N, device=metal_device) > 0.2

        torch.mps.synchronize()
        out_normal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
        )
        out_inverted = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=~mask, is_causal=False,
        )
        torch.mps.synchronize()

        cos = _cosine_similarity(out_normal, out_inverted)
        assert cos < 0.99, (
            f"Inverted mask should differ significantly: cos={cos:.6f} (too similar)"
        )

    @pytest.mark.parametrize("mask_ndim", [2, 3, 4])
    def test_bool_mask_broadcast_shapes(self, metal_device, mask_ndim):
        """Bool mask with various broadcast shapes should match PyTorch."""
        B, H, Nq, Nkv, D = 1, 4, 64, 48, 64
        dtype = torch.float16
        torch.manual_seed(42)
        q = torch.randn(B, H, Nq, D, device=metal_device, dtype=dtype)
        k = torch.randn(B, H, Nkv, D, device=metal_device, dtype=dtype)
        v = torch.randn(B, H, Nkv, D, device=metal_device, dtype=dtype)

        torch.manual_seed(123)
        if mask_ndim == 2:
            mask = torch.rand(Nq, Nkv, device=metal_device) > 0.2
        elif mask_ndim == 3:
            mask = torch.rand(1, Nq, Nkv, device=metal_device) > 0.2
        else:
            mask = torch.rand(1, 1, Nq, Nkv, device=metal_device) > 0.2

        # Reference
        mask_4d = mask
        if mask_4d.ndim == 2:
            mask_4d = mask_4d.unsqueeze(0).unsqueeze(0)
        elif mask_4d.ndim == 3:
            mask_4d = mask_4d.unsqueeze(1)
        ref = F.scaled_dot_product_attention(
            q.float().cpu(), k.float().cpu(), v.float().cpu(),
            attn_mask=mask_4d.cpu(),
        )

        torch.mps.synchronize()
        out = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
        )
        torch.mps.synchronize()

        cos = _cosine_similarity(out.float().cpu(), ref.float())
        assert cos >= 0.999, (
            f"Bool mask broadcast parity failed: cos={cos:.6f} ndim={mask_ndim}"
        )
