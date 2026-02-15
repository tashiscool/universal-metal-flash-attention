#!/usr/bin/env python3
"""
RC3 strict comparability/perf gate for WAN soak A/B runs.

This harness can either:
1) Launch fresh A/B soak runs, or
2) Parse existing soak result directories.

Outputs:
- JSON report (machine-readable gate results)
- Markdown report (human summary)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULTS_RE = re.compile(r"^Results:\s+(\S+)\s*$", re.MULTILINE)
JOBS_RE = re.compile(r"Jobs:\s+(\d+)\s+\(pass=(\d+)\s+fail=(\d+)\)")
ROUTE_RE = re.compile(r"\[MPS ATTN\] route=([^ ]+).*?n_kv=(\d+)")
METRICS_RE = re.compile(r"\[MPS ATTN METRICS\].*")
NATIVE_METRICS_RE = re.compile(
    r"native_bridge=(\d+) \(avg=([0-9.]+)ms p50=([0-9.]+)ms p95=([0-9.]+)ms\)"
)
SUBQ_METRICS_RE = re.compile(
    r"sub_quad=(\d+) \(avg=([0-9.]+)ms p50=([0-9.]+)ms p95=([0-9.]+)ms\)"
)
TIMING_TOTAL_RE = re.compile(
    r"\[METAL_SDPA_TIMING\] total=([0-9.]+)ms .*? Nkv=(\d+) .*"
)
FALLBACK_RE = re.compile(
    r"(native_exception:|top_level_exception:|top_level_unknown_exception|route=torch_sdpa)"
)
WATCHDOG_RE = re.compile(
    r"\*\*\*\s+(?:SWAP WATCHDOG|MEMORY PRESSURE)(?:\s*\(|:)",
    re.IGNORECASE,
)
RUNTIME_ERROR_RE = re.compile(
    r"(!!! Exception during processing !!!|RuntimeError: MPS backend out of memory|Traceback \(most recent call last\))"
)
_COMFY_PYTHON_CACHE: str | None = None


@dataclass
class SoakRun:
    label: str
    results_dir: str
    jobs_total: int
    jobs_pass: int
    jobs_fail: int
    per_job_times_s: list[float]
    per_job_swap_deltas_gb: list[float]
    route_counts_total: dict[str, int]
    route_counts_by_nkv: dict[str, dict[str, int]]
    fallback_markers: int
    watchdog_markers: int
    runtime_error_markers: int
    native_metrics: dict[str, float] | None
    subquad_metrics: dict[str, float] | None
    timing_total_by_nkv: dict[str, list[float]]

    def nkv_total(self, nkv: int) -> int:
        by_route = self.route_counts_by_nkv.get(str(nkv), {})
        return int(sum(by_route.values()))

    def max_swap_delta_gb(self) -> float:
        if not self.per_job_swap_deltas_gb:
            return 0.0
        return float(max(self.per_job_swap_deltas_gb))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    x = (len(s) - 1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return float(s[lo])
    frac = x - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _parse_float_list(line_value: str) -> list[float]:
    out: list[float] = []
    for token in line_value.split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _extract_results_dir(text: str) -> str:
    m = RESULTS_RE.search(text)
    if not m:
        raise RuntimeError("Could not find `Results:` line in soak output")
    return m.group(1).strip()


def _python_can_import_torch(python_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def _resolve_comfy_python() -> str:
    global _COMFY_PYTHON_CACHE
    if _COMFY_PYTHON_CACHE:
        return _COMFY_PYTHON_CACHE

    env_override = os.environ.get("COMFY_PYTHON", "").strip()
    if env_override:
        _COMFY_PYTHON_CACHE = env_override
        return _COMFY_PYTHON_CACHE

    candidates: list[str] = []
    for candidate in (
        os.environ.get("PYTHON", "").strip(),
        "python",
        "python3",
        "/Users/tkhan/anaconda3/bin/python",
        sys.executable,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if _python_can_import_torch(candidate):
            _COMFY_PYTHON_CACHE = candidate
            return _COMFY_PYTHON_CACHE

    _COMFY_PYTHON_CACHE = sys.executable
    return _COMFY_PYTHON_CACHE


def _parse_metrics(lines: list[str]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    native: dict[str, float] | None = None
    subq: dict[str, float] | None = None
    for line in lines:
        nm = NATIVE_METRICS_RE.search(line)
        if nm:
            native = {
                "count": float(nm.group(1)),
                "avg_ms": float(nm.group(2)),
                "p50_ms": float(nm.group(3)),
                "p95_ms": float(nm.group(4)),
            }
        sm = SUBQ_METRICS_RE.search(line)
        if sm:
            subq = {
                "count": float(sm.group(1)),
                "avg_ms": float(sm.group(2)),
                "p50_ms": float(sm.group(3)),
                "p95_ms": float(sm.group(4)),
            }
    return native, subq


def parse_soak_run(label: str, results_dir: str) -> SoakRun:
    rdir = Path(results_dir)
    summary_path = rdir / "summary.txt"
    log_path = rdir / "comfyui.log"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    summary = summary_path.read_text(errors="ignore")
    log_text = log_path.read_text(errors="ignore")

    jobs_total = jobs_pass = jobs_fail = 0
    jobs_match = JOBS_RE.search(summary)
    if jobs_match:
        jobs_total = int(jobs_match.group(1))
        jobs_pass = int(jobs_match.group(2))
        jobs_fail = int(jobs_match.group(3))

    per_job_times_s: list[float] = []
    per_job_swap_deltas_gb: list[float] = []
    for line in summary.splitlines():
        if line.startswith("Per-job times (seconds):"):
            per_job_times_s = _parse_float_list(line.split(":", 1)[1].strip())
        elif line.startswith("Per-job swap deltas (GB):"):
            per_job_swap_deltas_gb = _parse_float_list(line.split(":", 1)[1].strip())

    route_counts_total: dict[str, int] = {}
    route_counts_by_nkv: dict[str, dict[str, int]] = {}
    for route, nkv in ROUTE_RE.findall(log_text):
        route_counts_total[route] = route_counts_total.get(route, 0) + 1
        nkv_map = route_counts_by_nkv.setdefault(str(nkv), {})
        nkv_map[route] = nkv_map.get(route, 0) + 1

    metrics_lines = METRICS_RE.findall(log_text)
    native_metrics, subquad_metrics = _parse_metrics(metrics_lines)

    timing_total_by_nkv: dict[str, list[float]] = {}
    for total_ms, nkv in TIMING_TOTAL_RE.findall(log_text):
        timing_total_by_nkv.setdefault(str(nkv), []).append(float(total_ms))

    return SoakRun(
        label=label,
        results_dir=str(rdir),
        jobs_total=jobs_total,
        jobs_pass=jobs_pass,
        jobs_fail=jobs_fail,
        per_job_times_s=per_job_times_s,
        per_job_swap_deltas_gb=per_job_swap_deltas_gb,
        route_counts_total=route_counts_total,
        route_counts_by_nkv=route_counts_by_nkv,
        fallback_markers=len(FALLBACK_RE.findall(log_text)),
        watchdog_markers=len(WATCHDOG_RE.findall(log_text)),
        runtime_error_markers=len(RUNTIME_ERROR_RE.findall(log_text)),
        native_metrics=native_metrics,
        subquad_metrics=subquad_metrics,
        timing_total_by_nkv=timing_total_by_nkv,
    )


def run_soak_and_get_results_dir(
    label: str,
    workspace_root: Path,
    soak_script: Path,
    cmd_extra: list[str],
    env_extra: dict[str, str] | None = None,
) -> str:
    cmd = ["bash", str(soak_script)] + cmd_extra
    env = os.environ.copy()
    env.setdefault("COMFY_PYTHON", _resolve_comfy_python())
    if env_extra:
        env.update(env_extra)

    print(f"[{label}] Running: {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    rc = proc.wait()
    output_text = "".join(output_lines)
    if rc != 0:
        raise RuntimeError(f"Soak run `{label}` failed with exit code {rc}")
    return _extract_results_dir(output_text)


def to_dict(run: SoakRun) -> dict[str, Any]:
    return {
        "label": run.label,
        "results_dir": run.results_dir,
        "jobs_total": run.jobs_total,
        "jobs_pass": run.jobs_pass,
        "jobs_fail": run.jobs_fail,
        "per_job_times_s": run.per_job_times_s,
        "per_job_swap_deltas_gb": run.per_job_swap_deltas_gb,
        "route_counts_total": run.route_counts_total,
        "route_counts_by_nkv": run.route_counts_by_nkv,
        "fallback_markers": run.fallback_markers,
        "watchdog_markers": run.watchdog_markers,
        "runtime_error_markers": run.runtime_error_markers,
        "native_metrics": run.native_metrics,
        "subquad_metrics": run.subquad_metrics,
        "timing_total_by_nkv": run.timing_total_by_nkv,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="RC3 strict comparability/perf gate harness")
    ap.add_argument("--workspace-root", default="/Users/tkhan/qwen3-coder-next-mac")
    ap.add_argument("--soak-script", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--baseline-threshold", type=int, default=32768)
    ap.add_argument("--candidate-threshold", type=int, default=1024)
    ap.add_argument("--candidate-timing", action="store_true")
    ap.add_argument("--baseline-results-dir", default="")
    ap.add_argument("--candidate-results-dir", default="")
    ap.add_argument("--target-nkv", type=int, default=13824)
    ap.add_argument("--short-nkv", type=int, default=512)
    ap.add_argument("--call-volume-tolerance", type=float, default=0.10)
    ap.add_argument("--swap-gate-gb", type=float, default=2.0)
    ap.add_argument("--prev-native-p50-ms", type=float, default=None)
    ap.add_argument("--require-improvement", action="store_true")
    ap.add_argument("--output-json", default="/tmp/rc3_gate_report.json")
    ap.add_argument("--output-md", default="/tmp/rc3_gate_report.md")
    args = ap.parse_args()

    workspace_root = Path(args.workspace_root)
    soak_script = Path(args.soak_script) if args.soak_script else workspace_root / "soak_test.sh"
    if not soak_script.exists():
        raise FileNotFoundError(f"Soak script not found: {soak_script}")
    print(f"[rc3_harness] COMFY_PYTHON={_resolve_comfy_python()}")

    if args.baseline_results_dir:
        baseline_results_dir = args.baseline_results_dir
        print(f"[baseline] Parsing existing results: {baseline_results_dir}")
    else:
        baseline_cmd = ["--jobs", str(args.jobs), "--baseline"]
        if args.baseline_threshold != 32768:
            baseline_cmd += ["--fa-threshold", str(args.baseline_threshold)]
        baseline_results_dir = run_soak_and_get_results_dir(
            label="baseline",
            workspace_root=workspace_root,
            soak_script=soak_script,
            cmd_extra=baseline_cmd,
        )

    if args.candidate_results_dir:
        candidate_results_dir = args.candidate_results_dir
        print(f"[candidate] Parsing existing results: {candidate_results_dir}")
    else:
        candidate_cmd = ["--jobs", str(args.jobs), "--fa-threshold", str(args.candidate_threshold)]
        candidate_env = {"METAL_SDPA_TIMING": "1"} if args.candidate_timing else {}
        candidate_results_dir = run_soak_and_get_results_dir(
            label="candidate",
            workspace_root=workspace_root,
            soak_script=soak_script,
            cmd_extra=candidate_cmd,
            env_extra=candidate_env,
        )

    baseline = parse_soak_run("baseline", baseline_results_dir)
    candidate = parse_soak_run("candidate", candidate_results_dir)

    target = int(args.target_nkv)
    short = int(args.short_nkv)
    tol = float(args.call_volume_tolerance)

    base_target_calls = baseline.nkv_total(target)
    cand_target_calls = candidate.nkv_total(target)
    base_short_calls = baseline.nkv_total(short)
    cand_short_calls = candidate.nkv_total(short)

    target_ratio = (cand_target_calls / base_target_calls) if base_target_calls else float("nan")
    short_ratio = (cand_short_calls / base_short_calls) if base_short_calls else float("nan")

    comparability_target_ok = (
        base_target_calls > 0
        and cand_target_calls > 0
        and abs(target_ratio - 1.0) <= tol
    )
    comparability_short_ok = (
        base_short_calls == 0
        or (cand_short_calls > 0 and abs(short_ratio - 1.0) <= tol)
    )
    comparability_ok = comparability_target_ok and comparability_short_ok

    candidate_native_p50 = (
        float(candidate.native_metrics["p50_ms"]) if candidate.native_metrics else float("nan")
    )
    improvement_ok: bool | None
    if args.prev_native_p50_ms is None:
        improvement_ok = None
    else:
        improvement_ok = (
            not math.isnan(candidate_native_p50)
            and candidate_native_p50 <= float(args.prev_native_p50_ms) * 0.90
        )

    baseline_pass_ok = baseline.jobs_fail == 0 and baseline.jobs_pass == baseline.jobs_total
    candidate_pass_ok = candidate.jobs_fail == 0 and candidate.jobs_pass == candidate.jobs_total
    fallback_ok = candidate.fallback_markers == 0
    watchdog_ok = candidate.watchdog_markers == 0
    runtime_errors_ok = candidate.runtime_error_markers == 0
    swap_ok = candidate.max_swap_delta_gb() <= float(args.swap_gate_gb)

    required_checks = [
        baseline_pass_ok,
        candidate_pass_ok,
        fallback_ok,
        watchdog_ok,
        runtime_errors_ok,
        swap_ok,
        comparability_ok,
    ]
    if args.require_improvement:
        required_checks.append(bool(improvement_ok))
    overall_pass = all(required_checks)

    report = {
        "config": {
            "workspace_root": str(workspace_root),
            "soak_script": str(soak_script),
            "jobs": args.jobs,
            "baseline_threshold": args.baseline_threshold,
            "candidate_threshold": args.candidate_threshold,
            "candidate_timing": args.candidate_timing,
            "target_nkv": target,
            "short_nkv": short,
            "call_volume_tolerance": tol,
            "swap_gate_gb": args.swap_gate_gb,
            "prev_native_p50_ms": args.prev_native_p50_ms,
            "require_improvement": args.require_improvement,
        },
        "baseline": to_dict(baseline),
        "candidate": to_dict(candidate),
        "comparability": {
            "target_nkv": {
                "baseline_calls": base_target_calls,
                "candidate_calls": cand_target_calls,
                "ratio": target_ratio,
                "ok": comparability_target_ok,
            },
            "short_nkv": {
                "baseline_calls": base_short_calls,
                "candidate_calls": cand_short_calls,
                "ratio": short_ratio,
                "ok": comparability_short_ok,
            },
            "overall_ok": comparability_ok,
        },
        "gates": {
            "baseline_pass_rate_ok": baseline_pass_ok,
            "candidate_pass_rate_ok": candidate_pass_ok,
            "fallback_ok": fallback_ok,
            "watchdog_ok": watchdog_ok,
            "runtime_errors_ok": runtime_errors_ok,
            "swap_ok": swap_ok,
            "comparability_ok": comparability_ok,
            "improvement_ok": improvement_ok,
            "overall_pass": overall_pass,
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    cand_native_p50_s = f"{candidate_native_p50:.1f}" if not math.isnan(candidate_native_p50) else "N/A"
    cand_subq_p50 = (
        f"{candidate.subquad_metrics['p50_ms']:.1f}"
        if candidate.subquad_metrics
        else "N/A"
    )
    base_subq_p50 = (
        f"{baseline.subquad_metrics['p50_ms']:.1f}"
        if baseline.subquad_metrics
        else "N/A"
    )
    md_lines = [
        "# RC3 Gate Report",
        "",
        f"- Overall pass: **{'PASS' if overall_pass else 'FAIL'}**",
        f"- JSON report: `{out_json}`",
        "",
        "## Run Summary",
        "",
        "| Run | Jobs pass/fail | Job times (s) | Max swap delta (GB) | native p50 (ms) | sub_quad p50 (ms) |",
        "|---|---:|---|---:|---:|---:|",
        f"| Baseline | {baseline.jobs_pass}/{baseline.jobs_total} (fail={baseline.jobs_fail}) | {baseline.per_job_times_s} | {baseline.max_swap_delta_gb():.2f} | N/A | {base_subq_p50} |",
        f"| Candidate | {candidate.jobs_pass}/{candidate.jobs_total} (fail={candidate.jobs_fail}) | {candidate.per_job_times_s} | {candidate.max_swap_delta_gb():.2f} | {cand_native_p50_s} | {cand_subq_p50} |",
        "",
        "## Comparability",
        "",
        f"- Target Nkv={target}: baseline={base_target_calls}, candidate={cand_target_calls}, ratio={target_ratio:.3f}, ok={comparability_target_ok}",
        f"- Short  Nkv={short}: baseline={base_short_calls}, candidate={cand_short_calls}, ratio={short_ratio:.3f}, ok={comparability_short_ok}",
        "",
        "## Gates",
        "",
        f"- baseline pass-rate gate: {baseline_pass_ok}",
        f"- candidate pass-rate gate: {candidate_pass_ok}",
        f"- fallback gate: {fallback_ok} (candidate markers={candidate.fallback_markers})",
        f"- watchdog gate: {watchdog_ok} (candidate markers={candidate.watchdog_markers})",
        f"- runtime-error gate: {runtime_errors_ok} (candidate markers={candidate.runtime_error_markers})",
        f"- swap gate (max <= {args.swap_gate_gb} GB): {swap_ok}",
        f"- comparability gate: {comparability_ok}",
    ]
    if args.prev_native_p50_ms is not None:
        md_lines.append(
            f"- improvement gate (candidate p50 <= 90% of {args.prev_native_p50_ms}ms): {improvement_ok}"
        )
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[rc3_harness] Wrote JSON: {out_json}")
    print(f"[rc3_harness] Wrote Markdown: {out_md}")
    print(f"[rc3_harness] Overall: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
