# Metal SDPA Extension — test targets

PYTHON ?= python3
PYTEST = $(PYTHON) -m pytest
TEST_DIR = tests
QWEN_ROOT ?= /Users/tkhan/qwen3-coder-next-mac
RC3_REPORT_JSON ?= /tmp/rc3_gate_report.json
RC3_REPORT_MD ?= /tmp/rc3_gate_report.md

.PHONY: test test-gate test-gate-fast test-wan test-all rc3-gate

# Run all tests
test:
	$(PYTEST) $(TEST_DIR) -v

# Full regression gate (48-combo parity + WAN stress + rollout controls)
test-gate:
	$(PYTEST) $(TEST_DIR)/test_regression_gate.py -v

# Regression gate without WAN stress tests (fast, ~10s)
test-gate-fast:
	$(PYTEST) $(TEST_DIR)/test_regression_gate.py -v -m "not slow"

# WAN stress tests only
test-wan:
	$(PYTEST) $(TEST_DIR)/test_regression_gate.py -v -m "slow"

# Everything: all test files
test-all:
	$(PYTEST) $(TEST_DIR) -v --tb=long

# RC3 strict comparability/perf gate for WAN soak A/B
rc3-gate:
	$(PYTHON) tools/rc3_harness.py \
		--workspace-root "$(QWEN_ROOT)" \
		--jobs 1 \
		--baseline-threshold 32768 \
		--candidate-threshold 1024 \
		--output-json "$(RC3_REPORT_JSON)" \
		--output-md "$(RC3_REPORT_MD)"
