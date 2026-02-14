# Metal SDPA Extension — test targets

PYTEST = python -m pytest
TEST_DIR = tests

.PHONY: test test-gate test-gate-fast test-wan test-all

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
