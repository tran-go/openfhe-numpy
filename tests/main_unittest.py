"""
OpenFHE-NumPy Test Framework (Consolidated Version)

This module provides a unified CLI for running OpenFHE-NumPy tests.

Usage:
  python -m tests.main_unittest            # run all tests
  python -m tests.main_unittest --debug    # verbose + console logs
  python -m tests.main_unittest --file PATH  # run specific test file

For setup and requirements, see README.md.
"""

# Standard library imports
import atexit
import csv
import logging
import os
import sys
from argparse import ArgumentParser
from contextlib import redirect_stdout, nullcontext
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import unittest

# Third-party imports
import numpy as np
from ast import literal_eval

# OpenFHE imports
import openfhe
import openfhe_numpy as onp
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
    HEStd_128_classic,
    HEStd_192_classic,
    HEStd_256_classic,
    HEStd_NotSet,
    UNIFORM_TERNARY,
)

# ===============================
# Constants and Paths
# ===============================
TESTS_DIR = Path(__file__).parent
PARAMS_CSV = TESTS_DIR / "ckks_params.csv"
LOG_DIR = TESTS_DIR / "logs"
DEBUG_LOG_DIR = TESTS_DIR / "debug_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_LOG = LOG_DIR / "test_results.log"
DEBUG_LOG = DEBUG_LOG_DIR / "test_debug.log"

# Single devnull handle for stdout suppression
_devnull = open(os.devnull, "w")
atexit.register(_devnull.close)


# ===============================
# Logger Configuration
# ===============================
def _configure_root_logger() -> None:
    """Configure the root test logger and file handlers."""
    root = logging.getLogger("openfhe_numpy_test")
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh_info = logging.FileHandler(RESULTS_LOG)
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(fmt)
    root.addHandler(fh_info)

    fh_debug = logging.FileHandler(DEBUG_LOG)
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(fmt)
    root.addHandler(fh_debug)


# Initialize root logger once
_default_logger = logging.getLogger("openfhe_numpy_test")
if not _default_logger.handlers:
    _configure_root_logger()
_default_console_added = False


# ===============================
# Test-specific Logger
# ===============================
def get_test_logger(test_name: str) -> logging.Logger:
    """Return a logger for a given test suite, creating file handlers if needed."""
    simple = test_name.lower().removeprefix("test").strip("_")
    lg = logging.getLogger(f"openfhe_numpy_test.{simple}")
    if not lg.handlers:
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        fh1 = logging.FileHandler(LOG_DIR / f"{simple}_results.log")
        fh1.setLevel(logging.INFO)
        fh1.setFormatter(fmt)
        lg.addHandler(fh1)

        fh2 = logging.FileHandler(DEBUG_LOG_DIR / f"{simple}_debug.log")
        fh2.setLevel(logging.DEBUG)
        fh2.setFormatter(fmt)
        lg.addHandler(fh2)

    return lg


# ===============================
# Utility Functions
# ===============================
def suppress_stdout(suppress: bool = True):
    """Context manager: suppress stdout if suppress=True."""
    return redirect_stdout(_devnull) if suppress else nullcontext()


def generate_random_array(
    rows: int,
    cols: Optional[int] = None,
    low: float = 0,
    high: float = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a random NumPy array of shape (rows, cols)."""
    rng = np.random.default_rng(seed)
    cols = cols or rows
    shape = (rows, cols) if cols > 1 else (rows,)
    return rng.uniform(low, high, size=shape)


# ===============================
# Parameter Loading
# ===============================
CAST_MAP: Dict[str, Any] = {
    "ringDim": int,
    "multiplicativeDepth": int,
    "scalingModSize": int,
    "batchSize": int,
    "firstModSize": int,
    "numLargeDigits": int,
    "maxRelinSkDeg": int,
    "digitSize": int,
    "standardDeviation": float,
}


def load_ckks_params() -> List[Dict[str, Any]]:
    """Load and cast CKKS parameters from CSV file."""
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"Missing parameter file {PARAMS_CSV}")
    out: List[Dict[str, Any]] = []
    with open(PARAMS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p: Dict[str, Any] = {}
            for k, v in row.items():
                caster = CAST_MAP.get(k, lambda x: literal_eval(x) if x.isnumeric() else x)
                p[k] = caster(v)
            out.append(p)
    return out


# ===============================
# Crypto Context (cached)
# ===============================
def _params_to_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(params.items()))


@lru_cache(maxsize=None)
def _build_crypto_context(key: Tuple[Tuple[str, Any], ...]) -> Any:
    params = dict(key)
    p = CCParamsCKKSRNS()
    p.SetRingDim(params["ringDim"])
    p.SetMultiplicativeDepth(params["multiplicativeDepth"])
    p.SetScalingModSize(params["scalingModSize"])
    p.SetBatchSize(params["batchSize"])
    p.SetFirstModSize(params["firstModSize"])
    p.SetStandardDeviation(params["standardDeviation"])
    p.SetSecretKeyDist(getattr(openfhe, params["secretKeyDist"]))
    p.SetScalingTechnique(getattr(openfhe, params["scalTech"]))
    p.SetKeySwitchTechnique(getattr(openfhe, params["ksTech"]))
    p.SetSecurityLevel(getattr(openfhe, params["securityLevel"]))
    p.SetNumLargeDigits(params["numLargeDigits"])
    p.SetMaxRelinSkDeg(params["maxRelinSkDeg"])
    p.SetDigitSize(params["digitSize"])

    cc = GenCryptoContext(p)
    for feat in (PKESchemeFeature.PKE, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feat)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    return cc, keys


def gen_crypto_context(params: Dict[str, Any]) -> Any:
    return _build_crypto_context(_params_to_key(params))


# ===============================
# Timer
# ===============================
class Timer:
    def __enter__(self) -> "Timer":
        self.start = datetime.now()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = (datetime.now() - self.start).total_seconds()


# ===============================
# Custom TestResult
# ===============================
class CountingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_count = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1

    def addFailure(self, test, err):
        super().addFailure(test, err)
        lg = get_test_logger(test.__class__.__name__)
        inp = getattr(test, "_last_input", None)
        exp = getattr(test, "_last_expected", None)
        res = getattr(test, "_last_result", None)
        parts = []
        if inp is not None:
            parts.append(f"Input:    {np.array2string(np.array(inp), separator=', ')}")
        if exp is not None:
            parts.append(f"Expected: {np.array2string(np.array(exp), separator=', ')}")
        if res is not None:
            parts.append(f"Result:   {np.array2string(np.array(res), separator=', ')}")
        info = " | ".join(parts)
        tb = self._exc_info_to_string(err, test)
        lg.error("FAIL %s: %s\n%s", test.id(), info, tb)

    def addError(self, test, err):
        super().addError(self, err)
        lg = get_test_logger(test.__class__.__name__)
        inp = getattr(test, "_last_input", None)
        exp = getattr(test, "_last_expected", None)
        parts = []
        if inp is not None:
            parts.append(f"Input:    {np.array2string(np.array(inp), separator=', ')}")
        if exp is not None:
            parts.append(f"Expected: {np.array2string(np.array(exp), separator=', ')}")
        info = " | ".join(parts)
        tb = self._exc_info_to_string(err, test)
        lg.error("ERROR %s: %s\n%s", test.id(), info, tb)


# ===============================
# Base Test Class
# ===============================
class MainUnittest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = get_test_logger(cls.__name__)
        if not hasattr(cls, "_test_cases_generated"):
            has_explicit = any(name.startswith("test_") for name in cls.__dict__)
            if not has_explicit:
                if not hasattr(cls, "_generate_test_cases"):
                    raise NotImplementedError(
                        f"{cls.__name__} must define _generate_test_cases() or have test_ methods"
                    )
                cls._generate_test_cases()
            cls._test_cases_generated = True

    @classmethod
    def run_test_summary(cls, test_name: str = "", debug: bool = False) -> int:
        global _default_console_added
        print(f"Running {test_name} tests...")
        if debug and not _default_console_added:
            ch = logging.StreamHandler(sys.stderr)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _default_logger.addHandler(ch)
            _default_console_added = True

        runner = unittest.TextTestRunner(verbosity=2 if debug else 0, resultclass=CountingResult)
        with Timer() as t:
            result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(cls))

        total, passed = result.testsRun, result.success_count
        failed, errors = len(result.failures), len(result.errors)
        print(
            f"\n{'=' * 60}\n"
            f"{test_name} Summary (time={t.elapsed:.2f}s):\n"
            f"  Total  = {total}\n"
            f"  Passed = {passed}\n"
            f"  Failed = {failed}\n"
            f"  Errors = {errors}\n"
            f"{'=' * 60}"
        )
        return 0 if result.wasSuccessful() else 1

    @classmethod
    def generate_test_case(
        cls,
        func: Any,
        name: str,
        test_name: str,
        params: Dict[str, Any],
        input_data: List[Any],
        expected: Any,
        eps: float = onp.EPSILON,
        debug: bool = False,
    ) -> Any:
        """Generate a test method that logs input, expected, result on failure."""
        lg = get_test_logger(name)
        # stash input and expected
        inp = input_data[0]
        exp = expected

        def test(self):
            # stash for hooks
            self._last_input = inp
            self._last_expected = exp
            try:
                with suppress_stdout(not debug):
                    res = func(params, input_data)
                self._last_result = res
                flag, err = onp.check_equality_matrix(res, exp, eps)
                if flag:
                    lg.info("PASS %s", test_name)
                else:
                    self.fail(f"{test_name}: error={err:.2e}")
            except Exception:
                raise

        test.__name__ = test_name
        return test


# ===============================
# Module Discovery
# ===============================
def find_test_classes(module) -> List[Type[MainUnittest]]:
    """Find all MainUnittest subclasses in a module."""
    return [
        obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, MainUnittest) and obj is not MainUnittest
    ]


def load_test_module(file_path: str) -> Tuple[List[Type[MainUnittest]], str]:
    """Load a test module from file path and find test classes."""
    path = Path(file_path).absolute()

    if not path.exists():
        return [], f"Error: File not found: {path}"

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    test_classes = find_test_classes(module)
    return test_classes, f"Found {len(test_classes)} test classes in {path.name}"


# ===============================
# CLI Entrypoint
# ===============================
if __name__ == "__main__":
    parser = ArgumentParser(description="Run OpenFHE-NumPy tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--file", help="Test file to run")
    args = parser.parse_args()

    if args.file:
        import importlib.util

        test_classes, message = load_test_module(args.file)
        if not test_classes:
            print(message)
            sys.exit(1)

        exit_code = 0
        for cls in test_classes:
            exit_code |= cls.run_test_summary(cls.__name__, debug=args.debug)
        sys.exit(exit_code)
    else:
        name = os.path.basename(sys.argv[0])
        sys.exit(MainUnittest.run_test_summary(name, debug=args.debug))
