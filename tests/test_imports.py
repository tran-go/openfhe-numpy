"""Helper module to simplify imports for test files."""

import os
import sys

# Add the parent directory to path to make package imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import and expose what's needed
from tests.main_unittest import MainUnittest, suppress_stdout
from tests.main_unittest import load_ckks_params
from tests.main_unittest import generate_random_array
from tests.main_unittest import gen_crypto_context
from tests.main_unittest import get_cached_crypto_context

# Easy to add new imports as needed
import numpy as np
import openfhe_numpy as onp
import unittest
