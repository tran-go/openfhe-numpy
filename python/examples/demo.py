# Import OpenFHE and matrix utilities
import numpy as np
from openfhe import *
from openfhe_matrix import *

# Import OpenFHE NumPy-style interface
import openfhe_numpy as onp
from openfhe_numpy.utils import check_equality_matrix
from openfhe_numpy.config import MatrixOrder
import os
import numpy as np
from typing import Tuple


# Import OpenFHE NumPy modules
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils

from openfhe_numpy.utils import get_shape, next_power_of_two, is_power_of_two
from openfhe_numpy.config import MatrixOrder, DataType
from openfhe_numpy.constructors import ravel_matrix


class ctarray(Ciphertext):
    """Ciphertext array representation with shape and encoding metadata."""

    def __init__(
        self,
        data,
        original_shape,
        ndim,
        size,
        ncols,
        order: int = MatrixOrder.ROW_MAJOR,
    ):
        super().__init__(data)
        # object.__setattr__(self, "data", data)
        # object.__setattr__(self, "original_shape", original_shape)
        # object.__setattr__(self, "ndim", ndim)
        # object.__setattr__(self, "ncols", ncols)
        # object.__setattr__(self, "nrows", size // ncols)
        # object.__setattr__(self, "batch_size", size)
        # object.__setattr__(self, "order", order)
        self.original_shape = original_shape
        self.ndim = ndim
        self.ncols = ncols
        self.nrows = size // ncols
        self.size = size
        self.order = order

        # self._data = data

    # def __getattr__(self, name):
    #     # Forward attribute/method calls to self._ct
    #     return getattr(self._data, name)

    # def __setattr__(self, name, value):
    #     # Forward attribute setting to self._ct (except _ct itself)
    #     if name == {
    #         "data",
    #         "original_shape",
    #         "ndim",
    #         "ncols",
    #         "nrows",
    #         "batch_size",
    #         "encoding_size",
    #     }:
    #         object.__setattr__(self, name, value)
    #     else:
    #         setattr(self._data, name, value)

    # def __setattr__(self, name, value):
    #     if name in {"_ct", "shape"}:
    #         object.__setattr__(self, name, value)
    #     else:
    #         setattr(self._ct, name, value)

    def decrypt(self, cc, sk, isFormat=True, precision=None):
        result = cc.Decrypt(self._data, sk)
        result.SetLength(self.size)
        if precision is not None:
            result.GetFormattedValues(precision)
        result = result.GetRealPackedValue()
        if isFormat:
            result = utils.format(result, self.ndim, self.original_shape, self.shape)
        return result

    def copy(self):
        return ctarray(
            self._data,
            self.original_shape,
            self.ndim,
            self.size,
            self.ncols,
            self.order,
        )


def gen_crypto_context(ring_dim, mult_depth):
    """
    Generate a CryptoContext and key pair for CKKS encryption.

    Parameters
    ----------
    ring_dim : int
        Ring dimension (must be power of two).
    mult_depth : int
        Maximum multiplicative depth for the ciphertext.

    Returns
    -------
    tuple
        (CryptoContext, KeyPair)
    """
    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(HEStd_NotSet)
    params.SetRingDim(ring_dim)
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    params.SetBatchSize(ring_dim // 2)
    params.SetScalingTechnique(FIXEDAUTO)
    params.SetKeySwitchTechnique(HYBRID)
    params.SetSecretKeyDist(UNIFORM_TERNARY)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def xarray(
    cc,
    data: list,
    total_slots: int,
    batch_size: int = 1,
    encoding_type: int = MatrixOrder.ROW_MAJOR,
    type: str = DataType.CIPHERTEXT,
    public_key=None,
):
    org_rows, org_cols, ndim = get_shape(data)

    if ndim == 2:
        ncols = next_power_of_two(org_cols)
        packed_data = ravel_matrix(data, total_slots, ncols, encoding_type)
    else:
        ncols = batch_size
        packed_data = ravel_vector(data, total_slots, ncols, encoding_type)

    plaintext = cc.MakeCKKSPackedPlaintext(packed_data)

    if type == DataType.PLAINTEXT:
        return PTArray(plaintext, (org_rows, org_cols), ndim, total_slots, ncols, encoding_type)

    if public_key is None:
        raise ValueError("Public key must be provided for ciphertext encoding.")

    ciphertext = cc.Encrypt(public_key, plaintext)
    # print(type(ciphertext))

    print(ciphertext)
    return ctarray(ciphertext, (org_rows, org_cols), ndim, total_slots, ncols, encoding_type)


def demo():
    """
    Run a demonstration of homomorphic matrix multiplication using OpenFHE-NumPy.
    """
    ring_dim = 2**12
    mult_depth = 4
    total_slots = ring_dim // 2

    cc, keys = gen_crypto_context(ring_dim, mult_depth)

    # Sample input matrices (8x8)
    A = np.array(
        [
            [0, 7, 8, 10, 1, 2, 7, 6],
            [0, 1, 1, 9, 7, 5, 1, 7],
            [8, 8, 4, 5, 8, 2, 6, 1],
            [1, 0, 0, 1, 10, 3, 1, 7],
            [7, 8, 2, 5, 3, 2, 10, 9],
            [0, 3, 4, 10, 10, 5, 2, 5],
            [2, 5, 0, 2, 8, 8, 5, 9],
            [5, 1, 10, 6, 2, 8, 6, 3],
        ]
    )

    B = np.array(
        [
            [7, 0, 1, 3, 5, 0, 1, 8],
            [0, 5, 10, 3, 9, 0, 2, 10],
            [10, 8, 9, 8, 4, 9, 8, 8],
            [2, 9, 7, 9, 3, 8, 2, 8],
            [2, 8, 2, 2, 10, 7, 6, 0],
            [8, 7, 3, 0, 3, 10, 6, 5],
            [6, 6, 5, 9, 10, 5, 4, 7],
            [1, 4, 3, 4, 3, 9, 9, 4],
        ]
    )

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # Encrypt both matrices
    ctm_A = xarray(cc, A, total_slots, public_key=keys.publicKey)
    # ctm_B = onp.array(cc, B, total_slots, public_key=keys.publicKey)
    # c1 = ctm_A.GetCryptoContext()
    # if c1 == cc:
    #     print("HAHA")
    print(type(ctm_A))
    print(ctm_A)
    add = cc.EvalAdd(ctm_A, ctm_A)
    print(type(add))

    result = cc.Decrypt(keys.secretKey, add)
    result.SetLength(20)
    result = result.GetRealPackedValue()
    print(result)

    # result.SetLength(self.size)
    # if precision is not None:
    #     result.GetFormattedValues(precision)
    # result = result.GetRealPackedValue()
    # if isFormat:
    #     result = utils.format(result, self.ndim, self.original_shape, self.shape)
    # return result


if __name__ == "__main__":
    demo()
