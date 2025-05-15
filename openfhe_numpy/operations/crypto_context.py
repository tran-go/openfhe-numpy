# from openfhe import CryptoContext as _originCryptoContext
import openfhe_numpy as onp
from .. import _openfhe_numpy  # Import from parent package


def gen_sum_row_keys(context, private_key, block_size=0):
    print(f"[gen_sum_row_keys] block_size={block_size}")
    return context.EvalSumRowsKeyGen(private_key, None, block_size)


def gen_sum_col_keys(context, sk, ncols=0):
    indices = []
    for i in range(ncols):
        indices.append(i * ncols)
        indices.append(-i * ncols)

    context.EvalRotateKeyGen(sk, indices)


def gen_accumulate_rows_key(secretKey, ncols):
    _openfhe_numpy.EvalSumCumRowsKeyGen(secretKey, ncols)


def gen_accumulate_cols_key(secretKey, ncols):
    _openfhe_numpy.EvalSumCumColsKeyGen(secretKey, ncols)


def gen_rotation_keys(context, sk, rotation_indices):
    context.EvalRotateKeyGen(sk, rotation_indices)


def gen_lintrans_keys(private_key, block_size, lintrans_type, repetitions=0):
    _openfhe_numpy.EvalLinTransKeyGen(private_key, block_size, lintrans_type, repetitions)


def gen_square_matrix_product(private_key, block_size):
    _openfhe_numpy.EvalSquareMatMultRotateKeyGen(private_key, block_size)


def gen_transpose_keys(private_key, ct_matrix):
    _openfhe_numpy.EvalLinTransKeyGen(private_key, ct_matrix.ncols, onp.LinTransType.TRANSPOSE)
