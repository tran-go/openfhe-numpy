from openfhe_matrix import LinTransType
import openfhe_matrix as fp


def gen_sum_row_keys(context, sk, block_size):
    return context.EvalSumRowsKeyGen(sk, None, block_size)


def gen_sum_col_keys(context, sk, block_size):
    return context.EvalSumColsKeyGen(sk)


def gen_rotation_keys(context, sk, rotation_indices):
    context.EvalRotateKeyGen(sk, rotation_indices)


def gen_lintrans_keys(context, keys, block_size, type, repetitions=0):
    fp.EvalLinTransKeyGen(context, keys, block_size, type, repetitions)


def gen_square_matrix_product(context, keys, block_size):
    fp.MulMatRotateKeyGen(context, keys, block_size)


def gen_transpose_keys(context, keys, ct_matrix):
    block_size = ct_matrix.ncols
    fp.EvalLinTransKeyGen(context, keys, block_size, 4)
