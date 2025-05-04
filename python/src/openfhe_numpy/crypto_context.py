from openfhe_matrix import LinTransType
import openfhe_matrix as fp


def gen_sum_row_keys(context, private_key, block_size):
    return context.EvalSumRowsKeyGen(private_key, None, block_size)


def gen_sum_col_keys(context, private_key, block_size):
    return context.EvalSumColsKeyGen(private_key)


def gen_rotation_keys(context, sk, rotation_indices):
    context.EvalRotateKeyGen(sk, rotation_indices)


def gen_lintrans_keys(private_key, block_size, lintrans_type, repetitions=0):
    fp.EvalLinTransKeyGen(private_key, block_size, lintrans_type, repetitions)


def gen_square_matrix_product(private_key, block_size):
    fp.EvalSquareMatMultRotateKeyGen(private_key, block_size)


def gen_transpose_keys(private_key, ct_matrix):
    fp.EvalLinTransKeyGen(private_key, ct_matrix.ncols, 4)
