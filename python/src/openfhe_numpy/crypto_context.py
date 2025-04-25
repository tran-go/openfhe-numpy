from openfhe_matrix import LinTransType
import openfhe_matrix as fp


def gen_sum_row_keys(cc, sk, block_size):
    return cc.EvalSumRowsKeyGen(sk, None, block_size)


def gen_sum_col_keys(cc, sk, block_size):
    return cc.EvalSumColsKeyGen(sk)


def gen_rotation_keys(cc, sk, rotation_indices):
    cc.EvalRotateKeyGen(sk, rotation_indices)


def gen_lintrans_keys(cc, keys, block_size, type, repetitions=0):
    fp.EvalLinTransKeyGen(cc, keys, block_size, type, repetitions)


def gen_square_matrix_product(cc, keys, block_size):
    fp.MulMatRotateKeyGen(cc, keys, block_size)


def gen_transpose_keys(cc, keys, ct_matrix):
    block_size = ct_matrix.ncols
    fp.EvalLinTransKeyGen(cc, keys, block_size, 4)
