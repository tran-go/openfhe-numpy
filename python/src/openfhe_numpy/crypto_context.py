def gen_sum_row_keys(cc, sk, block_size):
    return cc.EvalSumRowsKeyGen(sk, None, block_size)


def gen_sum_col_keys(cc, sk, block_size):
    return cc.EvalSumColsKeyGen(sk)


def gen_rotation_keys(cc, sk, rotation_indices):
    cc.EvalRotateKeyGen(sk, rotation_indices)
