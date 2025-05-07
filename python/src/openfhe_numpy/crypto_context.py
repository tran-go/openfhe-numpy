from openfhe import CryptoContext as _originCryptoContext
import openfhe_matrix as fp


class CryptoContext(_originCryptoContext):
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            def unwrap(x):
                if isinstance(x, CTArray):
                    return x.data
                if isinstance(x, PTArray):
                    return x.data
                return x

            def wrap(x):
                if isinstance(x, PTArray):
                    return PTArray(x)
                if isinstance(x, CTArray):
                    return CTArray(x)
                return x

            u_args = [unwrap(a) for a in args]
            u_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

            # 2) Call the real C++ method
            result = attr(*u_args, **u_kwargs)

            # 3) Re-wrap results:
            if isinstance(result, (list, tuple)):
                return type(result)(wrap(x) for x in result)
            return wrapped(x)


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
