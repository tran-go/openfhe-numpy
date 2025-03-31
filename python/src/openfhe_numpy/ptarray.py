from typing import Tuple

# import openfhe related libraries
import openfhe
import openfhe_matrix

# import config and auxilarries files
from openfhe_numpy.config import *
from openfhe_numpy.matlib import *
import openfhe_numpy.utils as utils


class ptarray:
    def __init__(
        self,
        data: openfhe.Plaintext,
        original_shape: Tuple[
            int, int
        ],  # original dimensions. original_shape = (nrows,ncols) before padding
        ndim: int,
        size: int,
        ncols: int = 1,  # block_size
        encoding_order: int = MatrixEncoding.ROW_MAJOR,
    ):
        self.data = data
        self.original_shape = original_shape
        self.ndim = ndim  # plaintext matrix
        self.ncols = ncols  # padded cols
        self.nrows = size // ncols
        self.batch_size = size
        self.encoding_order = encoding_order

    def info(self):
        return [
            None,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        ]

    def copy(self, is_deep_copy: bool = 1):
        return ctarray(
            self.data,
            self.original_shape,
            self.ndim,
            self.batch_size,
            self.ncols,
            self.encoding_order,
        )


# encode_matrix
def ravel_mat(
    cc: CC,
    data: list,
    num_slots: int,
    row_size: int = 1,
    order: int = MatrixEncoding.ROW_MAJOR,
) -> PT:
    """Encode a matrix or data without padding or replicate"""
    print(order)
    if order == MatrixEncoding.ROW_MAJOR:
        packed_data = utils.pack_mat_row_wise(data, row_size, num_slots)
    elif order == MatrixEncoding.COL_MAJOR:
        packed_data = utils.pack_mat_col_wise(data, row_size, num_slots)
    else:
        # TODO Encoded Diagonal Matrix
        packed_data = [0]

    print("DEBUG[encode_matrix] ", packed_data)

    return cc.MakeCKKSPackedPlaintext(packed_data)


def ravel_vec(
    cc: CC, data: list, num_slots: int, row_size: int = 1, order: int = MatrixEncoding.ROW_MAJOR
) -> PT:
    """Encode a vector with n replication"""

    if row_size < 1:
        sys.exit("ERROR: Number of repetitions should be larger than 0")

    if row_size == 1 and order == MatrixEncoding.ROW_MAJOR:
        sys.exit("ERROR: Can't encode a vector row-wise with 0 repetitions")

    if not is_power2(row_size):
        sys.exit("ERROR: The number of repetitions in vector packing should be a power of two")

    if order == MatrixEncoding.ROW_MAJOR:
        packed_data = utils.pack_vec_row_wise(data, row_size, num_slots)
    elif order == MatrixEncoding.COL_MAJOR:
        packed_data = utils.pack_vec_col_wise(data, row_size, num_slots)
    else:
        packed_data = [0]

    # print("DEBUG[encode_vector] ", packed_data)

    return cc.MakeCKKSPackedPlaintext(packed_data)
