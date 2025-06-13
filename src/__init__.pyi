# openfhe_numpy/__init__.pyi

from .tensor import BaseTensor, FHETensor, PTArray, CTArray, BlockFHETensor, BlockCTArray, array
from .operations.matrix_api import add, multiply, dot, matmul, transpose, power, cumsum, cumreduce, sum
from .operations.crypto_context import (
    sum_row_keys,
    sum_col_keys,
    gen_rotation_keys,
    gen_lintrans_keys,
    gen_transpose_keys,
    gen_square_matmult_key,
    gen_accumulate_rows_key,
    gen_accumulate_cols_key,
)
from ._onp_cpp import (
    LinTransType,
    MatVecEncoding,
    MulDepthAccumulation,
    EvalLinTransKeyGen,
    EvalSquareMatMultRotateKeyGen,
    EvalSumCumRowsKeyGen,
    EvalSumCumColsKeyGen,
    EvalMultMatVec,
    EvalMatMulSquare,
    EvalTranspose,
    EvalSumCumRows,
    EvalSumCumCols,
)
from .utils.utils import is_power_of_two, next_power_of_two, check_equality_matrix, pack_vector_row_wise
from .config import MatrixOrder, DataType, EPSILON, EPSILON_HIGH, UnpackType
from .utils.log import ONP_WARNING, ONP_DEBUG, ONP_ERROR, ONPNotImplementedError

__version__: str
__all__: list[str]
