#include <pybind11/pybind11.h>
#include "enc_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(openfhe_matrix, m) {
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";

    m.def("EvalMultMatVec",     &EvalMultMatVec,     "Product of a matrix and a vector");
    m.def("EvalLinTransSigma", &EvalLinTransSigma, "Compute linear transformation Sigma");
    m.def("EvalLinTransTau",   &EvalLinTransTau,   "Compute linear transformation Tau");
    m.def("EvalLinTransPhi",   &EvalLinTransPhi,   "Compute linear transformation Phi");
    m.def("EvalLinTransPsi",   &EvalLinTransPsi,   "Compute linear transformation Psi");
    m.def("EvalMatMulSquare",  &EvalMatMulSquare,  "Product of two square matrices");
}
