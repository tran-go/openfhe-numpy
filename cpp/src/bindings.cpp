#include <pybind11/pybind11.h>

#include "enc_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(openfhe_matrix, m) {
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";

    py::enum_<LinTransType>(m, "LinTransType")
        .value("SIGMA", LinTransType::SIGMA)
        .value("TAU", LinTransType::TAU)
        .value("PHI", LinTransType::PHI)
        .value("PSI", LinTransType::PSI)
        .value("TRANSPOSE", LinTransType::TRANSPOSE)
        .export_values();

    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();

    // EvalLinTransKeyGen
    //     m.def("EvalLinTransKeyGen",
    //           static_cast<void (*)(CryptoContext&, const KeyPair&, int32_t, LinTransType,
    //           int32_t)>(&EvalLinTransKeyGen), "EvalLinTransKeyGen with KeyPair");
    m.def("EvalLinTransKeyGen", &EvalLinTransKeyGenFromInt, py::arg("cryptoContext"), py::arg("keyPair"),
          py::arg("rowSize"), py::arg("type"), py::arg("nRepeats") = 0,
          "Generate rotation keys using an integer type (0=SIGMA, 4=TRANSPOSE)");

    // EvalLinTransSigma
    m.def("EvalLinTransSigma",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t)>(&EvalLinTransSigma),
          "EvalLinTransSigma with PublicKey");

    m.def("EvalLinTransSigma",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(&EvalLinTransSigma),
          "EvalLinTransSigma with KeyPair");

    // EvalLinTransTau
    m.def("EvalLinTransTau",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(&EvalLinTransTau),
          "EvalLinTransTau");

    // EvalLinTransPhi
    m.def("EvalLinTransPhi",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t, int32_t)>(
              &EvalLinTransPhi),
          "EvalLinTransPhi with PublicKey");

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t, int32_t)>(
              &EvalLinTransPhi),
          "EvalLinTransPhi with KeyPair");

    // EvalLinTransPsi
    m.def("EvalLinTransPsi",
          static_cast<Ciphertext (*)(CryptoContext&, const Ciphertext&, int32_t, int32_t)>(&EvalLinTransPsi),
          "EvalLinTransPsi (default)");

    m.def("EvalLinTransPsi",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t, int32_t)>(
              &EvalLinTransPsi),
          "EvalLinTransPsi with KeyPair");

    // EvalMatMulSquare
    m.def("EvalMatMulSquare",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, const Ciphertext&, int32_t)>(
              &EvalMatMulSquare),
          "EvalMatMulSquare");

    // EvalMatrixTranspose
    m.def("EvalMatrixTranspose",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(&EvalMatrixTranspose),
          "EvalMatrixTranspose with KeyPair");
    m.def(
        "EvalMatrixTranspose",
        static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t)>(&EvalMatrixTranspose),
        "EvalMatrixTranspose with KeyPair");

    // EvalMultMatVec
    m.def("EvalMultMatVec",
          static_cast<Ciphertext (*)(CryptoContext&, MatKeys, MatVecEncoding, int32_t, const Ciphertext&,
                                     const Ciphertext&)>(&EvalMultMatVec),
          "EvalMultMatVec with MatKeys");
}
