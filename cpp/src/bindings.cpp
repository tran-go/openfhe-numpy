#include <pybind11/pybind11.h>

#include "enc_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(openfhe_matrix, m) {
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";

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
}
