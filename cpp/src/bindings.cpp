#include <pybind11/pybind11.h>
#include "enc_matrix.h"
#include "config.h"
using namespace fhemat;
using namespace lbcrypto;

namespace py = pybind11;
void bind_enums_and_constants(py::module& m);
void bind_matrix_funcs(py::module& m);
void bind_ciphertext(py::module& m);

PYBIND11_MODULE(openfhe_matrix, m) {
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";
    bind_enums_and_constants(m);
    bind_matrix_funcs(m);
    bind_ciphertext(m);
}

void bind_enums_and_constants(py::module& m) {
    // Linear Transformation Types
    py::enum_<LinTransType>(m, "LinTransType")
        .value("SIGMA", LinTransType::SIGMA)
        .value("TAU", LinTransType::TAU)
        .value("PHI", LinTransType::PHI)
        .value("PSI", LinTransType::PSI)
        .value("TRANSPOSE", LinTransType::TRANSPOSE)
        .export_values();

    // Matrix Vector Multiplication Types
    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();
}
void bind_matrix_funcs(py::module& m) {
    // EvalLinTransKeyGen
    m.def("EvalLinTransKeyGen",
          static_cast<void (*)(CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, int32_t, int, int32_t)>(
              &EvalLinTransKeyGenFromInt<DCRTPoly>),
          py::arg("cryptoContext"),
          py::arg("keyPair"),
          py::arg("rowSize"),
          py::arg("type"),
          py::arg("nRepeats") = 0);
    // EvalLinTransSigma
    m.def("EvalLinTransSigma",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const PublicKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransSigma),
          "EvalLinTransSigma with PublicKey<DCRTPoly>");

    m.def("EvalLinTransSigma",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransSigma),
          "EvalLinTransSigma with KeyPair<DCRTPoly>");

    // EvalLinTransTau
    m.def("EvalLinTransTau",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransTau),
          "EvalLinTransTau");

    // EvalLinTransPhi
    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const PublicKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPhi),
          "EvalLinTransPhi with PublicKey<DCRTPoly>");

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPhi),
          "EvalLinTransPhi with KeyPair<DCRTPoly>");

    // EvalLinTransPsi
    m.def(
        "EvalLinTransPsi",
        static_cast<Ciphertext<DCRTPoly> (*)(CryptoContext<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
            &EvalLinTransPsi),
        "EvalLinTransPsi (default)");

    m.def("EvalLinTransPsi",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPsi),
          "EvalLinTransPsi with KeyPair<DCRTPoly>");

    // EvalMatMulSquare
    m.def("EvalMatMulSquare",
          static_cast<Ciphertext<DCRTPoly> (*)(CryptoContext<DCRTPoly>&,
                                               const PublicKey<DCRTPoly>&,
                                               const Ciphertext<DCRTPoly>&,
                                               const Ciphertext<DCRTPoly>&,
                                               int32_t)>(&EvalMatMulSquare),
          "EvalMatMulSquare");

    // EvalMatrixTranspose
    m.def("EvalMatrixTranspose",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalMatrixTranspose),
          "EvalMatrixTranspose with KeyPair<DCRTPoly>");
    m.def("EvalMatrixTranspose",
          static_cast<Ciphertext<DCRTPoly> (*)(
              CryptoContext<DCRTPoly>&, const PublicKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalMatrixTranspose),
          "EvalMatrixTranspose with KeyPair<DCRTPoly>");

    // MulMatRotateKeyGen

    m.def("MulMatRotateKeyGen",
          static_cast<void (*)(CryptoContext<DCRTPoly>&, const KeyPair<DCRTPoly>&, int32_t)>(&MulMatRotateKeyGen),
          "MulMatRotateKeyGen with KeyPair<DCRTPoly>");

    // EvalMultMatVec
    m.def("EvalMultMatVec",
          static_cast<Ciphertext<DCRTPoly> (*)(CryptoContext<DCRTPoly>&,
                                               MatKeys<DCRTPoly>,
                                               MatVecEncoding,
                                               int32_t,
                                               const Ciphertext<DCRTPoly>&,
                                               const Ciphertext<DCRTPoly>&)>(&EvalMultMatVec),
          "EvalMultMatVec with MatKeys<DCRTPoly>");
}

void bind_ciphertext(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("Ciphertext");
    auto cls                  = py::reinterpret_borrow<py::class_<Ciphertext<DCRTPoly>>>(pyClsObj);
    cls.def("GetEncodingType", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetEncodingType(); });
}
