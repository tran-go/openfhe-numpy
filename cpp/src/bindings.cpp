#include <pybind11/pybind11.h>

#include "enc_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(openfhe_matrix, m)
{
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";
    bind_enums_and_constants(m);
    bind_matrix_funcs(m);
    bind_ciphertext(m);
}

void bind_enums_and_constants(py::module& m)
{
    // Linear Transformation Types
    py::enum_<LinTransType>(m, "LinTransType")
        .value("SIGMA", LinTransType::SIGMA)
        .value("TAU", LinTransType::TAU)
        .value("PHI", LinTransType::PHI)
        .value("PSI", LinTransType::PSI)
        .value("TRANSPOSE", LinTransType::TRANSPOSE)
        .export_values();
    //     m.attr("SIGMA") = py::cast("SIGMA", LinTransType::SIGMA);
    //     m.attr("TAU") = py::cast("TAU", LinTransType::TAU);
    //     m.attr("PHI") = py::cast("PHI", LinTransType::PHI);
    //     m.attr("PSI") = py::cast("PSI", LinTransType::PSI);
    //     m.attr("TRANSPOSE") = py::cast("TRANSPOSE", LinTransType::TRANSPOSE);

    // Matrix Vector Multiplication Types
    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();
    // m.attr("MM_CRC") = py::cast("MM_CRC", MatVecEncoding::MM_CRC);
    // m.attr("MM_RCR") = py::cast("MM_RCR", MatVecEncoding::MM_RCR);
    // m.attr("MM_DIAG") = py::cast("MM_DIAG", MatVecEncoding::MM_DIAG);
}
void bind_matrix_funcs(py::module& m)
{
    // EvalLinTransKeyGen
    m.def("EvalLinTransKeyGen", &EvalLinTransKeyGenFromInt, py::arg("cryptoContext"),
          py::arg("keyPair"), py::arg("rowSize"), py::arg("type"), py::arg("nRepeats") = 0,
          "Generate rotation keys using an integer type (0=SIGMA, 4=TRANSPOSE)");

    // EvalLinTransSigma
    m.def("EvalLinTransSigma",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t)>(
              &EvalLinTransSigma),
          "EvalLinTransSigma with PublicKey");

    m.def("EvalLinTransSigma",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(
              &EvalLinTransSigma),
          "EvalLinTransSigma with KeyPair");

    // EvalLinTransTau
    m.def("EvalLinTransTau",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(
              &EvalLinTransTau),
          "EvalLinTransTau");

    // EvalLinTransPhi
    m.def("EvalLinTransPhi",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t,
                                     int32_t)>(&EvalLinTransPhi),
          "EvalLinTransPhi with PublicKey");

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t,
                                     int32_t)>(&EvalLinTransPhi),
          "EvalLinTransPhi with KeyPair");

    // EvalLinTransPsi
    m.def("EvalLinTransPsi",
          static_cast<Ciphertext (*)(CryptoContext&, const Ciphertext&, int32_t, int32_t)>(
              &EvalLinTransPsi),
          "EvalLinTransPsi (default)");

    m.def("EvalLinTransPsi",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t,
                                     int32_t)>(&EvalLinTransPsi),
          "EvalLinTransPsi with KeyPair");

    // EvalMatMulSquare
    m.def("EvalMatMulSquare",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&,
                                     const Ciphertext&, int32_t)>(&EvalMatMulSquare),
          "EvalMatMulSquare");

    // EvalMatrixTranspose
    m.def("EvalMatrixTranspose",
          static_cast<Ciphertext (*)(CryptoContext&, const KeyPair&, const Ciphertext&, int32_t)>(
              &EvalMatrixTranspose),
          "EvalMatrixTranspose with KeyPair");
    m.def("EvalMatrixTranspose",
          static_cast<Ciphertext (*)(CryptoContext&, const PublicKey&, const Ciphertext&, int32_t)>(
              &EvalMatrixTranspose),
          "EvalMatrixTranspose with KeyPair");

    // MulMatRotateKeyGen

    m.def("MulMatRotateKeyGen",
          static_cast<void (*)(CryptoContext&, const KeyPair&, int32_t)>(&MulMatRotateKeyGen),
          "MulMatRotateKeyGen with KeyPair");

    // EvalMultMatVec
    m.def("EvalMultMatVec",
          static_cast<Ciphertext (*)(CryptoContext&, MatKeys, MatVecEncoding, int32_t,
                                     const Ciphertext&, const Ciphertext&)>(&EvalMultMatVec),
          "EvalMultMatVec with MatKeys");
}

void bind_ciphertext(py::module& m)
{
    py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext")
        .def("GetEncodingType", &CiphertextImpl<DCRTPoly>::GetEncodingType);
}

// void bind_ciphertext(py::module &m) {
//     py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m,
//     "Ciphertext")
//         .def(py::init<>())
//         .def(py::init<const A &>())
//         .def(
//             "__add__", [](const Ciphertext<DCRTPoly> &a, const Ciphertext<DCRTPoly> &b) { return
//             a + b; }, py::is_operator(), pybind11::keep_alive<0, 1>())

//         // .def(py::self + py::self);
//         // .def("GetDepth", &CiphertextImpl<DCRTPoly>::GetDepth)
//         // .def("SetDepth", &CiphertextImpl<DCRTPoly>::SetDepth)
//         .def("GetLevel", &CiphertextImpl<DCRTPoly>::GetLevel, ctx_GetLevel_docs)
//         .def("SetLevel", &CiphertextImpl<DCRTPoly>::SetLevel, ctx_SetLevel_docs,
//         py::arg("level")) .def("Clone", &CiphertextImpl<DCRTPoly>::Clone) .def("RemoveElement",
//         &RemoveElementWrapper, cc_RemoveElement_docs)
//         // .def("GetHopLevel", &CiphertextImpl<DCRTPoly>::GetHopLevel)
//         // .def("SetHopLevel", &CiphertextImpl<DCRTPoly>::SetHopLevel)
//         // .def("GetScalingFactor", &CiphertextImpl<DCRTPoly>::GetScalingFactor)
//         // .def("SetScalingFactor", &CiphertextImpl<DCRTPoly>::SetScalingFactor)
//         .def("GetSlots", &CiphertextImpl<DCRTPoly>::GetSlots)
//         .def("SetSlots", &CiphertextImpl<DCRTPoly>::SetSlots)
//         .def("GetNoiseScaleDeg", &CiphertextImpl<DCRTPoly>::GetNoiseScaleDeg)
//         .def("SetNoiseScaleDeg", &CiphertextImpl<DCRTPoly>::SetNoiseScaleDeg);
// }