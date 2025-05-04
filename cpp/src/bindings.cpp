#include <pybind11/pybind11.h>
// #include <pybind11/enum.h>
#include "enc_matrix.h"
#include "config.h"

using namespace openfhe_matrix;
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

    py::implicitly_convertible<int, LinTransType>();
    //     m.attr("SIGMA")     = py::cast(LinTransType::SIGMA);
    //     m.attr("TAU")       = py::cast(LinTransType::TAU);
    //     m.attr("PHI")       = py::cast(LinTransType::PHI);
    //     m.attr("PSI")       = py::cast(LinTransType::PSI);
    //     m.attr("TRANSPOSE") = py::cast(LinTransType::TRANSPOSE);

    // Matrix Vector Multiplication Types
    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();
    py::implicitly_convertible<int, MatVecEncoding>();
    //     m.attr("MM_CRC")  = py::cast(MatVecEncoding::MM_CRC);
    //     m.attr("MM_RCR")  = py::cast(MatVecEncoding::MM_RCR);
    //     m.attr("MM_DIAG") = py::cast(MatVecEncoding::MM_DIAG);
}

void bind_matrix_funcs(py::module& m) {
    // EvalLinTransKeyGen
    m.def("EvalLinTransKeyGen",
          static_cast<void (*)(PrivateKey<DCRTPoly>&, int32_t, LinTransType, int32_t)>(&EvalLinTransKeyGen<DCRTPoly>),
          py::arg("secretKey"),
          py::arg("rowSize"),
          py::arg("type"),
          py::arg("numRepeats") = 0);

    // EvalLinTransSigma
    m.def("EvalLinTransSigma",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransSigma),
        //   py::arg("secretKey"),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
          "EvalLinTransSigma with secretKey");

    m.def("EvalLinTransSigma",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransSigma),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
          "EvalLinTransSigma");

    // EvalLinTransTau
    m.def("EvalLinTransTau",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransTau),
        //   py::arg("secretKey"),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
          "EvalLinTransTau with SecretKey");

    m.def("EvalLinTransTau",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransTau),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
          "EvalLinTransTau");

    // EvalLinTransPhi

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPhi),
        //   py::arg("secretKey"),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
        //   py::arg("numRepeats"),
          "EvalLinTransPhi with SecretKey");

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPhi),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
        //   py::arg("numRepeats"),
          "EvalLinTransPhi");

    // EvalLinTransPsi
    m.def("EvalLinTransPsi",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPsi),
        //   py::arg("secretKey"),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
        //   py::arg("numRepeats"),
          "EvalLinTransPsi with SecretKey");

    m.def("EvalLinTransPsi",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPsi),
        //   py::arg("ciphertext"),
        //   py::arg("rowSize"),
        //   py::arg("numRepeats"),
          "EvalLinTransPsi");

    // EvalMatMulSquare
    m.def("EvalMatMulSquare",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalMatMulSquare),
          "EvalMatMulSquare");

    // EvalTranspose
    m.def("EvalTranspose",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalTranspose),
          "EvalTranspose with KeyPair<DCRTPoly>");

    m.def("EvalTranspose",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalTranspose),
          "EvalTranspose with KeyPair<DCRTPoly>");

    // EvalSquareMatMultRotateKeyGen

    m.def("EvalSquareMatMultRotateKeyGen",
          static_cast<void (*)(PrivateKey<DCRTPoly>&, int32_t)>(&EvalSquareMatMultRotateKeyGen),
          "EvalSquareMatMultRotateKeyGen with KeyPair<DCRTPoly>");

    // EvalMultMatVec
    m.def("EvalMultMatVec",
          static_cast<Ciphertext<DCRTPoly> (*)(
              MatKeys<DCRTPoly>, MatVecEncoding, int32_t, const Ciphertext<DCRTPoly>&, const Ciphertext<DCRTPoly>&)>(
              &EvalMultMatVec),
          "EvalMultMatVec with MatKeys<DCRTPoly>");
}

void bind_ciphertext(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("Ciphertext");
    auto cls                  = py::reinterpret_borrow<py::class_<Ciphertext<DCRTPoly>>>(pyClsObj);
    cls.def("GetEncodingType", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetEncodingType(); });
}
