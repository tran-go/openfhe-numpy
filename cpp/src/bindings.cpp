#include <pybind11/pybind11.h>
// #include <pybind11/enum.h>
#include "enc_matrix.h"
#include "config.h"
#include "array_metadata.h"

using namespace openfhe_matrix;
using namespace lbcrypto;

namespace py = pybind11;
void bind_enums_and_constants(py::module& m);
void bind_matrix_funcs(py::module& m);
void bind_ciphertext(py::module& m);
void bind_metadata(py::module& m);

PYBIND11_MODULE(openfhe_matrix, m) {
    m.doc() = "Python bindings for OpenFHE-Matrix homomorphic operations";
    bind_enums_and_constants(m);
    bind_matrix_funcs(m);
    bind_ciphertext(m);
    // bind_metadata(m);
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

    // Matrix Vector Multiplication Types
    py::enum_<MatVecEncoding>(m, "MatVecEncoding")
        .value("MM_CRC", MatVecEncoding::MM_CRC)
        .value("MM_RCR", MatVecEncoding::MM_RCR)
        .value("MM_DIAG", MatVecEncoding::MM_DIAG)
        .export_values();
    py::implicitly_convertible<int, MatVecEncoding>();

    py::enum_<ArrayEncodingType>(m, "ArrayEncodingType")
        .value("ROW_MAJOR", ArrayEncodingType::ROW_MAJOR)
        .value("COL_MAJOR", ArrayEncodingType::COL_MAJOR)
        .value("DIAG_MAJOR", ArrayEncodingType::DIAG_MAJOR)
        .export_values();
    py::implicitly_convertible<int, ArrayEncodingType>();
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
          "EvalLinTransSigma with secretKey");

    m.def("EvalLinTransSigma",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransSigma),
          "EvalLinTransSigma");

    // EvalLinTransTau
    m.def("EvalLinTransTau",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t)>(
              &EvalLinTransTau),
          "EvalLinTransTau with SecretKey");

    m.def("EvalLinTransTau",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t)>(&EvalLinTransTau),
          "EvalLinTransTau");

    // EvalLinTransPhi

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPhi),
          "EvalLinTransPhi with SecretKey");

    m.def("EvalLinTransPhi",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPhi),
          "EvalLinTransPhi");

    // EvalLinTransPsi
    m.def("EvalLinTransPsi",
          static_cast<Ciphertext<DCRTPoly> (*)(PrivateKey<DCRTPoly>&, const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(
              &EvalLinTransPsi),
          "EvalLinTransPsi with SecretKey");

    m.def("EvalLinTransPsi",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, int32_t, int32_t)>(&EvalLinTransPsi),
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

    // EvalAddAccumulateRows
    m.def("EvalAddAccumulateRows",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t)>(
              &EvalAddAccumulateRows<DCRTPoly>),
          py::arg("ciphertext"),
          py::arg("rowSize"),
          py::arg("subringDim") = 0,
          "Accumulate rows of a matrix ciphertext.");

    // EvalAddAccumulateCols
    m.def("EvalAddAccumulateCols",
          static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t)>(
              &EvalAddAccumulateCols<DCRTPoly>),
          py::arg("ciphertext"),
          py::arg("rowSize"),
          py::arg("subringDim") = 0,
          "Accumulate cols of a matrix ciphertext.");

              // EvalAddAccumulateRows
    m.def("EvalSubAccumulateRows",
        static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t)>(
            &EvalSubAccumulateRows<DCRTPoly>),
        py::arg("ciphertext"),
        py::arg("rowSize"),
        py::arg("subringDim") = 0,
        "Sub accumulate rows of a matrix ciphertext.");

  // EvalAddAccumulateCols
  m.def("EvalSubAccumulateCols",
        static_cast<Ciphertext<DCRTPoly> (*)(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t)>(
            &EvalSubAccumulateCols<DCRTPoly>),
        py::arg("ciphertext"),
        py::arg("rowSize"),
        py::arg("subringDim") = 0,
        "Sub accumulate cols of a matrix ciphertext.");
}

template <typename Element>
Ciphertext<Element> EvalAddAccumulateRows(ConstCiphertext<Element>& ciphertext,
                                          uint32_t rowSize,
                                          uint32_t subringDim = 0);

void bind_ciphertext(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("Ciphertext");
    auto cls                  = py::reinterpret_borrow<py::class_<Ciphertext<DCRTPoly>>>(pyClsObj);
    cls.def("GetEncodingType", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetEncodingType(); });
    cls.def("GetCryptoContext", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetCryptoContext(); });
}

// FOR LATER
// void bind_metadata(py::module& m) {
//     py::class_<ArrayMetadata, Metadata, std::shared_ptr<ArrayMetadata>>(m, "ArrayMetadata")

//         .def(py::init<std::array<int,2>, int32_t, int32_t, int32_t, ArrayEncodingType>(),
//              py::arg("initialShape"),
//              py::arg("ndim"),
//              py::arg("rowsize"),
//              py::arg("batchSize"),
//              py::arg("encodeType") = ArrayEncodingType::ROW_MAJOR)

//         // properties for all fields
//         // properties, using lambdas to disambiguate
//         .def_property(
//             "initialShape",
//             [](const ArrayMetadata& a) -> std::array<int,2> { return a.initialShape(); },
//             [](ArrayMetadata& a, std::array<int,2> v) { a.initialShape() = v; },
//             "Original (flattened) array length")
//         .def_property(
//             "ndim",
//             [](const ArrayMetadata& a) -> int32_t { return a.ndim(); },
//             [](ArrayMetadata& a, int32_t v) { a.ndim() = v; },
//             "Number of dimensions")
//         .def_property(
//             "rowsize",
//             [](const ArrayMetadata& a) -> int32_t { return a.rowsize(); },
//             [](ArrayMetadata& a, int32_t v) { a.rowsize() = v; },
//             "Number of columns")
//         .def_property(
//             "nrows",
//             [](const ArrayMetadata& a) -> int32_t { return a.nrows(); },
//             [](ArrayMetadata& a, int32_t v) { a.nrows() = v; },
//             "Number of rows")
//         .def_property(
//             "batchSize",
//             [](const ArrayMetadata& a) -> int32_t { return a.batchSize(); },
//             [](ArrayMetadata& a, int32_t v) { a.batchSize() = v; },
//             "Batch size")
//         .def_property(
//             "encodeType",
//             [](const ArrayMetadata& a) -> ArrayEncodingType { return a.encodeType(); },
//             [](ArrayMetadata& a, ArrayEncodingType e) { a.encodeType() = e; },
//             "Row‐ or column‐major encoding")

//         // Metadata interface
//         .def("Clone", &ArrayMetadata::Clone, "Return a deep copy of this ArrayMetadata")
//         .def(
//             "__eq__",
//             [](const ArrayMetadata& a, const Metadata& b) { return a == b; },
//             py::is_operator(),
//             "Compare metadata for equality")
//         .def(
//             "print",
//             [](const ArrayMetadata& a) {
//                 std::ostringstream os;
//                 a.print(os);
//                 return os.str();
//             },
//             "Stringify the metadata")

//         // Serialization
//         .def("SerializedObjectName", &ArrayMetadata::SerializedObjectName)
//         .def_static("SerializedVersion", &ArrayMetadata::SerializedVersion);

//     // 3) If you need the template helpers for DCRTPoly:
//     m.def("GetArrayMetadata_DCRTPoly",
//           &ArrayMetadata::GetMetadata<DCRTPoly>,
//           py::arg("ciphertext"),
//           "Retrieve ArrayMetadata from a Ciphertext<DCRTPoly>");
//     m.def("StoreArrayMetadata_DCRTPoly",
//           &ArrayMetadata::StoreMetadata<DCRTPoly>,
//           py::arg("ciphertext"),
//           py::arg("metadata"),
//           "Attach ArrayMetadata to a Ciphertext<DCRTPoly>");
// }
