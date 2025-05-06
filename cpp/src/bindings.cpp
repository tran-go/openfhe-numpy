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
    py::implicitly_convertible<int, MatVecEncoding>();
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
}

void bind_ciphertext(py::module& m) {
    py::object existingModule = py::module_::import("openfhe");
    py::object pyClsObj       = existingModule.attr("Ciphertext");
    auto cls                  = py::reinterpret_borrow<py::class_<Ciphertext<DCRTPoly>>>(pyClsObj);
    cls.def("GetEncodingType", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetEncodingType(); });
    cls.def("GetCryptoContext", [](const Ciphertext<DCRTPoly>& ct) { return ct->GetCryptoContext(); });
}

void bind_metadata(py::module& m) {
    py::class_<ArrayMetadata, Metadata, std::shared_ptr<ArrayMetadata>>(m, "ArrayMetadata")

        .def(py::init<int32_t, int32_t, int32_t, int32_t, ArrayEncodingType>(),
             py::arg("initialShape"),
             py::arg("ndim"),
             py::arg("ncols"),
             py::arg("batchSize"),
             py::arg("encodeType") = ArrayEncodingType::ROW_MAJOR)

        // properties for all fields
        .def_property(
            "initialShape", &ArrayMetadata::initialShape, (int32_t& (ArrayMetadata::*)())&ArrayMetadata::initialShape)
        .def_property("ndim", &ArrayMetadata::ndim, (int32_t& (ArrayMetadata::*)())&ArrayMetadata::ndim)
        .def_property("ncols", &ArrayMetadata::ncols, (int32_t& (ArrayMetadata::*)())&ArrayMetadata::ncols)
        .def_property("nrows", &ArrayMetadata::nrows, (int32_t& (ArrayMetadata::*)())&ArrayMetadata::nrows)
        .def_property("batchSize", &ArrayMetadata::batchSize, (int32_t& (ArrayMetadata::*)())&ArrayMetadata::batchSize)
        .def_property("encodeType",
                      &ArrayMetadata::encodeType,
                      (ArrayEncodingType & (ArrayMetadata::*)()) & ArrayMetadata::encodeType)

        // Metadata interface
        .def("Clone", &ArrayMetadata::Clone)
        .def("__eq__", [](const ArrayMetadata& self, const Metadata& other) { return self == other; })
        .def("print",
             [](const ArrayMetadata& self) {
                 std::ostringstream os;
                 self.print(os);
                 return os.str();
             })

        // serialization
        .def("SerializedObjectName", &ArrayMetadata::SerializedObjectName)
        .def_static("SerializedVersion", &ArrayMetadata::SerializedVersion);

    //
    m.def("GetArrayMetadata",
          &ArrayMetadata::GetMetadata<lbcrypto::DCRTPoly>,
          py::arg("ciphertext"),
          "Retrieve the ArrayMetadata attached to a Ciphertext<DCRTPoly>");

    m.def("StoreArrayMetadata",
          &ArrayMetadata::StoreMetadata<lbcrypto::DCRTPoly>,
          py::arg("ciphertext"),
          py::arg("metadata"),
          "Attach ArrayMetadata to a Ciphertext<DCRTPoly>");
}
}
