//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#ifndef FHEMAT_ARRAY_METADATA_H
#define FHEMAT_ARRAY_METADATA_H

#include <cstdint>
#include <ostream>
#include <memory>

/* ------------------------------------------------------------------ */
/*  ArrayMetadata declaration                                         */
/* ------------------------------------------------------------------ */
namespace openfhe_matrix {
class ArrayMetadata : public lbcrypto::Metadata {
public:
    /* constructors / destructor */
    constexpr ArrayMetadata() = default;
    ArrayMetadata(int32_t initialShape,
                  int32_t ndim,
                  int32_t ncols,
                  int32_t batchSize,
                  ArrayEncodingType enc = ArrayEncodingType::ROW_MAJOR);
    ~ArrayMetadata() override = default;

    int32_t& initialShape() noexcept {
        return m_initialShape;
    }
    constexpr int32_t initialShape() const noexcept {
        return m_initialShape;
    }

    int32_t& ndim() noexcept {
        return m_ndim;
    }
    constexpr int32_t ndim() const noexcept {
        return m_ndim;
    }

    int32_t& ncols() noexcept {
        return m_ncols;
    }
    constexpr int32_t ncols() const noexcept {
        return m_ncols;
    }

    int32_t& nrows() noexcept {
        return m_nrows;
    }
    constexpr int32_t nrows() const noexcept {
        return m_nrows;
    }

    int32_t& batchSize() noexcept {
        return m_batchSize;
    }
    constexpr int32_t batchSize() const noexcept {
        return m_batchSize;
    }

    ArrayEncodingType& encodeType() noexcept {
        return m_encodeType;
    }
    constexpr ArrayEncodingType encodeType() const noexcept {
        return m_encodeType;
    }

    /* Metadata interface overrides */
    std::shared_ptr<lbcrypto::Metadata> Clone() const;
    bool operator==(const lbcrypto::Metadata& rhs) const;
    std::ostream& print(std::ostream& os) const;

    template <class Archive>
    void save(Archive& ar, std::uint32_t) const;
    template <class Ar>
    void load(Archive& ar, std::uint32_t);

    std::string SerializedObjectName() const;
    static uint32_t SerializedVersion();

    /* helper to attach / get from ciphertext  (must stay in header) */
    template <class Element>
    static std::shared_ptr<ArrayMetadata> GetMetadata(
        const std::shared_ptr<const lbcrypto::CiphertextImpl<Element>>& ct);

    template <class Element>
    static void StoreMetadata(std::shared_ptr<lbcrypto::CiphertextImpl<Element>> ct,
                              std::shared_ptr<ArrayMetadata> meta);

private:
    int32_t m_initialShape{0};
    int32_t m_ndim{0};
    int32_t m_ncols{0};
    int32_t m_nrows{0};
    int32_t m_batchSize{0};
    ArrayEncodingType m_encType{ArrayEncodingType::ROW_MAJOR};
};

/* ---------- template bodies (header‑only) ------------------------ */
template <class Archive>
inline void ArrayMetadata::save(Archive& ar, std::uint32_t) const {
    ar(cereal::base_class<lbcrypto::Metadata>(this), m_initialShape, m_ndim, m_ncols, m_nrows, m_batchSize, m_encType);
}
template <class Archive>
inline void ArrayMetadata::load(Archive& ar, std::uint32_t ver) {
    if (ver > SerializedVersion())
        OPENFHE_THROW("ArrayMetadata: incompatible version");
    ar(cereal::base_class<lbcrypto::Metadata>(this), m_initialShape, m_ndim, m_ncols, m_nrows, m_batchSize, m_encType);
}

/* helper templates */
template <class Element>
inline std::shared_ptr<ArrayMetadata> ArrayMetadata::GetMetadata(
    const std::shared_ptr<const lbcrypto::CiphertextImpl<Element>>& ct) {
    auto it = ct->FindMetadataByKey(METADATA_ARRAYINFO_TAG);
    if (!ct->MetadataFound(it))
        OPENFHE_THROW("ArrayMetadata not set");
    return std::dynamic_pointer_cast<ArrayMetadata>(ct->GetMetadata(it));
}
template <class Element>
inline void ArrayMetadata::StoreMetadata(std::shared_ptr<lbcrypto::CiphertextImpl<Element>> ct,
                                         std::shared_ptr<ArrayMetadata> meta) {
    ct->SetMetadataByKey(METADATA_ARRAYINFO_TAG, std::move(meta));
}

/* inline constexpr getters that are trivial */
inline constexpr int32_t ArrayMetadata::initialShape() const noexcept {
    return m_initialShape;
}
inline constexpr int32_t ArrayMetadata::ndim() const noexcept {
    return m_ndim;
}
inline constexpr int32_t ArrayMetadata::ncols() const noexcept {
    return m_ncols;
}
inline constexpr int32_t ArrayMetadata::nrows() const noexcept {
    return m_nrows;
}
inline constexpr int32_t ArrayMetadata::batchSize() const noexcept {
    return m_batchSize;
}
inline constexpr ArrayEncodingType ArrayMetadata::encType() const noexcept {
    return m_encType;
}
}  // namespace openfhe_matrix
#endif  // FHEMAT_ARRAY_METADATA_H