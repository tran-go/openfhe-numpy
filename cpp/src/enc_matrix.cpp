#include "enc_matrix.h"
#include <iostream>
#include <stdexcept>

namespace openfhe_matrix {

using namespace lbcrypto;
/**
     * @brief Generate rotation indices required for linear transformation based on transformation
     * type.
     *
     * @param rowSize   The row size (number of columns) of the matrix.
     * @param type      The linear transformation type (SIGMA, TAU, PHI, PSI, TRANSPOSE).
     * @param numRepeats   Optional offset used by PHI and PSI types.
     * @return std::vector<int32_t> List of rotation indices to be used for EvalRotateKeyGen.
**/

std::vector<int32_t> GenLinTransIndices(int32_t rowSize, LinTransType type, int32_t numRepeats = 0) {
    std::vector<int32_t> rotationIndices;

    switch (type) {
        case LinTransType::SIGMA:
            // Generate indices from -rowSize to rowSize - 1
            for (int32_t k = -rowSize; k < (rowSize); ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case LinTransType::TAU:
            // Generate indices: 0, rowSize, 2*rowSize, ..., (rowSize-1)*rowSize
            for (int32_t k = 0; k < rowSize; ++k) {
                rotationIndices.push_back(rowSize * k);
            }
            break;

        case LinTransType::PHI:
            // Generate indices: numRepeats, numRepeats - rowSize
            for (int32_t k = 0; k < 2; ++k) {
                rotationIndices.push_back(numRepeats - k * rowSize);
            }
            break;

        case LinTransType::PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(rowSize * numRepeats);
            break;

        case LinTransType::TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            for (int32_t k = -rowSize + 1; k < rowSize; ++k) {
                rotationIndices.push_back((rowSize - 1) * k);
            }
            break;

        default:
            OPENFHE_THROW("Linear transformation is undefined");
            break;
    }

    return rotationIndices;
}

/**
     * @brief Generates rotation keys needed for a specific linear transformation type.
     *
     * This function wraps the EvalRotateKeyGen call using the appropriate rotation indices
     * computed via GenLinTransIndices. It ensures the crypto context is properly prepared
     * for applying a matrix-based linear transformation.
     *
     * @param cryptoContext  The OpenFHE CryptoContext<Element> to operate on.
     * @param keyPair        The KeyPair<Element> containing the secret key used to generate
     * rotation keys.
     * @param rowSize        The row size of the matrix being transformed.
     * @param type           The type of linear transformation.
     * @param numRepeats       Optional numRepeats used by PHI and PSI transformations.
**/
// template <typename Element>
// void EvalLinTransKeyGenFromInt(const PrivateKey<Element>& secretKey,
//                                int32_t rowSize,
//                                int typeInt,
//                                int32_t numRepeats) {
//     if (typeInt < 0 || typeInt > 4)
//         throw std::invalid_argument("Invalid LinTransType enum value.");
//     EvalLinTransKeyGen(secretKey, rowSize, static_cast<LinTransType>(typeInt), numRepeats);
// }

template <typename Element>
void EvalLinTransKeyGen(PrivateKey<Element>& secretKey, int32_t rowSize, LinTransType type, int32_t numRepeats) {
    auto rotationIndices = GenLinTransIndices(rowSize, type, numRepeats);
    auto cryptoContext   = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, rotationIndices);
}

/**
     * @brief Generates rotation keys for a matrix linear transformation.
     * @param rowSize        size of a row
     * @param type           type of linear transformation
     * @param numRepeats
*/

template <typename Element>
void EvalSquareMatMultRotateKeyGen(PrivateKey<Element>& secretKey, int32_t rowSize) {
    auto indicesSigma = GenLinTransIndices(rowSize, LinTransType::SIGMA);
    auto indicesTau   = GenLinTransIndices(rowSize, LinTransType::TAU);

    auto cryptoContext = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, indicesSigma);
    cryptoContext->EvalRotateKeyGen(secretKey, indicesTau);

    for (int32_t numRepeats = 1; numRepeats < rowSize; ++numRepeats) {
        auto indicesPhi = GenLinTransIndices(rowSize, LinTransType::PHI, numRepeats);
        auto indicesPsi = GenLinTransIndices(rowSize, LinTransType::PSI, numRepeats);

        cryptoContext->EvalRotateKeyGen(secretKey, indicesPhi);
        cryptoContext->EvalRotateKeyGen(secretKey, indicesPsi);
    }
}

/**
     * @brief Performs encrypted matrix-vector multiplication using the specified
     * encoding style.This function multiplies an encrypted matrix with an encrypted
     * vector using homomorphic multiplication from the paper
     * https://eprint.iacr.org/2018/254
     *
     * @param evalKeys       The evaluation keys used for rotations (row/column
     * summation).
     * @param encodeType     The encoding strategy (e.g., MM_CRC for column-wise,
     * MM_RCR for row-wise).
     * @param rowSize        The number of padded cols in the encoded matrix
     * @param ciphertextVec  The ciphertext encoding the input vector.
     * @param ciphertextMat  The ciphertext encoding the input matrix.
     *
     * @return The ciphertext resulting from the matrix-vector product.
     *
*/
// template <typename Element>
// Ciphertext<Element> EvalMultMatVecFromInt(MatKeys<Element> evalKeys,
//                                           int typeInt,
//                                           int32_t rowSize,
//                                           const Ciphertext<Element>& ctVector,
//                                           const Ciphertext<Element>& ctMatrix) {
//     if (typeInt < 0 || typeInt > 3) {
//         OPENFHE_THROW("Invalid MatVecEncoding enum value.");
//     }

//     return EvalMultMatVec(evalKeys, static_cast<MatVecEncoding>(typeInt), rowSize, ctVector, ctMatrix);
// }
template <typename Element>
Ciphertext<Element> EvalMultMatVec(MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t rowSize,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix) {
    Ciphertext<Element> ctProduct;
    auto cryptoContext = ctVector->GetCryptoContext();
    auto multiplied    = cryptoContext->EvalMult(ctMatrix, ctVector);
    if (encodeType == MatVecEncoding::MM_CRC) {
        ctProduct = cryptoContext->EvalSumCols(multiplied, rowSize, *evalKeys);
    }
    else if (encodeType == MatVecEncoding::MM_RCR) {
        ctProduct = cryptoContext->EvalSumRows(multiplied, rowSize, *evalKeys);
    }
    else {
        OPENFHE_THROW("EvalMultMatVec: Unsupported encoding style selected.");
    }

    return ctProduct;
}

/**
     * @brief Linear Transformation (Sigma) as described in the paper:
     * https://eprint.iacr.org/2018/1041
     *
     * The Sigma transformation corresponds to the permutation:
     *   sigma(A)_{i,j} = A_{i, i + j}
     * Its matrix representation is given by:
     *   U_{d·i + j, l} = 1 if l = d·i + (i + j) mod d, and 0 otherwise.
     * where d is the number of columns of the matrix 0 <= i,j < d and
     * @param rowSize   The number of padded cols in the encoded matrix
     */

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(PrivateKey<Element>& secretKey,
                                      const Ciphertext<Element>& ciphertext,
                                      int32_t rowSize) {
    EvalLinTransKeyGen(secretKey, rowSize, LinTransType::SIGMA);
    return EvalLinTransSigma(ciphertext, rowSize);
}

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(const Ciphertext<Element>& ciphertext, int32_t rowSize) {
    int32_t d          = rowSize * rowSize;
    auto cryptoContext = ciphertext->GetCryptoContext();
    bool flag          = true;
    Ciphertext<Element> ctResult;

    for (int k = -rowSize; k < rowSize; ++k) {
        auto diag       = GenSigmaDiag(rowSize, k);  // returns std::vector<double>
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ciphertext, k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag)
            ctResult = ctProduct;
        else
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
    }

    return ctResult;
}

/**
     * @brief Linear Transformation (Tau) as described in the paper:
     * https://eprint.iacr.org/2018/1041
     *
     * The Tau transformation corresponds to the permutation:
     * tau(A)_{i,j} = A_{i + j, j}
     * Its matrix representation is given by:
     * U_{d·i + j, l} = 1 if l = d.(i + j) mod d + j, and 0 otherwise.
     *
     * @param rowSize   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransTau(const Ciphertext<Element>& ctVector, int32_t rowSize) {
    int32_t permMatrixSize = rowSize * rowSize;
    auto cryptoContext     = ctVector->GetCryptoContext();
    bool flag              = true;
    Ciphertext<Element> ctResult;

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (auto k = 0; k < rowSize; ++k) {
        auto diag       = GenTauDiag(slots, rowSize, k);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, rowSize * k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
        }
        else {
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
        }
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransTau(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ciphertext,
                                    int32_t rowSize) {
    EvalLinTransKeyGen(secretKey, rowSize, LinTransType::TAU);
    return EvalLinTransTau(ciphertext, rowSize);
}

/**
     * @brief Linear Transformation (Phi) as described in the paper:
     * https://eprint.iacr.org/2018/1041
     *
     * The Phi transformation corresponds to the permutation:
     * phi(A)_{i,j} = A_{i, j+1}
     * Its k-th matrix representation is given by:
     * U_{d·i + j, l}^k = 1 if l = d.i + (j + k) mod d, and 0 otherwise.
     *
     * @param rowSize   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(const Ciphertext<Element>& ctVector, int32_t rowSize, int32_t numRepeats) {
    auto permMatrixSize = rowSize * rowSize;
    auto cryptoContext  = ctVector->GetCryptoContext();
    bool flag           = true;
    Ciphertext<Element> ctResult;

    for (auto i = 0; i < 2; ++i) {
        auto rotateIdx  = numRepeats - i * rowSize;
        auto diag       = GenPhiDiag(rowSize, numRepeats, i);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, rotateIdx);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag)
            ctResult = ctProduct;
        else
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, rowSize, LinTransType::PHI, numRepeats);
    return EvalLinTransPhi(ctVector, rowSize, numRepeats);
}

/**
     * @brief Linear Transformation (Psi) as described in the paper:
     * https://eprint.iacr.org/2018/1041
     *
     * The Psi transformation corresponds to the permutation:
     *   psi(A)_{i,j} = A_{i+1, j}
     * Its k-th matrix representation is given by:
     *   U_{d·i + j, l}^k = 1 if l = d.(i + k) + j mod d, and 0 otherwise.
     *
     * @param rowSize   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, rowSize, LinTransType::PSI, numRepeats);
    return EvalLinTransPsi(ctVector, rowSize, numRepeats);
}
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(const Ciphertext<Element>& ctVector, int32_t rowSize, int32_t numRepeats) {
    auto cryptoContext = ctVector->GetCryptoContext();
    return cryptoContext->EvalRotate(ctVector, rowSize * numRepeats);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
template <typename Element>
Ciphertext<Element> EvalMatMulSquare(const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t rowSize) {
    auto cryptoContext               = matrixA->GetCryptoContext();
    Ciphertext<Element> transformedA = EvalLinTransSigma(matrixA, rowSize);
    Ciphertext<Element> transformedB = EvalLinTransTau(matrixB, rowSize);
    Ciphertext<Element> ctProduct    = cryptoContext->EvalMult(transformedA, transformedB);

    for (auto k = 1; k < rowSize; ++k) {
        auto transformedAk = EvalLinTransPhi(transformedA, rowSize, k);
        auto transformedBk = EvalLinTransPsi(transformedB, rowSize, k);
        ctProduct          = cryptoContext->EvalAdd(ctProduct, cryptoContext->EvalMult(transformedAk, transformedBk));
    }

    return ctProduct;
}

// -------------------------------------------------------------
// EvalTranspose
// -------------------------------------------------------------
template <typename Element>
Ciphertext<Element> EvalTranspose(PrivateKey<Element>& secretKey,
                                  const Ciphertext<Element>& ciphertext,
                                  int32_t rowSize) {
    EvalLinTransKeyGen(secretKey, rowSize, LinTransType::TRANSPOSE);
    return EvalTranspose(ciphertext, rowSize);
}
template <typename Element>
Ciphertext<Element> EvalTranspose(const Ciphertext<Element>& ciphertext, int32_t rowSize) {
    try {
        int32_t totalElements = rowSize * rowSize;
        auto cryptoContext    = ciphertext->GetCryptoContext();
        uint32_t slots        = cryptoContext->GetEncodingParams()->GetBatchSize();
        bool flag             = true;
        Ciphertext<Element> ctResult;

        for (int32_t index = -rowSize + 1; index < rowSize; ++index) {
            int32_t rotationIndex = (rowSize - 1) * index;
            auto diagonalVector   = GenTransposeDiag(slots, rowSize, index);
            auto ptDiagonal       = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);
            auto ctRotated        = cryptoContext->EvalRotate(ciphertext, rotationIndex);
            auto ctProduct        = cryptoContext->EvalMult(ctRotated, ptDiagonal);
            if (flag)
                ctResult = ctProduct;
            else
                cryptoContext->EvalAddInPlace(ctResult, ctProduct);
        }

        return ctResult;
    }
    catch (const std::exception& e) {
        OPENFHE_THROW("EvalTranspose: Homomorphic operation failed. Details: " + std::string(e.what()));
    }
}
// -------------------------------------------------------------
// EvalSumAccumulate
// -------------------------------------------------------------

template <typename Element>
Ciphertext<Element> EvalAddAccumulateRows(ConstCiphertext<Element> ciphertext,
                                          uint32_t rowSize,
                                          const std::map<uint32_t, EvalKey<Element>>& evalSumKeys,
                                          uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    auto cc                   = ciphertext->GetCryptoContext();

    if ((encodingParams->GetBatchSize() == 0))
        OPENFHE_THROW(
            "Packed encoding parameters 'batch size' is not set. Please check the EncodingParams passed to the crypto context.");

    uint32_t slots = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() : subringDim;

    if (!IsPowerOfTwo(slots))
        OPENFHE_THROW("Matrix summation accumulation of row-vectors is not supported for arbitrary cyclotomics.");

    uint32_t colSize = slots / (4 * rowSize);

    Ciphertext<Element> newCiphertext(std::make_shared<CiphertextImpl<Element>>(*ciphertext));

    std::vector<Ciphertext<Element>> ciphertextVec(colSize, std::make_shared<CiphertextImpl<Element>>(*ciphertext));

    std::vector<std::complex<double>> mask(slots, 0);  // create a mask vector and set all its elements to zero
    for (size_t j = 0; j < colSize; j++) {
        mask[j] = 1;
    }

    newCiphertext = cc->EvalMult(ciphertext, cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));

    for (size_t i = 1; i < static_cast<size_t>(colSize); ++i) {
        mask = std::vector<std::complex<double>>(slots, 0);  // create a mask vector and set all its elements to zero
        for (size_t j = 0; j < colSize; j++) {
            mask[i * colSize + j] = 1;
        }

        auto ciphertextRotated = cc->EvalRotate(ciphertext, i * colSize);
        auto ciphertextTmp     = cc->EvalRotate(cc->EvalAdd(ciphertext, ciphertextRotated), -i * colSize);
        cc->EvalMultInPlace(ciphertextTmp, cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));
        cc->EvalAddInPlace(newCiphertext, ciphertextTmp);
    }

    return newCiphertext;
}

template <typename Element>
Ciphertext<Element> EvalAddAccumulateCols(ConstCiphertext<Element> ciphertext,
                                          uint32_t rowSize,
                                          const std::map<uint32_t, EvalKey<Element>>& evalSumKeys,
                                          uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    auto cc                   = ciphertext->GetCryptoContext();

    if ((encodingParams->GetBatchSize() == 0))
        OPENFHE_THROW(
            "Packed encoding parameters 'batch size' is not set. Please check the EncodingParams passed to the crypto context.");

    uint32_t slots = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() : subringDim;

    if (!IsPowerOfTwo(slots))
        OPENFHE_THROW("Matrix summation accumulation of row-vectors is not supported for arbitrary cyclotomics.");

    uint32_t colSize = slots / (4 * rowSize);

    Ciphertext<Element> newCiphertext(std::make_shared<CiphertextImpl<Element>>(*ciphertext));

    std::vector<Ciphertext<Element>> ciphertextVec(colSize, std::make_shared<CiphertextImpl<Element>>(*ciphertext));

    std::vector<std::complex<double>> mask(slots, 0);  // create a mask vector and set all its elements to zero
    for (size_t j = 0; j < colSize; j++) {
        mask[j] = 1;
    }

    newCiphertext = cc->EvalMult(ciphertext, cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));

    for (size_t i = 1; i < static_cast<size_t>(colSize); ++i) {
        mask = std::vector<std::complex<double>>(slots, 0);  // create a mask vector and set all its elements to zero
        for (size_t j = 0; j < colSize; j++) {
            mask[i * colSize + j] = 1;
        }

        auto ciphertextRotated = cc->EvalRotate(ciphertext, i * colSize);
        auto ciphertextTmp     = cc->EvalRotate(cc->EvalAdd(ciphertext, ciphertextRotated), -i * colSize);
        cc->EvalMultInPlace(ciphertextTmp, cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));
        cc->EvalAddInPlace(newCiphertext, ciphertextTmp);
    }

    return newCiphertext;
}

// template void EvalLinTransKeyGenFromInt(CryptoContext<DCRTPoly>& cryptoContext,
//                                         const KeyPair<DCRTPoly>& keyPair,
//                                         int32_t rowSize,
//                                         int linTransTypeInt,
//                                         int32_t numRepeats = 0);

template void EvalLinTransKeyGen(PrivateKey<DCRTPoly>& secretKey,
                                 int32_t rowSize,
                                 LinTransType type,
                                 int32_t numRepeats);

template void EvalSquareMatMultRotateKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t rowSize);

template Ciphertext<DCRTPoly> EvalMultMatVec(MatKeys<DCRTPoly> evalKeys,
                                             MatVecEncoding encodeType,
                                             int32_t rowSize,
                                             const Ciphertext<DCRTPoly>& ctVector,
                                             const Ciphertext<DCRTPoly>& ctMatrix);

template Ciphertext<DCRTPoly> EvalLinTransSigma(const Ciphertext<DCRTPoly>& ciphertext, int32_t rowSize);

template Ciphertext<DCRTPoly> EvalLinTransSigma(PrivateKey<DCRTPoly>& secretKey,
                                                const Ciphertext<DCRTPoly>& ciphertext,
                                                int32_t rowSize);

template Ciphertext<DCRTPoly> EvalLinTransTau(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ciphertext,
                                              int32_t rowSize);
template Ciphertext<DCRTPoly> EvalLinTransTau(const Ciphertext<DCRTPoly>& ctVector, int32_t rowSize);

template Ciphertext<DCRTPoly> EvalLinTransPhi(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPhi(const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPsi(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPsi(const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalMatMulSquare(const Ciphertext<DCRTPoly>& matrixA,
                                               const Ciphertext<DCRTPoly>& matrixB,
                                               int32_t rowSize);

template Ciphertext<DCRTPoly> EvalTranspose(PrivateKey<DCRTPoly>& secretKey,
                                            const Ciphertext<DCRTPoly>& ciphertext,
                                            int32_t rowSize);
template Ciphertext<DCRTPoly> EvalTranspose(const Ciphertext<DCRTPoly>& ctMatrix, int32_t rowSize);

// template Ciphertext<DCRTPoly> EvalTranspose(CryptoContext<DCRTPoly>& cryptoContext,
//                                                   const PublicKey<DCRTPoly>& publicKey,
//                                                   const Ciphertext<DCRTPoly>& ctMatrix,
//                                                   int32_t rowSize);

// template Ciphertext<DCRTPoly> EvalAddAccumulateCols(ConstCiphertext<DCRTPoly> ciphertext,
//                                                     uint32_t numRows,
//                                                     const std::map<uint32_t, EvalKey<DCRTPoly>>& evalSumKeys,
//                                                     uint32_t subringDim);

// template Ciphertext<DCRTPoly> EvalAddAccumulateRows(ConstCiphertext<DCRTPoly> ciphertext,
//                                                     uint32_t numRows,
//                                                     const std::map<uint32_t, EvalKey<DCRTPoly>>& evalSumKeys,
//                                                     uint32_t subringDim);

}  // namespace openfhe_matrix