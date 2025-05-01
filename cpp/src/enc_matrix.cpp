#include "enc_matrix.h"

#include <iostream>
#include <stdexcept>

namespace fhemat {

using namespace lbcrypto;
/**
     * @brief Generate rotation indices required for linear transformation based on transformation
     * type.
     *
     * @param rowSize   The row size (number of columns) of the matrix.
     * @param type      The linear transformation type (SIGMA, TAU, PHI, PSI, TRANSPOSE).
     * @param nRepeats   Optional offset used by PHI and PSI types.
     * @return std::vector<int32_t> List of rotation indices to be used for EvalRotateKeyGen.
     */

std::vector<int32_t> GenLinTransIndices(int32_t rowSize, LinTransType type, int32_t nRepeats = 0) {
    std::vector<int32_t> rotationIndices;

    switch (type) {
        case SIGMA:
            // Generate indices from -rowSize to rowSize - 1
            for (int32_t k = -rowSize; k < (rowSize); ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case TAU:
            // Generate indices: 0, rowSize, 2*rowSize, ..., (rowSize-1)*rowSize
            for (int32_t k = 0; k < rowSize; ++k) {
                rotationIndices.push_back(rowSize * k);
            }
            break;

        case PHI:
            // Generate indices: nRepeats, nRepeats - rowSize
            for (auto i = 0; i < 2; ++i) {
                rotationIndices.push_back(nRepeats - i * rowSize);
            }
            break;

        case PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(rowSize * nRepeats);
            break;

        case TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            for (int32_t k = -rowSize + 1; k < rowSize; ++k) {
                rotationIndices.push_back((rowSize - 1) * k);
            }
            break;

        default:
            // No action for undefined types
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
     * @param nRepeats       Optional nRepeats used by PHI and PSI transformations.
     */
template <typename Element>
void EvalLinTransKeyGenFromInt(CryptoContext<Element>& cc,
                               const KeyPair<Element>& keyPair,
                               int32_t rowSize,
                               int typeInt,
                               int32_t repeats) {
    if (typeInt < 0 || typeInt > 4) {
        throw std::invalid_argument("Invalid LinTransType enum value.");
    }
    auto type = static_cast<LinTransType>(typeInt);
    EvalLinTransKeyGen(cc, keyPair, rowSize, type, repeats);
}
template <typename Element>
void EvalLinTransKeyGen(CryptoContext<Element>& cryptoContext,
                        const KeyPair<Element>& keyPair,
                        int32_t rowSize,
                        LinTransType type,
                        int32_t nRepeats) {
    auto rotationIndices = GenLinTransIndices(rowSize, type, nRepeats);

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
}

/**
     * @brief Generates rotation keys for a matrix linear transformation.
     * @param rowSize        size of a row
     * @param type           type of linear transformation
     * @param nRepeats
     */

template <typename Element>
void MulMatRotateKeyGen(CryptoContext<Element>& cryptoContext, const KeyPair<Element>& keyPair, int32_t rowSize) {
    auto indicesSigma = GenLinTransIndices(rowSize, SIGMA);
    auto indicesTau   = GenLinTransIndices(rowSize, TAU);

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesSigma);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesTau);

    for (int32_t nRepeats = 1; nRepeats < rowSize; ++nRepeats) {
        auto indicesPhi = GenLinTransIndices(rowSize, PHI, nRepeats);
        auto indicesPsi = GenLinTransIndices(rowSize, PSI, nRepeats);

        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesPhi);
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesPsi);
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
template <typename Element>
Ciphertext<Element> EvalMultMatVecFromInt(CryptoContext<Element>& cryptoContext,
                                          MatKeys<Element> evalKeys,
                                          int typeInt,
                                          int32_t rowSize,
                                          const Ciphertext<Element>& ctVector,
                                          const Ciphertext<Element>& ctMatrix) {
    if (typeInt < 0 || typeInt > 3) {
        OPENFHE_THROW("Invalid MatVecEncoding enum value.");
    }
    auto type = static_cast<MatVecEncoding>(typeInt);
    return EvalMultMatVec(cryptoContext, evalKeys, type, rowSize, ctVector, ctMatrix);
}
template <typename Element>
Ciphertext<Element> EvalMultMatVec(CryptoContext<Element>& cryptoContext,
                                   MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t rowSize,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix) {
    Ciphertext<Element> ctProduct;
    auto multiplied = cryptoContext->EvalMult(ctMatrix, ctVector);
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
Ciphertext<Element> EvalLinTransSigma(CryptoContext<Element>& cryptoContext,
                                      const PublicKey<Element>& publicKey,
                                      const Ciphertext<Element>& ctVector,
                                      int32_t rowSize) {
    int32_t permMatrixSize       = rowSize * rowSize;
    Plaintext ptZeros            = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    Ciphertext<Element> ctResult = cryptoContext->Encrypt(publicKey, ptZeros);

    for (int k = -rowSize; k < rowSize; ++k) {
        auto diag    = GenSigmaDiag(rowSize, k);  // returns std::vector<double>
        auto ptDiag  = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVector, k);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(rotated, ptDiag));
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransSigma(CryptoContext<Element>& cryptoContext,
                                      const KeyPair<Element>& keyPair,
                                      const Ciphertext<Element>& ctVector,
                                      int32_t rowSize) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, SIGMA);
    return EvalLinTransSigma(cryptoContext, keyPair.publicKey, ctVector, rowSize);
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
Ciphertext<Element> EvalLinTransTau(CryptoContext<Element>& cryptoContext,
                                    const PublicKey<Element>& publicKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize) {
    int32_t permMatrixSize       = rowSize * rowSize;
    Plaintext ptZeros            = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    Ciphertext<Element> ctResult = cryptoContext->Encrypt(publicKey, ptZeros);

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (auto k = 0; k < rowSize; ++k) {
        auto diag    = GenTauDiag(slots, rowSize, k);
        auto ptDiag  = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVector, rowSize * k);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(ptDiag, rotated));
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransTau(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, TAU);
    return EvalLinTransTau(cryptoContext, keyPair.publicKey, ctVector, rowSize);
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
Ciphertext<Element> EvalLinTransPhi(CryptoContext<Element>& cryptoContext,
                                    const PublicKey<Element>& publicKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats) {
    auto permMatrixSize          = rowSize * rowSize;
    auto ptZeros                 = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    Ciphertext<Element> ctResult = cryptoContext->Encrypt(publicKey, ptZeros);

    for (auto i = 0; i < 2; ++i) {
        auto rotateIdx = nRepeats - i * rowSize;
        auto diag      = GenPhiDiag(rowSize, nRepeats, i);
        auto ptDiag    = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated   = cryptoContext->EvalRotate(ctVector, rotateIdx);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(rotated, ptDiag));
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, PHI, nRepeats);
    return EvalLinTransPhi(cryptoContext, keyPair.publicKey, ctVector, rowSize, nRepeats);
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
Ciphertext<Element> EvalLinTransPsi(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, PSI, nRepeats);
    return EvalLinTransPsi(cryptoContext, ctVector, rowSize, nRepeats);
}
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(CryptoContext<Element>& cryptoContext,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats) {
    return cryptoContext->EvalRotate(ctVector, rowSize * nRepeats);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
template <typename Element>
Ciphertext<Element> EvalMatMulSquare(CryptoContext<Element>& cryptoContext,
                                     const PublicKey<Element>& publicKey,
                                     const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t rowSize) {
    Ciphertext<Element> transformedA = EvalLinTransSigma(cryptoContext, publicKey, matrixA, rowSize);
    Ciphertext<Element> transformedB = EvalLinTransTau(cryptoContext, publicKey, matrixB, rowSize);
    Ciphertext<Element> ctProduct    = cryptoContext->EvalMult(transformedA, transformedB);

    for (auto k = 1; k < rowSize; ++k) {
        auto transformedAk = EvalLinTransPhi(cryptoContext, publicKey, transformedA, rowSize, k);
        auto transformedBk = EvalLinTransPsi(cryptoContext, transformedB, rowSize, k);
        ctProduct          = cryptoContext->EvalAdd(ctProduct, cryptoContext->EvalMult(transformedAk, transformedBk));
    }

    return ctProduct;
}

// -------------------------------------------------------------
// EvalMatrixTranspose
// -------------------------------------------------------------
template <typename Element>
Ciphertext<Element> EvalMatrixTranspose(CryptoContext<Element>& cryptoContext,
                                        const KeyPair<Element>& keyPair,
                                        const Ciphertext<Element>& ctMatrix,
                                        int32_t rowSize) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, TRANSPOSE);
    return EvalMatrixTranspose(cryptoContext, keyPair.publicKey, ctMatrix, rowSize);
}
template <typename Element>
Ciphertext<Element> EvalMatrixTranspose(CryptoContext<Element>& cryptoContext,
                                        const PublicKey<Element>& publicKey,
                                        const Ciphertext<Element>& ctMatrix,
                                        int32_t rowSize) {
    try {
        int32_t totalElements = rowSize * rowSize;
        const uint32_t slots  = cryptoContext->GetEncodingParams()->GetBatchSize();

        std::vector<double> zeroVector(totalElements, 0.0);
        Plaintext plaintext          = cryptoContext->MakeCKKSPackedPlaintext(zeroVector);
        Ciphertext<Element> ctResult = cryptoContext->Encrypt(publicKey, plaintext);

        for (int32_t index = -rowSize + 1; index < rowSize; ++index) {
            int32_t rotationIndex = (rowSize - 1) * index;
            auto diagonalVector   = GenTransposeDiag(slots, rowSize, index);
            auto ptDiagonal       = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);
            auto ctRotated        = cryptoContext->EvalRotate(ctMatrix, rotationIndex);
            auto ctProduct        = cryptoContext->EvalMult(ctRotated, ptDiagonal);
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
        }

        return ctResult;
    }
    catch (const std::exception& e) {
        OPENFHE_THROW("EvalMatrixTranspose: Homomorphic operation failed. Details: " + std::string(e.what()));
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

        auto ciphertextRotated = cc->EvalRotate(ciphertext, i*colSize);
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

        auto ciphertextRotated = cc->EvalRotate(ciphertext, i*colSize);
        auto ciphertextTmp     = cc->EvalRotate(cc->EvalAdd(ciphertext, ciphertextRotated), -i * colSize);
        cc->EvalMultInPlace(ciphertextTmp, cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots));
        cc->EvalAddInPlace(newCiphertext, ciphertextTmp);
    }

    return newCiphertext;
}

template void EvalLinTransKeyGenFromInt(CryptoContext<DCRTPoly>& cryptoContext,
                                        const KeyPair<DCRTPoly>& keyPair,
                                        int32_t rowSize,
                                        int linTransTypeInt,
                                        int32_t repeats = 0);

template void EvalLinTransKeyGen(CryptoContext<DCRTPoly>& cryptoContext,
                                 const KeyPair<DCRTPoly>& keyPair,
                                 int32_t rowSize,
                                 LinTransType linTransType,
                                 int32_t nRepeats = 0);

template void MulMatRotateKeyGen(CryptoContext<DCRTPoly>& cryptoContext,
                                 const KeyPair<DCRTPoly>& keyPair,
                                 int32_t rowSize);

template Ciphertext<DCRTPoly> EvalMultMatVec(CryptoContext<DCRTPoly>& cryptoContext,
                                             MatKeys<DCRTPoly> evalKeys,
                                             MatVecEncoding encodeType,
                                             int32_t rowSize,
                                             const Ciphertext<DCRTPoly>& ctVector,
                                             const Ciphertext<DCRTPoly>& ctMatrix);

template Ciphertext<DCRTPoly> EvalLinTransSigma(CryptoContext<DCRTPoly>& cryptoContext,
                                                const PublicKey<DCRTPoly>& publicKey,
                                                const Ciphertext<DCRTPoly>& ctVector,
                                                int32_t rowSize);

template Ciphertext<DCRTPoly> EvalLinTransSigma(CryptoContext<DCRTPoly>& cryptoContext,
                                                const KeyPair<DCRTPoly>& keyPair,
                                                const Ciphertext<DCRTPoly>& ctVector,
                                                int32_t rowSize);

template Ciphertext<DCRTPoly> EvalLinTransTau(CryptoContext<DCRTPoly>& cryptoContext,
                                              const KeyPair<DCRTPoly>& keyPair,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize);
template Ciphertext<DCRTPoly> EvalLinTransPhi(CryptoContext<DCRTPoly>& cryptoContext,
                                              const PublicKey<DCRTPoly>& publicKey,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t nRepeats);
template Ciphertext<DCRTPoly> EvalLinTransPhi(CryptoContext<DCRTPoly>& cryptoContext,
                                              const KeyPair<DCRTPoly>& keyPair,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t nRepeats);
template Ciphertext<DCRTPoly> EvalLinTransPsi(CryptoContext<DCRTPoly>& cryptoContext,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t nRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPsi(CryptoContext<DCRTPoly>& cryptoContext,
                                              const KeyPair<DCRTPoly>& keyPair,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t rowSize,
                                              int32_t nRepeats);

template Ciphertext<DCRTPoly> EvalMatMulSquare(CryptoContext<DCRTPoly>& cryptoContext,
                                               const PublicKey<DCRTPoly>& publicKey,
                                               const Ciphertext<DCRTPoly>& matrixA,
                                               const Ciphertext<DCRTPoly>& matrixB,
                                               int32_t rowSize);

template Ciphertext<DCRTPoly> EvalMatrixTranspose(CryptoContext<DCRTPoly>& cryptoContext,
                                                  const KeyPair<DCRTPoly>& keyPair,
                                                  const Ciphertext<DCRTPoly>& ctMatrix,
                                                  int32_t rowSize);

template Ciphertext<DCRTPoly> EvalMatrixTranspose(CryptoContext<DCRTPoly>& cryptoContext,
                                                  const PublicKey<DCRTPoly>& publicKey,
                                                  const Ciphertext<DCRTPoly>& ctMatrix,
                                                  int32_t rowSize);

// template Ciphertext<DCRTPoly> EvalAddAccumulateCols(ConstCiphertext<DCRTPoly> ciphertext,
//                                                     uint32_t numRows,
//                                                     const std::map<uint32_t, EvalKey<DCRTPoly>>& evalSumKeys,
//                                                     uint32_t subringDim);

// template Ciphertext<DCRTPoly> EvalAddAccumulateRows(ConstCiphertext<DCRTPoly> ciphertext,
//                                                     uint32_t numRows,
//                                                     const std::map<uint32_t, EvalKey<DCRTPoly>>& evalSumKeys,
//                                                     uint32_t subringDim);

}  // namespace fhemat