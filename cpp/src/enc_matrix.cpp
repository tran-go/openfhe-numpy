#include "enc_matrix.h"

#include <iostream>
#include <stdexcept>

#include "log.h"


/**
 * @brief Generate rotation indices required for linear transformation based on transformation type.
 *
 * @param rowSize   The row size of the matrix.
 * @param type      The linear transformation type (SIGMA, TAU, PHI, PSI, TRANSPOSE).
 * @param offset    Optional offset used by PHI and PSI types.
 * @return std::vector<int32_t> List of rotation indices to be used for EvalRotateKeyGen.
 */
std::vector<int32_t> GenLinTransIndices(uint32_t rowSize, LinTransType type, int32_t offset = 0) {
    std::vector<int32_t> rotationIndices;

    switch (type) {
        case SIGMA:
            // Generate indices from -rowSize to rowSize - 1
            for (int32_t k = -static_cast<int32_t>(rowSize); k < static_cast<int32_t>(rowSize); ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case TAU:
            // Generate indices: 0, rowSize, 2*rowSize, ..., (rowSize-1)*rowSize
            for (uint32_t k = 0; k < rowSize; ++k) {
                rotationIndices.push_back(static_cast<int32_t>(rowSize * k));
            }
            break;

        case PHI:
            // Generate indices: offset, offset - rowSize
            for (int i = 0; i < 2; ++i) {
                rotationIndices.push_back(offset - i * static_cast<int32_t>(rowSize));
            }
            break;

        case PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(static_cast<int32_t>(rowSize * offset));
            break;

        case TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            for (int32_t diag = -static_cast<int32_t>(rowSize) + 1;
                 diag < static_cast<int32_t>(rowSize); ++diag) {
                rotationIndices.push_back((rowSize - 1) * diag);
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
 * @param cryptoContext  The OpenFHE CryptoContext to operate on.
 * @param keyPair        The KeyPair containing the secret key used to generate rotation keys.
 * @param rowSize        The row size of the matrix being transformed.
 * @param type           The type of linear transformation.
 * @param offset         Optional offset used by PHI and PSI transformations.
 */
void EvalLinTransKeyGen(CC& cryptoContext,
                        const KeyPair<DCRTPoly>& keyPair,
                        uint32_t rowSize,
                        LinTransType type,
                        int32_t offset = 0) {
    auto rotationIndices = GenLinTransIndices(rowSize, type, offset);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
}

/// @brief Create Rotation key Matrix Product
/// @param rowSize size of a row
/// @param type type of linear transformation
/// @param offset
void MulMatRotateKeyGen(CC& cryptoContext, const KeyPair& keyPair, int32_t rowSize) {
    auto indicesSigma = GenLinTransIndices(rowSize, SIGMA);
    auto indicesTau = GenLinTransIndices(rowSize, TAU);

    for (int32_t offset = 1; offset < rowSize; ++offset) {
        auto indicesPhi = GenLinTransIndices(rowSize, PHI, offset);
        auto indicesPsi = GenLinTransIndices(rowSize, PSI, offset);

        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesPhi);
        cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesPsi);
    }

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesSigma);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indicesTau);
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
 * @throws OPENFHE_ERROR if the encoding style is unsupported.
 */
CT EvalMultMatVec(CC& cryptoContext, MatKeys evalKeys, MatVecEncoding encodeType, int rowSize, const CT& ciphertextVec,
                  const CT& ciphertextMat) {
    CT ciphertextProduct;
    auto multiplied = cryptoContext->EvalMult(ciphertextMat, ciphertextVec);
    if (encodeType == MatVecEncoding::MM_CRC) {
        ciphertextProduct = cryptoContext->EvalSumCols(multiplied, rowSize, *evalKeys);
    } else if (encodeType == MatVecEncoding::MM_RCR) {
        ciphertextProduct = cryptoContext->EvalSumRows(multiplied, rowSize, *evalKeys);
    } else {
        OPENFHE_ERROR("EvalMultMatVec: Unsupported encoding style selected.");
    }

    return ciphertextProduct;
}

// -------------------------------------------------------------
// Linear Transformations (Sigma)
// -------------------------------------------------------------

CT EvalLinTransSigma(CC cryptoContext, const PublicKey& publicKey, const CT& ctVec, int rowSize) {
    int permMatrixSize = rowSize * rowSize;
    PT zeroPlaintext = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    CT ciphertextResult = cryptoContext->Encrypt(publicKey, zeroPlaintext);

    for (int k = -rowSize; k < rowSize; ++k) {
        auto diag = GenSigmaDiag(rowSize, k);  // returns std::vector<double>
        auto diagPlaintext = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVec, k);
        cryptoContext->EvalAddInPlace(ciphertextResult, cryptoContext->EvalMult(rotated, diagPlaintext));
    }

    return ciphertextResult;
}

CT EvalLinTransSigma(CC cryptoContext, const KeyPair& keyPair, const CT& ctVec, int rowSize) {
    auto indices = GenLinTransIndices(cryptoContext, keyPair, rowSize, SIGMA);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, indices);
    return EvalLinTransSigma(cryptoContext, keyPair.publicKey, ctVec, rowSize);
}

// -------------------------------------------------------------
// Linear Transformations (Tau)
// -------------------------------------------------------------
CT EvalLinTransTau(CC cryptoContext, const PublicKey& publicKey, const CT& ctVec, int rowSize) {
    int permMatrixSize = rowSize * rowSize;
    PT zeroPlaintext = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    CT ciphertextResult = cryptoContext->Encrypt(publicKey, zeroPlaintext);

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (int k = 0; k < rowSize; ++k) {
        auto diag = GenTauDiag(slots, rowSize, k);
        auto diagPlaintext = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVec, rowSize * k);
        cryptoContext->EvalAddInPlace(ciphertextResult, cryptoContext->EvalMult(diagPlaintext, rotated));
    }

    return ciphertextResult;
}

CT EvalLinTransTau(CC cryptoContext, const KeyPair& keyPair, const CT& ctVec, int rowSize) {
    auto rotations = GenLinTransIndices(rowSize, TAU);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations);
    return EvalLinTransTau(cryptoContext, keyPair.publicKey, ctVec, rowSize);
}

// -------------------------------------------------------------
// Linear Transformations (Phi)
// -------------------------------------------------------------

CT EvalLinTransPhi(CC cryptoContext, const PublicKey& publicKey, const CT& ctVec, int rowSize,  int shiftIndex) {
    int permMatrixSize = rowSize * rowSize;
    CT ciphertextResult =
        cryptoContext->Encrypt(publicKey, cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0)));

    for (int i = 0; i < 2; ++i) {
        int rotateIdx = shiftIndex - i * rowSize;
        auto diag = GenPhiDiag(rowSize, shiftIndex, i);
        auto diagPlaintext = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVec, rotateIdx);
        cryptoContext->EvalAddInPlace(ciphertextResult, cryptoContext->EvalMult(rotated, diagPlaintext));
    }

    return ciphertextResult;
}

CT EvalLinTransPhi(CC cryptoContext, const KeyPair& keyPair, const CT& ctVec, int rowSize, int shiftIndex) {

    auto rotations1 = GenLinTransIndices(rowSize, PHI, shiftIndex);
    auto rotations2 = GenLinTransIndices(rowSize, PHI, shiftIndex);

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations1);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations2);

    return EvalLinTransTau(cryptoContext, keyPair.publicKey, ctVec, rowSize);
}


// -------------------------------------------------------------
// Linear Transformations (Psi)
// -------------------------------------------------------------

CT EvalLinTransPsi(CC cryptoContext, const CT ctVec, const int rowSize, const int shiftIndex) {
    return cryptoContext->EvalRotate(ctVec, rowSize * shiftIndex);
}

CT EvalLinTransPsi(CC cryptoContext, const KeyPair& keyPair, const CT ctVec, const int rowSize, const int shiftIndex) {
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {rowSize * shiftIndex});
    return EvalLinTransPsi(cryptoContext, ctVec, rowSize, shiftIndex);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
CT EvalMatMulSquare(const CC cryptoContext, const lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey, const CT matrixA,
                    const CT matrixB, int32_t rowSize) {
    // MulMatRotateKeyGen(cryptoContext, keyPair, rowSize);
    CT transformedA = EvalLinTransSigma(cryptoContext, publicKey, matrixA, rowSize);
    CT transformedB = EvalLinTransTau(cryptoContext, publicKey, matrixB, rowSize);
    CT productCiphertext = cryptoContext->EvalMult(transformedA, transformedB);

    for (int32_t k = 1; k < rowSize; ++k) {
        auto transformedA_k = EvalLinTransPhi(cryptoContext, publicKey, transformedA, rowSize, k);
        auto transformedB_k = EvalLinTransPsi(cryptoContext, transformedB, rowSize, k);
        productCiphertext =
            cryptoContext->EvalAdd(productCiphertext, cryptoContext->EvalMult(transformedA_k, transformedB_k));
    }

    return productCiphertext;
}

// -------------------------------------------------------------
// EvalMatrixTranspose
// -------------------------------------------------------------
CT EvalMatrixTranspose(const CC cryptoContext, const lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
                       const CT& inputCiphertext, int32_t matrixSize) {
    try {
        int64_t totalElements = static_cast<int64_t>(matrixSize) * matrixSize;
        size_t slotCount = cryptoContext->GetEncodingParams()->GetBatchSize();

        std::vector<double> zeroVector(totalElements, 0.0);
        PT initialPlaintext = cryptoContext->MakeCKKSPackedPlaintext(zeroVector);
        CT resultCiphertext = cryptoContext->Encrypt(publicKey, initialPlaintext);

        OPENFHE_INFO("EvalMatrixTranspose: Using " << slotCount << " available slots for encoding.");

        for (int32_t diagonalIndex = -matrixSize + 1; diagonalIndex < matrixSize; ++diagonalIndex) {
            int32_t rotationIndex = (matrixSize - 1) * diagonalIndex;

            auto diagonalVector = GenTransposeDiag(slotCount, matrixSize, diagonalIndex);
            OPENFHE_DEBUG("EvalMatrixTranspose: Generated diagonal vector for index " << diagonalIndex);
            auto diagonalPlaintext = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);

            // cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {rotationIndex});
            auto rotatedCiphertext = cryptoContext->EvalRotate(inputCiphertext, rotationIndex);
            // Debug(cryptoContext, publicKey, rotatedCiphertext, "[DEBUG] EvalMatrixTranspose: Rotated ciphertext");

            auto productCiphertext = cryptoContext->EvalMult(rotatedCiphertext, diagonalPlaintext);
            cryptoContext->EvalAddInPlace(resultCiphertext, productCiphertext);
            // Debug(cryptoContext, publicKey, resultCiphertext, "[DEBUG] EvalMatrixTranspose: Accumulated result
            // ciphertext");
        }

        return resultCiphertext;
    } catch (const std::exception& e) {
        OPENFHE_ERROR("EvalMatrixTranspose: Exception encountered - " << e.what());
        throw std::runtime_error("EvalMatrixTranspose: Homomorphic operation failed.");
    }
}
