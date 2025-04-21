#include "enc_matrix.h"

#include <iostream>
#include <stdexcept>

#include "log.h"

/**
 * @brief Generate rotation indices required for linear transformation based on transformation type.
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
 * @param cryptoContext  The OpenFHE CryptoContext to operate on.
 * @param keyPair        The KeyPair containing the secret key used to generate rotation keys.
 * @param rowSize        The row size of the matrix being transformed.
 * @param type           The type of linear transformation.
 * @param nRepeats       Optional nRepeats used by PHI and PSI transformations.
 */
void EvalLinTransKeyGen(CryptoContext& cryptoContext, const KeyPair& keyPair, int32_t rowSize, LinTransType type,
                        int32_t nRepeats = 0) {
    auto rotationIndices = GenLinTransIndices(rowSize, type, nRepeats);

    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
}

/**
 * @brief Generates rotation keys for a matrix linear transformation.
 * @param rowSize        size of a row
 * @param type           type of linear transformation
 * @param nRepeats
 */

void MulMatRotateKeyGen(CryptoContext& cryptoContext, const KeyPair& keyPair, int32_t rowSize) {
    auto indicesSigma = GenLinTransIndices(rowSize, SIGMA);
    auto indicesTau = GenLinTransIndices(rowSize, TAU);

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
 * @throws OPENFHE_ERROR if the encoding style is unsupported.
 */
Ciphertext EvalMultMatVec(CryptoContext& cryptoContext, MatKeys evalKeys, MatVecEncoding encodeType, int32_t rowSize,
                          const Ciphertext& ctVector, const Ciphertext& ctMatrix) {
    Ciphertext ctProduct;
    auto multiplied = cryptoContext->EvalMult(ctMatrix, ctVector);
    if (encodeType == MatVecEncoding::MM_CRC) {
        ctProduct = cryptoContext->EvalSumCols(multiplied, rowSize, *evalKeys);
    } else if (encodeType == MatVecEncoding::MM_RCR) {
        ctProduct = cryptoContext->EvalSumRows(multiplied, rowSize, *evalKeys);
    } else {
        ERROR("EvalMultMatVec: Unsupported encoding style selected.");
    }

    return ctProduct;
}

/**
 * @brief Linear Transformation (Sigma) as described in the paper: https://eprint.iacr.org/2018/1041
 *
 * The Sigma transformation corresponds to the permutation:
 *   sigma(A)_{i,j} = A_{i, i + j}
 * Its matrix representation is given by:
 *   U_{d·i + j, l} = 1 if l = d·i + (i + j) mod d, and 0 otherwise.
 * where d is the number of columns of the matrix 0 <= i,j < d and
 * @param rowSize   The number of padded cols in the encoded matrix
 */

Ciphertext EvalLinTransSigma(CryptoContext cryptoContext, const PublicKey& publicKey, const Ciphertext& ctVector,
                             int32_t rowSize) {
    int32_t permMatrixSize = rowSize * rowSize;

    Plaintext ptZeros = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    Ciphertext ctResult = cryptoContext->Encrypt(publicKey, ptZeros);

    for (int k = -rowSize; k < rowSize; ++k) {
        auto diag = GenSigmaDiag(rowSize, k);  // returns std::vector<double>
        auto ptDiag = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVector, k);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(rotated, ptDiag));
    }

    return ctResult;
}

Ciphertext EvalLinTransSigma(CryptoContext cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                             int32_t rowSize) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, SIGMA);
    return EvalLinTransSigma(cryptoContext, keyPair.publicKey, ctVector, rowSize);
}

/**
 * @brief Linear Transformation (Tau) as described in the paper: https://eprint.iacr.org/2018/1041
 *
 * The Tau transformation corresponds to the permutation:
 *   tau(A)_{i,j} = A_{i + j, j}
 * Its matrix representation is given by:
 *   U_{d·i + j, l} = 1 if l = d.(i + j) mod d + j, and 0 otherwise.
 *
 * @param rowSize   The number of padded cols in the encoded matrix
 */
Ciphertext EvalLinTransTau(CryptoContext cryptoContext, const PublicKey& publicKey, const Ciphertext& ctVector,
                           int32_t rowSize) {
    int32_t permMatrixSize = rowSize * rowSize;
    Plaintext ptZeros = cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0));
    Ciphertext ctResult = cryptoContext->Encrypt(publicKey, ptZeros);

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (auto k = 0; k < rowSize; ++k) {
        auto diag = GenTauDiag(slots, rowSize, k);
        auto ptDiag = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVector, rowSize * k);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(ptDiag, rotated));
    }

    return ctResult;
}

Ciphertext EvalLinTransTau(CryptoContext cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize) {
    auto rotations = GenLinTransIndices(rowSize, TAU);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations);
    return EvalLinTransTau(cryptoContext, keyPair.publicKey, ctVector, rowSize);
}

/**
 * @brief Linear Transformation (Phi) as described in the paper: https://eprint.iacr.org/2018/1041
 *
 * The Phi transformation corresponds to the permutation:
 *   phi(A)_{i,j} = A_{i, j+1}
 * Its k-th matrix representation is given by:
 *   U_{d·i + j, l}^k = 1 if l = d.i + (j + k) mod d, and 0 otherwise.
 *
 * @param rowSize   The number of padded cols in the encoded matrix
 */
Ciphertext EvalLinTransPhi(CryptoContext cryptoContext, const PublicKey& publicKey, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats) {
    auto permMatrixSize = rowSize * rowSize;
    Ciphertext ctResult = cryptoContext->Encrypt(
        publicKey, cryptoContext->MakeCKKSPackedPlaintext(std::vector<double>(permMatrixSize, 0.0)));

    for (auto i = 0; i < 2; ++i) {
        auto rotateIdx = nRepeats - i * rowSize;
        auto diag = GenPhiDiag(rowSize, nRepeats, i);
        auto ptDiag = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto rotated = cryptoContext->EvalRotate(ctVector, rotateIdx);
        cryptoContext->EvalAddInPlace(ctResult, cryptoContext->EvalMult(rotated, ptDiag));
    }

    return ctResult;
}

Ciphertext EvalLinTransPhi(CryptoContext cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, PHI, nRepeats);
    return EvalLinTransPhi(cryptoContext, keyPair.publicKey, ctVector, rowSize, nRepeats);
}

/**
 * @brief Linear Transformation (Psi) as described in the paper: https://eprint.iacr.org/2018/1041
 *
 * The Psi transformation corresponds to the permutation:
 *   psi(A)_{i,j} = A_{i+1, j}
 * Its k-th matrix representation is given by:
 *   U_{d·i + j, l}^k = 1 if l = d.(i + k) + j mod d, and 0 otherwise.
 *
 * @param rowSize   The number of padded cols in the encoded matrix
 */
Ciphertext EvalLinTransPsi(CryptoContext cryptoContext, const Ciphertext& ctVector, int32_t rowSize, int32_t nRepeats) {
    return cryptoContext->EvalRotate(ctVector, rowSize * nRepeats);
}

Ciphertext EvalLinTransPsi(CryptoContext cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats) {
    EvalLinTransKeyGen(cryptoContext, keyPair, rowSize, PSI, nRepeats);
    return EvalLinTransPsi(cryptoContext, ctVector, rowSize, nRepeats);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
Ciphertext EvalMatMulSquare(const CryptoContext cryptoContext, const PublicKey& publicKey, const Ciphertext& matrixA,
                            const Ciphertext& matrixB, int32_t rowSize) {
    Ciphertext transformedA = EvalLinTransSigma(cryptoContext, publicKey, matrixA, rowSize);
    Ciphertext transformedB = EvalLinTransTau(cryptoContext, publicKey, matrixB, rowSize);
    Ciphertext ctProduct = cryptoContext->EvalMult(transformedA, transformedB);

    for (auto k = 1; k < rowSize; ++k) {
        auto transformedAk = EvalLinTransPhi(cryptoContext, publicKey, transformedA, rowSize, k);
        auto transformedBk = EvalLinTransPsi(cryptoContext, transformedB, rowSize, k);
        ctProduct = cryptoContext->EvalAdd(ctProduct, cryptoContext->EvalMult(transformedAk, transformedBk));
    }

    return ctProduct;
}

// -------------------------------------------------------------
// EvalMatrixTranspose
// -------------------------------------------------------------
Ciphertext EvalMatrixTranspose(const CryptoContext cryptoContext, const KeyPair keyPair, const Ciphertext& ctMatrix,
                               int32_t rowSize) {
    auto rotations = GenLinTransIndices(rowSize, TRANSPOSE);
    cryptoContext->EvalRotateKeyGen(keyPair.secretKey, rotations);
    return EvalMatrixTranspose(cryptoContext, keyPair.publicKey, ctMatrix, rowSize);
}
Ciphertext EvalMatrixTranspose(const CryptoContext cryptoContext, const PublicKey publicKey, const Ciphertext& ctMatrix,
                               int32_t rowSize) {
    try {
        int32_t totalElements = rowSize * rowSize;
        size_t slotCount = cryptoContext->GetEncodingParams()->GetBatchSize();

        std::vector<double> zeroVector(totalElements, 0.0);
        Plaintext plaintext = cryptoContext->MakeCKKSPackedPlaintext(zeroVector);
        Ciphertext ctResult = cryptoContext->Encrypt(publicKey, plaintext);

        DEBUG("EvalMatrixTranspose: Using " << slotCount << " available slots for encoding.");

        for (int32_t index = -rowSize + 1; index < rowSize; ++index) {
            DEBUG("EvalMatrixTranspose: Generated diagonal vector for index " << index);
            int32_t rotationIndex = (rowSize - 1) * index;
            auto diagonalVector = GenTransposeDiag(slotCount, rowSize, index);
            auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);

            // cryptoContext->EvalRotateKeyGen(keyPair.secretKey, {rotationIndex});
            auto ctRotated = cryptoContext->EvalRotate(ctMatrix, rotationIndex);
            // Debug(cryptoContext, publicKey, rotatedCiphertext, "[DEBUG] EvalMatrixTranspose: Rotated ciphertext");

            auto ctProduct = cryptoContext->EvalMult(ctRotated, ptDiagonal);
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
            // Debug(cryptoContext, publicKey, resultCiphertext, "[DEBUG] EvalMatrixTranspose: Accumulated result
            // ciphertext");
        }

        return ctResult;
    } catch (const std::exception& e) {
        ERROR("EvalMatrixTranspose: Exception encountered - " << e.what());
        throw std::runtime_error("EvalMatrixTranspose: Homomorphic operation failed.");
    }
}
