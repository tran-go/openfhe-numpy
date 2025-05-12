#include "enc_matrix.h"
#include <iostream>
#include <stdexcept>
#include <format>

namespace openfhe_matrix {

using namespace lbcrypto;
/**
     * @brief Generate rotation indices required for linear transformation based on transformation
     * type.
     *
     * @param numCols   The row size (number of columns) of the matrix.
     * @param type      The linear transformation type (SIGMA, TAU, PHI, PSI, TRANSPOSE).
     * @param numRepeats   Optional offset used by PHI and PSI types.
     * @return std::vector<int32_t> List of rotation indices to be used for EvalRotateKeyGen.
**/

std::vector<int32_t> GenLinTransIndices(int32_t numCols, LinTransType type, int32_t numRepeats = 0) {
    std::vector<int32_t> rotationIndices;

    switch (type) {
        case LinTransType::SIGMA:
            // Generate indices from -numCols to numCols - 1
            for (int32_t k = -numCols; k < (numCols); ++k) {
                rotationIndices.push_back(k);
            }
            break;

        case LinTransType::TAU:
            // Generate indices: 0, numCols, 2*numCols, ..., (numCols-1)*numCols
            for (int32_t k = 0; k < numCols; ++k) {
                rotationIndices.push_back(numCols * k);
            }
            break;

        case LinTransType::PHI:
            // Generate indices: numRepeats, numRepeats - numCols
            for (int32_t k = 0; k < 2; ++k) {
                rotationIndices.push_back(numRepeats - k * numCols);
            }
            break;

        case LinTransType::PSI:
            // Generate a single index based on offset
            rotationIndices.push_back(numCols * numRepeats);
            break;

        case LinTransType::TRANSPOSE:
            // Generate indices for transposing a square matrix via diagonals
            for (int32_t k = -numCols + 1; k < numCols; ++k) {
                rotationIndices.push_back((numCols - 1) * k);
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
     * @param numCols        The row size of the matrix being transformed.
     * @param type           The type of linear transformation.
     * @param numRepeats       Optional numRepeats used by PHI and PSI transformations.
**/

template <typename Element>
void EvalLinTransKeyGen(PrivateKey<Element>& secretKey, int32_t numCols, LinTransType type, int32_t numRepeats) {
    auto rotationIndices = GenLinTransIndices(numCols, type, numRepeats);
    auto cryptoContext   = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, rotationIndices);
}

/**
     * @brief Generates rotation keys for a matrix linear transformation.
     * @param numCols        size of a row
     * @param type           type of linear transformation
     * @param numRepeats
*/

template <typename Element>
void EvalSquareMatMultRotateKeyGen(PrivateKey<Element>& secretKey, int32_t numCols) {
    auto indicesSigma = GenLinTransIndices(numCols, LinTransType::SIGMA);
    auto indicesTau   = GenLinTransIndices(numCols, LinTransType::TAU);

    auto cryptoContext = secretKey->GetCryptoContext();
    cryptoContext->EvalRotateKeyGen(secretKey, indicesSigma);
    cryptoContext->EvalRotateKeyGen(secretKey, indicesTau);

    for (int32_t numRepeats = 1; numRepeats < numCols; ++numRepeats) {
        auto indicesPhi = GenLinTransIndices(numCols, LinTransType::PHI, numRepeats);
        auto indicesPsi = GenLinTransIndices(numCols, LinTransType::PSI, numRepeats);

        cryptoContext->EvalRotateKeyGen(secretKey, indicesPhi);
        cryptoContext->EvalRotateKeyGen(secretKey, indicesPsi);
    }
}

template <typename Element>
void EvalSumCumRowsKeyGen(PrivateKey<Element>& secretKey, int32_t numCols) {
    auto cryptoContext = secretKey->GetCryptoContext();
    // std::vector<int32_t> indices;

    // for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
    //     indices.push_back(-i * numCols);
    // }
    // cryptoContext->EvalRotateKeyGen(secretKey, indices);
    //     std::vector<int32_t> indices;

    // std::vector<int32_t> indices = [-1];
    // for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
    //     indices.push_back(-i * numCols);
    // }

    cryptoContext->EvalRotateKeyGen(secretKey, {-numCols});
}

template <typename Element>
void EvalSumCumColsKeyGen(PrivateKey<Element>& secretKey, int32_t numCols) {
    auto cryptoContext = secretKey->GetCryptoContext();
    // std::vector<int32_t> indices;

    // for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
    //     indices.push_back(-i * numCols);
    // }
    // cryptoContext->EvalRotateKeyGen(secretKey, indices);
    //     std::vector<int32_t> indices;

    // std::vector<int32_t> indices = [-1];
    // for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
    //     indices.push_back(-i * numCols);
    // }

    cryptoContext->EvalRotateKeyGen(secretKey, {-1});
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
     * @param numCols        The number of padded cols in the encoded matrix
     * @param ciphertextVec  The ciphertext encoding the input vector.
     * @param ciphertextMat  The ciphertext encoding the input matrix.
     *
     * @return The ciphertext resulting from the matrix-vector product.
     *
*/

template <typename Element>
Ciphertext<Element> EvalMultMatVec(MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t numCols,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix) {
    Ciphertext<Element> ctProduct;
    auto cryptoContext = ctVector->GetCryptoContext();
    auto multiplied    = cryptoContext->EvalMult(ctMatrix, ctVector);
    if (encodeType == MatVecEncoding::MM_CRC) {
        ctProduct = cryptoContext->EvalSumCols(multiplied, numCols, *evalKeys);
    }
    else if (encodeType == MatVecEncoding::MM_RCR) {
        ctProduct = cryptoContext->EvalSumRows(multiplied, numCols, *evalKeys);
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
     * @param numCols   The number of padded cols in the encoded matrix
     */

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(PrivateKey<Element>& secretKey,
                                      const Ciphertext<Element>& ciphertext,
                                      int32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::SIGMA);
    return EvalLinTransSigma(ciphertext, numCols);
}

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(const Ciphertext<Element>& ciphertext, int32_t numCols) {
    // int32_t d          = numCols * numCols;
    auto cryptoContext = ciphertext->GetCryptoContext();
    bool flag          = true;
    Ciphertext<Element> ctResult;

    for (int k = -numCols; k < numCols; ++k) {
        auto diag       = GenSigmaDiag(numCols, k);  // returns std::vector<double>
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ciphertext, k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
        }
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
     * @param numCols   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransTau(const Ciphertext<Element>& ctVector, int32_t numCols) {
    // int32_t permMatrixSize = numCols * numCols;
    auto cryptoContext = ctVector->GetCryptoContext();
    bool flag          = true;
    Ciphertext<Element> ctResult;

    int32_t slots = cryptoContext->GetEncodingParams()->GetBatchSize();
    for (auto k = 0; k < numCols; ++k) {
        auto diag       = GenTauDiag(slots, numCols, k);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, numCols * k);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
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
                                    int32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::TAU);
    return EvalLinTransTau(ciphertext, numCols);
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
     * @param numCols   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats) {
    // auto permMatrixSize = numCols * numCols;
    auto cryptoContext = ctVector->GetCryptoContext();
    bool flag          = true;
    Ciphertext<Element> ctResult;

    for (auto i = 0; i < 2; ++i) {
        auto rotateIdx  = numRepeats - i * numCols;
        auto diag       = GenPhiDiag(numCols, numRepeats, i);
        auto ptDiagonal = cryptoContext->MakeCKKSPackedPlaintext(diag);
        auto ctRotated  = cryptoContext->EvalRotate(ctVector, rotateIdx);
        auto ctProduct  = cryptoContext->EvalMult(ctRotated, ptDiagonal);
        if (flag) {
            ctResult = ctProduct;
            flag     = false;
        }
        else
            cryptoContext->EvalAddInPlace(ctResult, ctProduct);
    }

    return ctResult;
}
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::PHI, numRepeats);
    return EvalLinTransPhi(ctVector, numCols, numRepeats);
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
     * @param numCols   The number of padded cols in the encoded matrix
     */
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::PSI, numRepeats);
    return EvalLinTransPsi(ctVector, numCols, numRepeats);
}
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats) {
    auto cryptoContext = ctVector->GetCryptoContext();
    return cryptoContext->EvalRotate(ctVector, numCols * numRepeats);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
template <typename Element>
Ciphertext<Element> EvalMatMulSquare(const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t numCols) {
    auto cryptoContext               = matrixA->GetCryptoContext();
    Ciphertext<Element> transformedA = EvalLinTransSigma(matrixA, numCols);
    Ciphertext<Element> transformedB = EvalLinTransTau(matrixB, numCols);
    Ciphertext<Element> ctProduct    = cryptoContext->EvalMult(transformedA, transformedB);

    for (auto k = 1; k < numCols; ++k) {
        auto transformedAk = EvalLinTransPhi(transformedA, numCols, k);
        auto transformedBk = EvalLinTransPsi(transformedB, numCols, k);
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
                                  int32_t numCols) {
    EvalLinTransKeyGen(secretKey, numCols, LinTransType::TRANSPOSE);
    return EvalTranspose(ciphertext, numCols);
}
template <typename Element>
Ciphertext<Element> EvalTranspose(const Ciphertext<Element>& ciphertext, int32_t numCols) {
    try {
        // int32_t totalElements = numCols * numCols;
        auto cryptoContext = ciphertext->GetCryptoContext();
        uint32_t slots     = cryptoContext->GetEncodingParams()->GetBatchSize();
        bool flag          = true;
        Ciphertext<Element> ctResult;

        for (int32_t index = -numCols + 1; index < numCols; ++index) {
            int32_t rotationIndex = (numCols - 1) * index;
            auto diagonalVector   = GenTransposeDiag(slots, numCols, index);
            auto ptDiagonal       = cryptoContext->MakeCKKSPackedPlaintext(diagonalVector);
            auto ctRotated        = cryptoContext->EvalRotate(ciphertext, rotationIndex);
            auto ctProduct        = cryptoContext->EvalMult(ctRotated, ptDiagonal);
            if (flag) {
                ctResult = ctProduct;
                flag     = false;
            }
            else
                cryptoContext->EvalAddInPlace(ctResult, ctProduct);
        }

        return ctResult;
    }
    catch (const std::exception& e) {
        OPENFHE_THROW("EvalTranspose: Homomorphic operation failed. Details: " + std::string(e.what()));
    }
};
// -------------------------------------------------------------
// EvalSumAccumulate
// -------------------------------------------------------------
std::vector<std::complex<double>> GenMaskSumCols(int k, int slots, int numCols) {
    auto n = (int)(slots / numCols);

    std::vector<std::complex<double>> result(slots, 0);

    for (int i = 0; i < n; ++i) {
        result[i * numCols + k] = 1.0;
    }
    return result;
};

std::vector<std::complex<double>> GenMaskSumRows(int k, int slots, int numRows, int numCols) {
    auto blockSize = numCols * numRows;
    auto n         = slots / blockSize;
    std::vector<std::complex<double>> mask(slots, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < numCols; j++) {
            if (i * blockSize + numCols * k + j < slots)
                mask[i * blockSize + numCols * k + j] = 1;
        }
    }
    return mask;
};

template <typename Element>
Ciphertext<Element> EvalSumCumCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<std::complex<double>> mask = GenMaskSumCols(0, subringDim, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numCols); ++i) {
        auto mask          = GenMaskSumCols(i, subringDim, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        auto rotated       = cc->EvalRotate(ctSum, -1);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalAddInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

template <typename Element>
Ciphertext<Element> EvalReduceCumCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams = ciphertext->GetCryptoParameters();
    const auto cc           = ciphertext->GetCryptoContext();

    subringDim = (subringDim == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : subringDim;

    std::vector<std::complex<double>> mask = GenMaskSumCols(0, subringDim, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numCols); ++i) {
        auto mask          = GenMaskSumCols(i, subringDim, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, subringDim);
        auto rotated       = cc->EvalRotate(ctSum, -1);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

template <class Element>
Ciphertext<Element> EvalSumCumRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows,
                                          uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();

    slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;

    std::cout << numCols << " " << numRows << " " << slots;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<std::complex<double>> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalAddInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

template <class Element>
Ciphertext<Element> EvalReduceCumRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows,
                                          uint32_t slots) {
    if (ciphertext->GetEncodingType() != CKKS_PACKED_ENCODING)
        OPENFHE_THROW("Matrix summation of row-vectors is only supported for CKKS packed encoding.");

    const auto cryptoParams   = ciphertext->GetCryptoParameters();
    const auto encodingParams = cryptoParams->GetEncodingParams();
    const auto cc             = ciphertext->GetCryptoContext();

    if (numCols)
        slots = (slots == 0) ? cryptoParams->GetElementParams()->GetCyclotomicOrder() / 4 : slots;
    numRows = (numRows == 0) ? slots / numCols : numRows;

    std::cout << numCols << " " << numRows << " " << slots;

    if (numRows * numCols > slots)
        OPENFHE_THROW("The size of the matrix is bigger than the total slots.");

    std::vector<std::complex<double>> mask = GenMaskSumRows(0, slots, numRows, numCols);

    auto ctSum = ciphertext->Clone();

    for (size_t i = 1; i < static_cast<size_t>(numRows); ++i) {
        mask               = GenMaskSumRows(i, slots, numRows, numCols);
        auto ptmask        = cc->MakeCKKSPackedPlaintext(mask, 1, 0, nullptr, slots);
        auto rotated       = cc->EvalRotate(ctSum, -numCols);
        auto maskedRotated = cc->EvalMult(rotated, ptmask);
        cc->EvalSubInPlace(ctSum, maskedRotated);
    }
    return ctSum;
};

// template

template void EvalLinTransKeyGen(PrivateKey<DCRTPoly>& secretKey,
                                 int32_t numCols,
                                 LinTransType type,
                                 int32_t numRepeats);

template void EvalSquareMatMultRotateKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols);

template void EvalSumCumColsKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols);

template void EvalSumCumRowsKeyGen(PrivateKey<DCRTPoly>& secretKey, int32_t numCols);

template Ciphertext<DCRTPoly> EvalMultMatVec(MatKeys<DCRTPoly> evalKeys,
                                             MatVecEncoding encodeType,
                                             int32_t numCols,
                                             const Ciphertext<DCRTPoly>& ctVector,
                                             const Ciphertext<DCRTPoly>& ctMatrix);

template Ciphertext<DCRTPoly> EvalLinTransSigma(const Ciphertext<DCRTPoly>& ciphertext, int32_t numCols);

template Ciphertext<DCRTPoly> EvalLinTransSigma(PrivateKey<DCRTPoly>& secretKey,
                                                const Ciphertext<DCRTPoly>& ciphertext,
                                                int32_t numCols);

template Ciphertext<DCRTPoly> EvalLinTransTau(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ciphertext,
                                              int32_t numCols);
template Ciphertext<DCRTPoly> EvalLinTransTau(const Ciphertext<DCRTPoly>& ctVector, int32_t numCols);

template Ciphertext<DCRTPoly> EvalLinTransPhi(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t numCols,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPhi(const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t numCols,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPsi(PrivateKey<DCRTPoly>& secretKey,
                                              const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t numCols,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalLinTransPsi(const Ciphertext<DCRTPoly>& ctVector,
                                              int32_t numCols,
                                              int32_t numRepeats);

template Ciphertext<DCRTPoly> EvalMatMulSquare(const Ciphertext<DCRTPoly>& matrixA,
                                               const Ciphertext<DCRTPoly>& matrixB,
                                               int32_t numCols);

template Ciphertext<DCRTPoly> EvalTranspose(PrivateKey<DCRTPoly>& secretKey,
                                            const Ciphertext<DCRTPoly>& ciphertext,
                                            int32_t numCols);
template Ciphertext<DCRTPoly> EvalTranspose(const Ciphertext<DCRTPoly>& ctMatrix, int32_t numCols);

template Ciphertext<DCRTPoly> EvalSumCumRows(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t, uint32_t);

template Ciphertext<DCRTPoly> EvalSumCumCols(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t);

template Ciphertext<DCRTPoly> EvalReduceCumRows(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t, uint32_t);

template Ciphertext<DCRTPoly> EvalReduceCumCols(const Ciphertext<DCRTPoly>&, uint32_t, uint32_t);
}  // namespace openfhe_matrix