#include "enc_matrix.h"
#include <stdexcept>
#include <iostream>

// Logging macros
#define OPENFHE_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define OPENFHE_WARN(msg) std::cout << "[WARNING] " << msg << std::endl
#define OPENFHE_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#ifdef DEBUG
#define OPENFHE_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define OPENFHE_DEBUG(msg)
#endif

// -------------------------------------------------------------
// Test Function
// -------------------------------------------------------------
void test() {
    OPENFHE_INFO("Hello Boca Raton!!!");
}

// -------------------------------------------------------------
// EvalMultMatVec
// Implements matrix-vector multiplication from:
// https://eprint.iacr.org/2018/254
// -------------------------------------------------------------
CT EvalMultMatVec(CC& context, KeyPair keyPair, MatKeys evalKeys, int encodeType,
                  int rowSize, const CT& ciphertextVec, const CT& ciphertextMat) {
    CT ciphertextProduct;
    auto multiplied = context->EvalMult(ciphertextMat, ciphertextVec);

    OPENFHE_DEBUG("EvalMultMatVec: Decrypting intermediate multiplication result.");
    PT plaintextMult;
    context->Decrypt(keyPair.secretKey, multiplied, &plaintextMult);
    plaintextMult->SetLength(rowSize * 4);
    PrintVector(plaintextMult->GetRealPackedValue());

    if (encodeType == EncodeStyle::MM_CRC) {
        OPENFHE_INFO("EvalMultMatVec: Applying MM_CRC encoding style with rowSize = " << rowSize);
        ciphertextProduct = context->EvalSumCols(multiplied, rowSize, *evalKeys);

        PT plaintextTmp;
        context->Decrypt(keyPair.secretKey, ciphertextProduct, &plaintextTmp);
        plaintextTmp->SetLength(rowSize * 4);
        PrintVector(plaintextTmp->GetRealPackedValue());
    } else if (encodeType == EncodeStyle::MM_RCR) {
        OPENFHE_INFO("EvalMultMatVec: Applying MM_RCR encoding style with rowSize = " << rowSize);
        ciphertextProduct = context->EvalSumRows(multiplied, rowSize, *evalKeys);
    } else {
        OPENFHE_WARN("EvalMultMatVec: Unsupported encoding style selected.");
    }

    return ciphertextProduct;
}

// -------------------------------------------------------------
// Linear Transformations (Sigma, Tau, Phi, Psi)
// -------------------------------------------------------------
CT EvalLinTransSigma(CC cc, KeyPair keyPair, const CT ctVec, const int rowSize) {
    int matrixSize = rowSize * rowSize;
    CT ciphertextResult;
    PT zeroPlaintext = cc->MakeCKKSPackedPlaintext(std::vector<double>(matrixSize, 0.0));
    ciphertextResult = cc->Encrypt(keyPair.publicKey, zeroPlaintext);

    std::vector<int> rotationIndices(2 * rowSize);
    for (int k = -rowSize; k < rowSize; ++k) {
        rotationIndices[k + rowSize] = k;
    }
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);

    for (int k = -rowSize; k < rowSize; ++k) {
        auto diag = GenSigmaDiag(rowSize, k);
        auto diagPlaintext = cc->MakeCKKSPackedPlaintext(diag);
        auto rotated = cc->EvalRotate(ctVec, k);
        cc->EvalAddInPlace(ciphertextResult, cc->EvalMult(rotated, diagPlaintext));
    }

    return ciphertextResult;
}

CT EvalLinTransTau(CC cc, KeyPair keyPair, const CT ctVec, const int rowSize) {
    int matrixSize = rowSize * rowSize;
    CT ciphertextResult = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(matrixSize, 0.0)));

    int32_t slotCount = cc->GetEncodingParams()->GetBatchSize();
    for (int k = 0; k < rowSize; ++k) {
        auto diag = GenTauDiag(slotCount, rowSize, k);
        auto diagPlaintext = cc->MakeCKKSPackedPlaintext(diag);
        cc->EvalRotateKeyGen(keyPair.secretKey, {rowSize * k});
        auto rotated = cc->EvalRotate(ctVec, rowSize * k);
        cc->EvalAddInPlace(ciphertextResult, cc->EvalMult(diagPlaintext, rotated));
    }

    return ciphertextResult;
}

CT EvalLinTransPhi(CC cc, KeyPair keyPair, const CT ctVec, const int rowSize, const int shiftIndex) {
    int matrixSize = rowSize * rowSize;
    CT ciphertextResult = cc->Encrypt(keyPair.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(matrixSize, 0.0)));

    for (int i = 0; i < 2; ++i) {
        int rotateIdx = shiftIndex - i * rowSize;
        auto diag = GenPhiDiag(rowSize, shiftIndex, i);
        auto diagPlaintext = cc->MakeCKKSPackedPlaintext(diag);
        cc->EvalRotateKeyGen(keyPair.secretKey, {rotateIdx});
        auto rotated = cc->EvalRotate(ctVec, rotateIdx);
        cc->EvalAddInPlace(ciphertextResult, cc->EvalMult(rotated, diagPlaintext));
    }

    return ciphertextResult;
}

CT EvalLinTransPsi(CC cc, KeyPair keyPair, const CT ctVec, const int rowSize, const int shiftIndex) {
    cc->EvalRotateKeyGen(keyPair.secretKey, {rowSize * shiftIndex});
    return cc->EvalRotate(ctVec, rowSize * shiftIndex);
}

// -------------------------------------------------------------
// EvalMatMulSquare (based on https://eprint.iacr.org/2018/1041)
// -------------------------------------------------------------
CT EvalMatMulSquare(CC cc, KeyPair keyPair, const CT matrixA, const CT matrixB, int32_t rowSize) {
    CT transformedA = EvalLinTransSigma(cc, keyPair, matrixA, rowSize);
    CT transformedB = EvalLinTransTau(cc, keyPair, matrixB, rowSize);
    CT productCiphertext = cc->EvalMult(transformedA, transformedB);

    for (int32_t k = 1; k < rowSize; ++k) {
        auto transformedA_k = EvalLinTransPhi(cc, keyPair, transformedA, rowSize, k);
        auto transformedB_k = EvalLinTransPsi(cc, keyPair, transformedB, rowSize, k);
        productCiphertext = cc->EvalAdd(productCiphertext, cc->EvalMult(transformedA_k, transformedB_k));
    }

    return productCiphertext;
}

// -------------------------------------------------------------
// EvalMatrixTranspose
// -------------------------------------------------------------
CT EvalMatrixTranspose(const CC& cc, const KeyPair& keyPair, const CT& inputCiphertext, int32_t matrixSize) {
    try {
        int64_t totalElements = static_cast<int64_t>(matrixSize) * matrixSize;
        size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

        std::vector<double> zeroVector(totalElements, 0.0);
        PT initialPlaintext = cc->MakeCKKSPackedPlaintext(zeroVector);
        CT resultCiphertext = cc->Encrypt(keyPair.publicKey, initialPlaintext);

        OPENFHE_INFO("EvalMatrixTranspose: Using " << slotCount << " available slots for encoding.");

        for (int32_t diagonalIndex = -matrixSize + 1; diagonalIndex < matrixSize; ++diagonalIndex) {
            int32_t rotationIndex = (matrixSize - 1) * diagonalIndex;

            auto diagonalVector = GenTransposeDiag(slotCount, matrixSize, diagonalIndex);
            OPENFHE_DEBUG("EvalMatrixTranspose: Generated diagonal vector for index " << diagonalIndex);
            auto diagonalPlaintext = cc->MakeCKKSPackedPlaintext(diagonalVector);

            cc->EvalRotateKeyGen(keyPair.secretKey, {rotationIndex});
            auto rotatedCiphertext = cc->EvalRotate(inputCiphertext, rotationIndex);
            Debug(cc, keyPair, rotatedCiphertext, "[DEBUG] EvalMatrixTranspose: Rotated ciphertext");

            auto productCiphertext = cc->EvalMult(rotatedCiphertext, diagonalPlaintext);
            cc->EvalAddInPlace(resultCiphertext, productCiphertext);
            Debug(cc, keyPair, resultCiphertext, "[DEBUG] EvalMatrixTranspose: Accumulated result ciphertext");
        }

        return resultCiphertext;
    } catch (const std::exception& e) {
        OPENFHE_ERROR("EvalMatrixTranspose: Exception encountered - " << e.what());
        throw std::runtime_error("EvalMatrixTranspose: Homomorphic operation failed.");
    }
}
