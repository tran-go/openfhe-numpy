#include "enc_matrix.h"
#include "openfhe.h"
#include "utils.h"

// ============================================================
// Demo: Homomorphic Matrix-Vector Multiplication
// ============================================================

void RunMatrixVectorDemo(bool verbose = true) {
    int32_t numRows = 2;
    int32_t numCols = 4;
    int32_t paddedRowSize = NextPow2(numCols);

    // Input plaintext data
    std::vector<std::vector<double>> inputMatrix = {
        {1, 1, 1, 0},
        {2, 2, 2, 0},
    };
    std::vector<double> inputVector = {1, 2, 3, 0};
    std::vector<double> flatMatrix = {1, 1, 1, 0, 2, 2, 2, 0};
    std::vector<double> flatVector = {1, 2, 3, 0, 1, 2, 3, 0};

    // Set CKKS encryption parameters
    CryptoParams encryptionParams;
    encryptionParams.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    encryptionParams.SetRingDim(1 << 12);
    encryptionParams.SetBatchSize(1 << 11);
    encryptionParams.SetMultiplicativeDepth(2);
    encryptionParams.SetScalingModSize(59);
    encryptionParams.SetFirstModSize(60);
    encryptionParams.SetKeySwitchTechnique(lbcrypto::HYBRID);
    encryptionParams.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    encryptionParams.SetNumLargeDigits(0);
    encryptionParams.SetMaxRelinSkDeg(1);

    std::cout << "Initializing CryptoContext...\n";
    CryptoContext cryptoContext = GenCryptoContext(encryptionParams);
    cryptoContext->Enable(lbcrypto::PKE);
    cryptoContext->Enable(lbcrypto::LEVELEDSHE);
    cryptoContext->Enable(lbcrypto::ADVANCEDSHE);

    std::cout << "Generating key pair...\n";
    KeyPair keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);

    MatKeys columnSumKeys = cryptoContext->EvalSumColsKeyGen(keyPair.secretKey);
    MatKeys rowSumKeys = cryptoContext->EvalSumRowsKeyGen(keyPair.secretKey, nullptr, paddedRowSize);

    // Encode and encrypt matrix and vector
    Plaintext encodedMatrix = cryptoContext->MakeCKKSPackedPlaintext(flatMatrix);
    Plaintext encodedVector = cryptoContext->MakeCKKSPackedPlaintext(flatVector);
    Ciphertext encryptedMatrix = cryptoContext->Encrypt(keyPair.publicKey, encodedMatrix);
    Ciphertext encryptedVector = cryptoContext->Encrypt(keyPair.publicKey, encodedVector);

    if (verbose) {
        std::cout << "--- Plaintext Matrix-Vector Product ---\n";
        PrintVector(MulMatVec(inputMatrix, inputVector));
    }

    // Perform encrypted matrix-vector multiplication
    Ciphertext encryptedProduct = EvalMultMatVec(cryptoContext, columnSumKeys, MatVecEncoding::MM_CRC, paddedRowSize,
                                                 encryptedVector, encryptedMatrix);

    // Decrypt result
    Plaintext ptResult;
    cryptoContext->Decrypt(keyPair.secretKey, encryptedProduct, &ptResult);
    ptResult->SetLength(numCols * numRows);
    std::vector<double> resultVector = ptResult->GetRealPackedValue();

    if (verbose) {
        std::cout << "--- Encrypted Matrix-Vector Result ---\n";
        PrintVector(resultVector);
    }

    std::cout << "Matrix-Vector Demo Complete.\n";
}

// ============================================================
// Demo: Homomorphic Matrix-Matrix Multiplication
// ============================================================
void RunMatrixMultiplicationDemo(bool verbose = true) {
    std::vector<std::vector<double>> matrixA = {
        {1, 1, 1, 0},
        {2, 2, 2, 0},
        {3, 3, 3, 0},
        {4, 4, 4, 0},
    };

    std::vector<std::vector<double>> matrixB = {
        {1, 0, 1, 0},
        {1, 1, 0, 0},
        {3, 0, 3, 0},
        {3, 0, 2, 0},
    };

    uint32_t numRows = matrixA.size();
    uint32_t numCols = matrixA[0].size();
    uint32_t paddedRowSize = NextPow2(numCols);
    int32_t batchSize = (1 << 10) / 2;

    std::vector<double> encodedA = EncodeMatrix<double>(matrixA, batchSize);
    std::vector<double> encodedB = EncodeMatrix<double>(matrixB, batchSize);

    // Set CKKS encryption parameters
    CryptoParams encryptionParams;
    encryptionParams.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    encryptionParams.SetRingDim(1 << 10);
    encryptionParams.SetBatchSize(batchSize);
    encryptionParams.SetMultiplicativeDepth(20);
    encryptionParams.SetScalingModSize(59);
    encryptionParams.SetFirstModSize(60);
    encryptionParams.SetKeySwitchTechnique(lbcrypto::HYBRID);
    encryptionParams.SetScalingTechnique(lbcrypto::FIXEDAUTO);
    encryptionParams.SetNumLargeDigits(0);
    encryptionParams.SetMaxRelinSkDeg(1);

    std::cout << "Initializing CryptoContext...\n";
    CryptoContext cryptoContext = GenCryptoContext(encryptionParams);
    cryptoContext->Enable(lbcrypto::PKE);
    cryptoContext->Enable(lbcrypto::LEVELEDSHE);
    cryptoContext->Enable(lbcrypto::ADVANCEDSHE);

    std::cout << "Generating key pair...\n";
    KeyPair keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);

    // Encrypt matrices
    Plaintext ptA = cryptoContext->MakeCKKSPackedPlaintext(encodedA);
    Plaintext ptB = cryptoContext->MakeCKKSPackedPlaintext(encodedB);
    Ciphertext encryptedA = cryptoContext->Encrypt(keyPair.publicKey, ptA);
    Ciphertext encryptedB = cryptoContext->Encrypt(keyPair.publicKey, ptB);

    if (verbose) {
        std::cout << "--- Plaintext Matrix-Matrix Product ---\n";
        PrintMatrix(MulMats(matrixA, matrixB));
    }

    // Perform encrypted matrix-matrix multiplication
    MulMatRotateKeyGen(cryptoContext, keyPair, paddedRowSize);
    Ciphertext ctProduct = EvalMatMulSquare(cryptoContext, keyPair.publicKey, encryptedA, encryptedB, paddedRowSize);

    // Decrypt result
    Plaintext ptResult;
    cryptoContext->Decrypt(keyPair.secretKey, ctProduct, &ptResult);
    ptResult->SetLength(numCols * numRows);
    std::vector<double> resultVector = ptResult->GetRealPackedValue();

    if (verbose) {
        std::cout << "--- Encrypted Matrix-Matrix Result ---\n";
        RoundVector(resultVector);
        PrintVector(resultVector);
    }

    // Transpose the first encrypted matrix
    Ciphertext encryptedTranspose = EvalMatrixTranspose(cryptoContext, keyPair, encryptedA, paddedRowSize);
    cryptoContext->Decrypt(keyPair.secretKey, encryptedTranspose, &ptResult);
    // ptResult->SetLength(numCols * numRows);
    resultVector = ptResult->GetRealPackedValue();

    if (verbose) {
        std::cout << "--- Encrypted Transposed Matrix Result ---\n";
        RoundVector(resultVector);
        PrintVector(resultVector);
    }

    std::cout << "Matrix-Matrix Demo Complete.\n";
}


int main(int argc, char* argv[]) {
    RunMatrixMultiplicationDemo();
    // RunMatrixVectorDemo();
    return 0;
}
