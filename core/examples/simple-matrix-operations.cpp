// #include "enc_matrix.h"
// #include "openfhe.h"
// #include "utils.h"

// using namespace openfhe_numpy;

// void AutomaticRescaleDemo(ScalingTechnique scalTech);
// void ManualRescaleDemo(ScalingTechnique scalTech);
// void HybridKeySwitchingDemo1();
// void HybridKeySwitchingDemo2();
// void FastRotationsDemo1();
// void FastRotationsDemo2();

// CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
//     // Step 1: Setup CryptoContext

//     // A. Specify main parameters

//     /* A2) Bit-length of scaling factor.
//     * CKKS works for real numbers, but these numbers are encoded as integers.
//     * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
//     * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
//     * integer). Say the result of a computation based on m' is 130, then at
//     * decryption, the scaling factor is removed so the user is presented with
//     * the real number result of 0.13.
//     *
//     * Parameter 'scaleModSize' determines the bit-length of the scaling
//     * factor D, but not the scaling factor itself. The latter is implementation
//     * specific, and it may also vary between ciphertexts in certain versions of
//     * CKKS (e.g., in FLEXIBLEAUTO).
//     *
//     * Choosing 'scaleModSize' depends on the desired accuracy of the
//     * computation, as well as the remaining parameters like multDepth or security
//     * standard. This is because the remaining parameters determine how much noise
//     * will be incurred during the computation (remember CKKS is an approximate
//     * scheme that incurs small amounts of noise with every operation). The
//     * scaling factor should be large enough to both accommodate this noise and
//     * support results that match the desired accuracy.
//     */
//     uint32_t scaleModSize = 50;

//     /* A4) Desired security level based on FHE standards.
//     * This parameter can take four values. Three of the possible values
//     * correspond to 128-bit, 192-bit, and 256-bit security, and the fourth value
//     * corresponds to "NotSet", which means that the user is responsible for
//     * choosing security parameters. Naturally, "NotSet" should be used only in
//     * non-production environments, or by experts who understand the security
//     * implications of their choices.
//     *
//     * If a given security level is selected, the library will consult the current
//     * security parameter tables defined by the FHE standards consortium
//     * (https://homomorphicencryption.org/introduction/) to automatically
//     * select the security parameters. Please see "TABLES of RECOMMENDED
//     * PARAMETERS" in  the following reference for more details:
//     * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
//     */
//     CCParams<CryptoContextCKKSRNS> parameters;
//     parameters.SetMultiplicativeDepth(multDepth);
//     parameters.SetScalingModSize(scaleModSize);
//     parameters.SetBatchSize(batchSize);

//     CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

//     // Enable the features that you wish to use
//     cc->Enable(lbcrypto::PKE);
//     cc->Enable(lbcrypto::LEVELEDSHE);
//     cc->Enable(lbcrypto::ADVANCEDSHE);
//     std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
//     return cc;
// }

// // int64_t GetMultDepth(int type, int64_t rowSize, int64_t colSize) {
// //     switch (type) {
// //         case 1:
// //             return 10;

// //         default:
// //             break;
// //     }
// // }

// void MatrixVectorProduct_CRC() {
//     // Input plaintext data
//     std::vector<std::vector<double>> inputMatrix = {
//         {1, 1, 1, 0},
//         {2, 2, 2, 0},
//     };
//     std::vector<double> inputVector = {1, 2, 3, 0};
//     std::vector<double> flatMat     = {1, 1, 1, 0, 2, 2, 2, 0};
//     std::vector<double> flatVec     = {1, 2, 3, 0, 1, 2, 3, 0};

//     int64_t nRows          = inputMatrix.size();
//     int64_t nCols          = !inputMatrix.empty() ? inputMatrix[0].size() : 0;
//     int32_t paddedRowCount = NextPow2(nCols);
//     int32_t multDepth      = 20;

//     std::cout << "Initializing CryptoContext...\n";
//     CryptoContext<DCRTPoly> cc = GenerateCryptoContext(multDepth);

//     auto keyPair = cc->KeyGen();
//     cc->EvalMultKeyGen(keyPair.secretKey);
//     auto sumColsKey = cc->EvalSumColsKeyGen(keyPair.secretKey);

//     // Encode and encrypt mat and vector
//     auto ptMat = cc->MakeCKKSPackedPlaintext(flatMat);
//     auto ptVec = cc->MakeCKKSPackedPlaintext(flatVec);
//     auto ctMat = cc->Encrypt(keyPair.publicKey, ptMat);
//     auto ctVec = cc->Encrypt(keyPair.publicKey, ptVec);

//     std::cout << "--- Plaintext Matrix-Vector Product ---\n";
//     PrintVector(MulMatVec(inputMatrix, inputVector));

//     // Perform encrypted mat-vector multiplication
//     auto ctProd = EvalMultMatVec(sumColsKey, MatVecEncoding::MM_CRC, paddedRowCount, ctVec, ctMat);

//     // Decrypt result
//     Plaintext ptResult;
//     cc->Decrypt(keyPair.secretKey, ctProd, &ptResult);
//     ptResult->SetLength(nCols * nRows);
//     std::vector<double> resultVector = ptResult->GetRealPackedValue();
//     PrintVector(resultVector);
//     std::cout << "Matrix-Vector Demo Complete.\n";
// }

// // ============================================================
// // Demo: Homomorphic Matrix-Matrix Multiplication
// // ============================================================
// void DemoMatrixProduct(bool verbose = true) {
//     std::vector<std::vector<double>> matA = {
//         {1, 1, 1, 0},
//         {2, 2, 2, 0},
//         {3, 3, 3, 0},
//         {4, 4, 4, 0},
//     };

//     std::vector<std::vector<double>> matB = {
//         {1, 0, 1, 0},
//         {1, 1, 0, 0},
//         {3, 0, 3, 0},
//         {3, 0, 2, 0},
//     };

//     // Set CKKS encryption parameters
//     std::cout << "\nInitializing CryptoContext ...\n";
//     CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
//     usint ringDim              = cc->GetRingDimension();

//     std::cout << "\nGenerating key pair ...\n";
//     auto keyPair = cc->KeyGen();
//     cc->EvalMultKeyGen(keyPair.secretKey);
//     cc->EvalSumKeyGen(keyPair.secretKey);

//     // Encrypt
//     auto nRows          = matA.size();
//     auto nCols          = matA[0].size();
//     auto paddedRowCount = NextPow2(nCols);
//     auto batchSize      = ringDim / 2;

//     std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);
//     std::vector<double> encMatB = EncodeMatrix<double>(matB, batchSize);

//     std::cout << "\nMatrix Parameters ...\n";
//     std::cout << "nRows = " << nRows << std::endl;
//     std::cout << "nCols = " << nCols << std::endl;
//     std::cout << "paddedRowCount = " << paddedRowCount << std::endl;
//     std::cout << "batchSize = " << batchSize << std::endl;
//     std::cout << "encMatA = " << encMatA << std::endl;

//     Plaintext ptMatA = cc->MakeCKKSPackedPlaintext(encMatA);
//     Plaintext ptMatB = cc->MakeCKKSPackedPlaintext(encMatB);
//     auto ctMatA      = cc->Encrypt(keyPair.publicKey, ptMatA);
//     auto ctMatB      = cc->Encrypt(keyPair.publicKey, ptMatB);

//     if (verbose) {
//         std::cout << "\n--- [Plaintext] Matrix Product Result ---\n";
//         PrintMatrix(MulMats(matA, matB));
//     }

//     // Perform encrypted mat-mat multiplication
//     EvalSquareMatMultRotateKeyGen(keyPair.secretKey, paddedRowCount);
//     auto ctProduct = EvalMatMulSquare(ctMatA, ctMatB, paddedRowCount);

//     // Decrypt result
//     Plaintext ptResult;
//     cc->Decrypt(keyPair.secretKey, ctProduct, &ptResult);
//     ptResult->SetLength(nCols * nRows);
//     std::vector<double> res = ptResult->GetRealPackedValue();

//     if (verbose) {
//         std::cout << "\n--- [Ciphertext] Square Matrix Product Result ---\n";
//         RoundVector(res);
//         PrintVector(res);
//     }

//     // Transpose the first encrypted matrix
//     auto encryptedTranspose = EvalTranspose(keyPair.secretKey, ctMatA, paddedRowCount);
//     cc->Decrypt(keyPair.secretKey, encryptedTranspose, &ptResult);
//     ptResult->SetLength(nCols * nRows);
//     auto resultVector = ptResult->GetRealPackedValue();

//     if (verbose) {
//         std::cout << "--- Encrypted Transposed Matrix Result ---\n";
//         RoundVector(resultVector);
//         PrintVector(resultVector);
//     }

//     std::cout << "Matrix-Matrix Demo Complete.\n";
// }

// int main(int argc, char* argv[]) {
//     DemoMatrixProduct();
//     // RunMatrixVectorDemo();
//     return 0;
// }


// Add this if it's missing:
int main() {
    // Your code here
    return 0;
}