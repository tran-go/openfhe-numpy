#include "openfhe_numpy/enc_matrix.h"
#include "openfhe.h"
#include "openfhe_numpy/utils.h"

using namespace openfhe_numpy;
using namespace lbcrypto;

CryptoContext<DCRTPoly> GenerateCryptoContext(uint32_t multDepth, uint32_t batchSize = 0) {
    // Step 1: Setup CryptoContext

    // A. Specify main parameters

    /* A2) Bit-length of scaling factor.
    * CKKS works for real numbers, but these numbers are encoded as integers.
    * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
    * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
    * integer). Say the result of a computation based on m' is 130, then at
    * decryption, the scaling factor is removed so the user is presented with
    * the real number result of 0.13.
    *
    * Parameter 'scaleModSize' determines the bit-length of the scaling
    * factor D, but not the scaling factor itself. The latter is implementation
    * specific, and it may also vary between ciphertexts in certain versions of
    * CKKS (e.g., in FLEXIBLEAUTO).
    *
    * Choosing 'scaleModSize' depends on the desired accuracy of the
    * computation, as well as the remaining parameters like multDepth or security
    * standard. This is because the remaining parameters determine how much noise
    * will be incurred during the computation (remember CKKS is an approximate
    * scheme that incurs small amounts of noise with every operation). The
    * scaling factor should be large enough to both accommodate this noise and
    * support results that match the desired accuracy.
    */
    uint32_t scaleModSize = 50;

    /* A4) Desired security level based on FHE standards.
    * This parameter can take four values. Three of the possible values
    * correspond to 128-bit, 192-bit, and 256-bit security, and the fourth value
    * corresponds to "NotSet", which means that the user is responsible for
    * choosing security parameters. Naturally, "NotSet" should be used only in
    * non-production environments, or by experts who understand the security
    * implications of their choices.
    *
    * If a given security level is selected, the library will consult the current
    * security parameter tables defined by the FHE standards consortium
    * (https://homomorphicencryption.org/introduction/) to automatically
    * select the security parameters. Please see "TABLES of RECOMMENDED
    * PARAMETERS" in  the following reference for more details:
    * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
    */
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetBatchSize(1 << 11);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    return cc;
}

// std::vector<std::complex<double>> GenMaskSumRows(int k, int slots, int numRows, int numCols) {
//     auto blockSize = numCols * numRows;
//     auto n         = (int)(slots / blockSize);
//     std::vector<std::complex<double>> mask(slots, 0);

//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < numCols; j++) {
//             if (i * blockSize + numCols * k + j < slots)
//                 mask[i * blockSize + numCols * k + j] = 1.0;
//         }
//     }
//     return mask;
// }

// std::vector<std::complex<double>> GenMaskSumCols(int k, int slots, int numCols) {
//     auto n = (int)(slots / numCols);

//     std::vector<std::complex<double>> result(slots, 0);

//     for (int i = 0; i < n; ++i) {
//         result[i * numCols + k] = 1.0;
//     }
//     return result;
// }

// void Debug(CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keys, Ciphertext<DCRTPoly> ct, std::string msg, int length) {
//     Plaintext pt;
//     cc->Decrypt(keys.secretKey, ct, &pt);
//     pt->SetLength(length);
//     std::vector<double> v = pt->GetRealPackedValue();
//     std::cout << msg << std::endl;
//     RoundVector(v);
//     PrintVector(v);
//     std::cout << std::endl;
// };

void DemoAccumulationOperations() {
    std::cout << "\n~~~ Demo Matrix Accumulation by Rows ~~~\n";
    std::vector<std::vector<double>> matA = {
        {1, 1, 1, 0, 1, 0, 0, 0},
        {2, 2, 2, 0, 2, 0, 0, 0},
        {3, 3, 3, 0, 3, 0, 0, 0},
        {4, 4, 4, 0, 4, 0, 0, 0},
    };


    // Set CKKS encryption parameters
    std::cout << "\nInitializing CryptoContext ...\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
    usint ringDim              = cc->GetRingDimension();

    std::cout << "\nGenerating key pair ...\n";
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Matrix Information
    auto nRows     = matA.size();
    auto nCols     = matA[0].size();
    auto batchSize = ringDim / 2;
    nRows          = NextPow2(nRows);
    nCols          = NextPow2(nCols);

    // Vectorize the matrix before encryption
    std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);
    std::vector<double> tmp(encMatA.begin(), encMatA.size() > 64 ? encMatA.begin() + 64 : encMatA.end());

    std::cout << "Parameters ...\n";
    std::cout << "Ring dimension = " << ringDim << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "nRows = " << nRows << std::endl;
    std::cout << "nCols = " << nCols << std::endl;
    // std::cout << "nPaddedCols = " << nPaddedCols << std::endl;
    std::cout << "tmp = " << tmp << std::endl;
    std::cout << "encMatA size = " << encMatA.size() << std::endl;
    std::cout << "tmp size = " << tmp.size() << std::endl;

    auto slots = batchSize;

    Plaintext ptMatA = cc->MakeCKKSPackedPlaintext(encMatA);
    auto ciphertext = cc->Encrypt(keyPair.publicKey, ptMatA);
    Plaintext ptResult;

    // Sum the matrix by rows
    std::cout << "============================== " << std::endl;
    EvalSumCumRowsKeyGen(keyPair.secretKey, nCols);
    auto ctResult = EvalSumCumRows(ciphertext, nCols, nRows, slots);
    std::cout << "============================ 222222222= " << std::endl;

    // Decrypt the ciphertext to check the result
    
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nCols * nRows);
    auto resultVector = ptResult->GetRealPackedValue();

    std::cout << "--- Encrypted Sum Accumulation on Rows ---\n";
    RoundVector(resultVector);
    PrintVector(resultVector);

    // Sum the matrix by cols
    std::cout << "============================== " << std::endl;
    EvalSumCumColsKeyGen(keyPair.secretKey, nCols);
    ctResult = EvalSumCumCols(ciphertext, nCols);
    std::cout << "============================ 222222222= " << std::endl;

    // Decrypt the ciphertext to check the result
    cc->Decrypt(keyPair.secretKey, ctResult, &ptResult);
    ptResult->SetLength(nCols * nRows);
    resultVector = ptResult->GetRealPackedValue();

    std::cout << "--- Encrypted Sum Accumulation on Cols ---\n";
    RoundVector(resultVector);
    PrintVector(resultVector);
    

    // // std::vector<std::complex<double>> mask = GenMaskSumCols(0, slots, nCols);

    // auto ctTmp = ciphertext->Clone();

    // Debug(cc, keyPair, ctTmp, "ctTmp", 64);
    // cc->EvalRotateKeyGen(keyPair.secretKey, {-1});
    
    // // for (size_t i = 1; i < static_cast<size_t>(nCols); ++i) {
    // //     std::cout << "i = " << i << std::endl;
    // //     auto mask        = GenMaskSumCols(i, slots, nCols);
    // //     auto ptmask = cc->MakeCKKSPackedPlaintext(mask, ciphertext->GetScalingFactor(), 0, nullptr, slots);

    // //     auto rotated = cc->EvalRotate(ctTmp, -1);
    // //     Debug(cc, keyPair, rotated, "rotated ctTmp", 64);
    // //     auto maskedRotated = cc->EvalMult(rotated, ptmask);
    // //     Debug(cc, keyPair, maskedRotated, "maskedRotated", 64);
    // //     cc->EvalAddInPlace(ctTmp, maskedRotated);
    // //     Debug(cc, keyPair, ctTmp, "ctTmp", 64);
    // // }

    // ctTmp = EvalSumCummCols(ctTmp,nCols);

    // // Decrypt the ciphertext to check the result
    // Plaintext ptResult;
    // cc->Decrypt(keyPair.secretKey, ctTmp, &ptResult);
    // ptResult->SetLength(nCols * nRows);
    // auto resultVector = ptResult->GetRealPackedValue();

    // std::cout << "--- Encrypted Accumulation Matrix Result ---\n";
    // RoundVector(resultVector);
    // PrintVector(resultVector);

    std::cout << "~~~ Demo Complete ~~~\n";
}

// ============================================================
// Demo: Homomorphic Matrix-Matrix Multiplication
// ============================================================
void DemoMatrixTranspose() {
    std::cout << "\n~~~ Demo Matrix Transpose ~~~\n";
    std::vector<std::vector<double>> matA = {
        {1, 1, 1, 0},
        {2, 2, 2, 0},
        {3, 3, 3, 0},
        {4, 4, 4, 0},
    };

    // Set CKKS encryption parameters
    std::cout << "\nInitializing CryptoContext ...\n";
    CryptoContext<DCRTPoly> cc = GenerateCryptoContext(20);
    usint ringDim              = cc->GetRingDimension();

    std::cout << "\nGenerating key pair ...\n";
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    // Matrix Information
    auto nRows          = matA.size();
    auto nCols          = matA[0].size();
    auto paddedRowCount = NextPow2(nCols);
    auto batchSize      = ringDim / 2;

    // Vectorize the matrix before encryption
    std::vector<double> encMatA = EncodeMatrix<double>(matA, batchSize);

    std::vector<double> tmp(encMatA.begin(), encMatA.size() > 32 ? encMatA.begin() + 32 : encMatA.end());

    std::cout << "\nMatrix Parameters ...\n";
    std::cout << "nRows = " << nRows << std::endl;
    std::cout << "nCols = " << nCols << std::endl;
    std::cout << "paddedRowCount = " << paddedRowCount << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "encMatA = " << tmp << std::endl;

    Plaintext ptMatA = cc->MakeCKKSPackedPlaintext(encMatA);
    auto ctMatA      = cc->Encrypt(keyPair.publicKey, ptMatA);

    // Transpose the matrix
    Plaintext ptResult;
    auto encryptedTranspose = EvalTranspose(keyPair.secretKey, ctMatA, paddedRowCount);
    cc->Decrypt(keyPair.secretKey, encryptedTranspose, &ptResult);
    ptResult->SetLength(nCols * nRows);
    auto resultVector = ptResult->GetRealPackedValue();

    std::cout << "--- Encrypted Transposed Matrix Result ---\n";
    RoundVector(resultVector);
    PrintVector(resultVector);

    std::cout << "~~~ Demo Complete ~~~\n";
}

int main(int argc, char* argv[]) {
    // DemoMatrixTranspose();
    DemoAccumulationOperations();
    return 0;
}
