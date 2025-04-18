#include "enc_matrix.h"

void test() { std::cout << "Hello Boca Raton!!!\n"; }

/*
This work implemented matrix multiplications from the paper
https://eprint.iacr.org/2018/254
*/

// CT EvalMultMatVec(const CC &context, const KeyPair &keys,
//                   const MatKeys &eval_keys, int type, int row_size,
//                   const CT &ct_vec, const CT &ct_mat) {

CT EvalMultMatVec(CC& context, KeyPair keys, MatKeys eval_keys, int type,
                  int row_size, const CT& ct_vec, const CT& ct_mat) {
  CT ct_product;
  auto ct_mult = context->EvalMult(ct_mat, ct_vec);

#ifdef DEBUG
  std::cout << "Debugging enabled" << std::endl;
  PT pt_mult;
  context->Decrypt(keys.secretKey, ct_mult, &pt_mult);
  pt_mult->SetLength(row_size * 4);
  std::vector<double> vec = pt_mult->GetRealPackedValue();
  PrintVector(vec);
#endif

  if (type == EncodeStyle::MM_CRC) {
    std::cout << "Using EncodeStyle = " << EncodeStyle::MM_CRC
              << " with row_size = " << row_size << std::endl;
    ct_product = context->EvalSumCols(ct_mult, row_size, *eval_keys);

#ifdef DEBUG
    PT pt_tmp;
    context->Decrypt(keys.secretKey, ct_product, &pt_tmp);
    pt_tmp->SetLength(row_size * 4);
    std::vector<double> vec = pt_tmp->GetRealPackedValue();
    PrintVector(vec);
#endif
  } else if (type == EncodeStyle::MM_RCR) {
    std::cout << "Using EncodeStyle = " << EncodeStyle::MM_RCR
              << " with row_size = " << row_size << std::endl;

    ct_product = context->EvalSumRows(ct_mult, row_size, *eval_keys);
  } else {
    printf("To be continue ... \n");
  }
  return ct_product;
};

/*
Generate the diagonals for the W permutation
*/

CT EvalLinTransSigma(CC cc, KeyPair keys, const CT c_vec, const int row_size) {
  int n = row_size * row_size;
  CT c_result, c_tmp;
  std::vector<double> result(n, 0.0);

  PT p_result = cc->MakeCKKSPackedPlaintext(result);
  c_result = cc->Encrypt(keys.publicKey, p_result);

  std::vector<int> rotate_indices(2 * row_size, 0);
  for (int k = -row_size; k < row_size; k++) {
    rotate_indices[k + row_size] = k;
  }

  cc->EvalRotateKeyGen(keys.secretKey, rotate_indices);
  for (int k = -row_size; k < row_size; k++) {
    std::vector<double> diag = GenSigmaDiag(row_size, k);
    auto p_diag = cc->MakeCKKSPackedPlaintext(diag);
    auto c_rotated = cc->EvalRotate(c_vec, k);
    cc->EvalAddInPlace(c_result, cc->EvalMult(c_rotated, p_diag));
  }

  return c_result;
};

CT EvalLinTransTau(CC cc, KeyPair keys, const CT c_vec, const int row_size) {
  int n = row_size * row_size;
  std::vector<double> zeros(n, 0.0);
  PT p_result = cc->MakeCKKSPackedPlaintext(zeros);
  CT c_result = cc->Encrypt(keys.publicKey, p_result);

  int32_t slots = cc->GetEncodingParams()->GetBatchSize();
  // std::cout<<"slots = "<<slots <<std::endl<<std::endl;

  // std::vector<int> rotate_indices(row_size, 0);
  // for (int k = 0; k < row_size; k++) {
  //   rotate_indices[k] = k * row_size;
  // }

  // cc->EvalRotateKeyGen(keys.secretKey, rotate_indices);

  for (auto k = 0; k < row_size; k++) {
    // std::cout<<"k = "<<k<< " rotate "<< row_size *k <<std::endl;

    auto diag = GenTauDiag(slots, row_size, k);
    // PrintVector(diag);
    auto p_diag = cc->MakeCKKSPackedPlaintext(diag);
    // std::cout<<"Generate key "<<k<< " and rotate " << row_size * k
    // <<std::endl;
    cc->EvalRotateKeyGen(keys.secretKey, {row_size * k});
    auto c_rotated = cc->EvalRotate(c_vec, row_size * k);
    // Debug(cc, keys, c_rotated, "c_rotated in GenTauDiag ");
    auto tmp = cc->EvalMult(p_diag, c_rotated);
    // Debug(cc, keys, tmp, "p_diag * c_rotated in GenTauDiag ");
    cc->EvalAddInPlace(c_result, tmp);

    // Debug(cc, keys, c_result, "c_result in GenTauDiag");
  }

  return c_result;
};

CT EvalLinTransPhi(CC cc, KeyPair keys, const CT c_vec, const int row_size,
                   const int k) {
  int n = row_size * row_size;
  CT c_result, c_tmp;
  std::vector<double> result(n, 0.0);
  PT p_result = cc->MakeCKKSPackedPlaintext(result);
  c_result = cc->Encrypt(keys.publicKey, p_result);

  for (int i = 0; i < 2; i++) {
    auto idx = k - i * row_size;
    auto diag = GenPhiDiag(row_size, k, i);
    PT p_diag = cc->MakeCKKSPackedPlaintext(diag);
    cc->EvalRotateKeyGen(keys.secretKey, {idx});
    CT c_rotated = cc->EvalRotate(c_vec, idx);
    cc->EvalAddInPlace(c_result, cc->EvalMult(c_rotated, p_diag));
  }

  return c_result;
};

CT EvalLinTransPsi(CC cc, KeyPair keys, const CT c_vec, const int row_size,
                   const int k) {
  cc->EvalRotateKeyGen(keys.secretKey, {row_size * k});
  CT c_result = cc->EvalRotate(c_vec, row_size * k);
  return c_result;
}

/*
This work implemented matrix multiplications from the paper
https://eprint.iacr.org/2018/1041
*/

CT EvalMatMulSquare(CC cc, KeyPair keys, const CT cmat_A, const CT cmat_B,
                    const int32_t row_size) {
  // Step 1 - 1
  // std::cout<<"Step 1 - 1"<<std::endl;
  CT ct_A0 = EvalLinTransSigma(cc, keys, cmat_A, row_size);
  // Debug(cc, keys, ct_A0, "_Step 1 - 1: Compute cA0");

  // Step 1 - 2
  // std::cout<<"Step 1 - 2"<<std::endl;
  // Debug(cc, keys, cmat_B, "cmat_B = ");
  CT ct_B0 = EvalLinTransTau(cc, keys, cmat_B, row_size);
  // Debug(cc, keys, ct_B0, "_Step 1 - 2: Compute cB0");
  CT cAB = cc->EvalMult(ct_A0, ct_B0);
  // Debug(cc, keys, cAB, "_Step 1 - 3: Compute cAB");

  // Step 2 and 3
  // std::cout<<"Step 2 and 3"<<std::endl;
  for (int32_t k = 1; k < row_size; k++) {
    // std::cout << "Loop: " << k << std::endl;
    auto cA = EvalLinTransPhi(cc, keys, ct_A0, row_size, k);
    // Debug(cc, keys, cA, "cA");
    auto cB = EvalLinTransPsi(cc, keys, ct_B0, row_size, k);
    // Debug(cc, keys, cB, "cB");
    cAB = cc->EvalAdd(cAB, cc->EvalMult(cA, cB));
    // Debug(cc, keys, cAB, "cAB");
  }
  // std::cout << "Finish MM " << std::endl;
  return cAB;
}

#include <stdexcept>

/**
 * @brief Applies a matrix transposition to an encrypted matrix using diagonal
 * encoding.
 *
 * This function simulates matrix transposition in the CKKS scheme by
 * multiplying the ciphertext with structured diagonal plaintext vectors and
 * applying rotations.
 *
 * @param cc                The CryptoContext for CKKS operations.
 * @param keyPair           The public/secret key pair for encryption and
 * rotation.
 * @param inputCiphertext   The encrypted input matrix (packed as a vector).
 * @param matrixSize        The size of the square matrix (d x d).
 *
 * @return Ciphertext<DCRTPoly> Resulting ciphertext encoding the transposed
 * matrix.
 *
 * @throws std::runtime_error if encryption or homomorphic operations fail.
 */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatrixTranspose(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& inputCiphertext, int32_t matrixSize) {
  try {
    // Total number of elements in the square matrix
    const int64_t totalElements = static_cast<int64_t>(matrixSize) * matrixSize;

    // Number of available slots for packed encoding
    const size_t slotCount = cc->GetEncodingParams()->GetBatchSize();

    // Create a plaintext of zeros to initialize the result ciphertext
    std::vector<double> zeroVector(totalElements, 0.0);
    lbcrypto::Plaintext initialPlaintext = cc->MakeCKKSPackedPlaintext(zeroVector);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> resultCiphertext =
        cc->Encrypt(keyPair.publicKey, initialPlaintext);

    std::cout << "EvaluateMatrixTranspose - Slot count: " << slotCount
              << std::endl;

    // Process each diagonal in the matrix
    for (int32_t diagonalIndex = -matrixSize + 1; diagonalIndex < matrixSize;
         ++diagonalIndex) {
      const int32_t rotationIndex = (matrixSize - 1) * diagonalIndex;

      // Generate and encode the transpose diagonal vector
      std::vector<double> diagonalVector =
      GenTransposeDiag(slotCount, matrixSize, diagonalIndex);
      PrintVector(diagonalVector);  // Optional for debugging
      lbcrypto::Plaintext diagonalPlaintext = cc->MakeCKKSPackedPlaintext(diagonalVector);

      // Generate rotation key (once per rotation index)
      cc->EvalRotateKeyGen(keyPair.secretKey, {rotationIndex});

      // Apply homomorphic rotation and multiplication
      lbcrypto::Ciphertext<lbcrypto::DCRTPoly> rotatedCiphertext =
          cc->EvalRotate(inputCiphertext, rotationIndex);
      Debug(cc, keyPair, rotatedCiphertext, "Rotated ciphertext");

      lbcrypto::Ciphertext<lbcrypto::DCRTPoly> productCiphertext =
          cc->EvalMult(rotatedCiphertext, diagonalPlaintext);

      // Accumulate result
      cc->EvalAddInPlace(resultCiphertext, productCiphertext);
      Debug(cc, keyPair, resultCiphertext, "Updated result ciphertext");
    }

    return resultCiphertext;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] EvaluateMatrixTranspose failed: " << e.what()
              << std::endl;
    throw std::runtime_error(
        "EvaluateMatrixTranspose: Homomorphic operation failed.");
  }
}
