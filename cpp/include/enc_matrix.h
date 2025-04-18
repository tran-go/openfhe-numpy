#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

// #include "openfhe.h"
#include "config.h"
#include "helper.h"
#include "utils.h"

#define ENC_MATRIX_API

void test();
CT EvalMultMatVec(CC& cc, KeyPair keys, MatKeys eval_keys, int type,
                  int row_size, const CT& ct_vec, const CT& ct_mat);

CT EvalLinTransSigma(CC cc, KeyPair keys, const CT c_vec, const int row_size);

CT EvalLinTransTau(CC cc, KeyPair keys, const CT c_vec, const int row_size);

CT EvalLinTransPhi(CC cc, KeyPair keys, const CT c_vec, const int row_size,
                   const int32_t k);

CT EvalLinTransPsi(CC cc, KeyPair keys, const CT c_vec, const int row_size,
                   const int32_t k);

CT EvalMatMulSquare(CC cc, KeyPair keys, const CT cmat_A, const CT cmat_B,
                    const int32_t row_size);

CT EvalTranspose(CC cc, KeyPair keys, const CT ct_mat, const int32_t row_size);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatrixTranspose(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& inputCiphertext,
    int32_t matrixSize);

#endif  // ENC_MATRIX_H