#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

// #include "openfhe.h"
#include "config.h"
#include "utils.h"
#include "helper.h"

void test();
CT EvalMultMatVec(CC& cc, KeyPair keys, MatKeys eval_keys, int type, uint32_t row_size, const CT& ct_vec, const CT& ct_mat);

CT EvalLinTransShift(CC cc, KeyPair keys, const CT c_vec, const int opt, const uint32_t row_size);

CT EvalLinTransPhi(CC cc, KeyPair keys, const CT c_vec, const uint32_t row_size, const uint32_t k);

CT EvalLinTransPsi(CC cc, KeyPair keys, const CT c_vec, const uint32_t row_size, const uint32_t k);

CT EvalMatMulSquare(CC cc, KeyPair keys, const CT cmat_A, const CT cmat_B, const uint32_t row_size);

CT EvalLinTrans(CC& cc, KeyPair keys, MatKeys eval_keys, int type, uint32_t row_size, const CT& ct_vec, const CT& ct_mat);

#endif // ENC_MATRIX_H