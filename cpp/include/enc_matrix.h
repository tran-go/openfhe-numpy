#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

// #include "openfhe.h"
#include "config.h"
#include "helper.h"
#include "utils.h"

#define ENC_MATRIX_API

void EvalLinTransKeyGenFromInt(CryptoContext& cc, const KeyPair& keyPair, int32_t rowSize, int typeInt,
    int32_t repeats = 0);

void EvalLinTransKeyGen(CryptoContext& cryptoContext, const KeyPair& keyPair, int32_t rowSize, LinTransType type,
                        int32_t nRepeats = 0);
void MulMatRotateKeyGen(CryptoContext& cryptoContext, const KeyPair& keyPair, int32_t rowSize);
Ciphertext EvalMultMatVec(CryptoContext& cryptoContext, MatKeys evalKeys, MatVecEncoding encodeType, int32_t rowSize,
                          const Ciphertext& ctVector, const Ciphertext& ctMatrix);

Ciphertext EvalLinTransSigma(CryptoContext& cryptoContext, const PublicKey& publicKey, const Ciphertext& ctVector,
                             int32_t rowSize);
Ciphertext EvalLinTransSigma(CryptoContext& cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                             int32_t rowSize);
Ciphertext EvalLinTransTau(CryptoContext& cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize);
Ciphertext EvalLinTransPhi(CryptoContext& cryptoContext, const PublicKey& publicKey, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats);

Ciphertext EvalLinTransPhi(CryptoContext& cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats);
Ciphertext EvalLinTransPsi(CryptoContext& cryptoContext, const Ciphertext& ctVector, int32_t rowSize, int32_t nRepeats);
Ciphertext EvalLinTransPsi(CryptoContext& cryptoContext, const KeyPair& keyPair, const Ciphertext& ctVector,
                           int32_t rowSize, int32_t nRepeats);

Ciphertext EvalMatMulSquare(CryptoContext& cryptoContext, const PublicKey& publicKey, const Ciphertext& matrixA,
                            const Ciphertext& matrixB, int32_t rowSize);
Ciphertext EvalMatrixTranspose(CryptoContext& cryptoContext, const KeyPair& keyPair, const Ciphertext& ctMatrix,
                               int32_t rowSize);
Ciphertext EvalMatrixTranspose(CryptoContext& cryptoContext, const PublicKey& publicKey, const Ciphertext& ctMatrix,
                               int32_t rowSize);
#endif  // ENC_MATRIX_H