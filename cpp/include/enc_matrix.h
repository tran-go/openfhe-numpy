#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

#include "config.h"
#include "helper.h"
#include "utils.h"

#define ENC_MATRIX_API
namespace openfhe_matrix {

using namespace lbcrypto;

template <typename Element>
void EvalLinTransKeyGen(PrivateKey<Element>& secretKey, int32_t rowSize, LinTransType type, int32_t numRepeats = 0);

template <typename Element>
void EvalAccumulationKeyGen(PrivateKey<Element>& secretKey, int32_t rowSize);

template <typename Element>
void EvalSquareMatMultRotateKeyGen(PrivateKey<Element>& secretKey, int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalMultMatVec(MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t rowSize,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(PrivateKey<Element>& secretKey,
                                      const Ciphertext<Element>& ciphertext,
                                      int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(const Ciphertext<Element>& ciphertext, int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(const Ciphertext<Element>& ctVector, int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ciphertext,
                                    int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(const Ciphertext<Element>& ctVector, int32_t rowSize, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(const Ciphertext<Element>& ctVector, int32_t rowSize, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalMatMulSquare(const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalTranspose(PrivateKey<Element>& secretKey,
                                  const Ciphertext<Element>& ciphertext,
                                  int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalTranspose(const Ciphertext<Element>& ciphertext, int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalAddAccumulateRows(ConstCiphertext<Element>& ciphertext, uint32_t rowSize, uint32_t subringDim=0);

}  // namespace openfhe_matrix
#endif  // ENC_MATRIX_H