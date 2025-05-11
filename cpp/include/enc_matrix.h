#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

#include "config.h"
#include "helper.h"
#include "utils.h"

#define ENC_MATRIX_API
namespace openfhe_matrix {

using namespace lbcrypto;

//TODO: Change from const Ciphertext to ConstCiphertext
//TODO: using references

template <typename Element>
void EvalLinTransKeyGen(PrivateKey<Element>& secretKey, int32_t numCols, LinTransType type, int32_t numRepeats = 0);

template <typename Element>
void EvalAccumulationKeyGen(PrivateKey<Element>& secretKey, int32_t numRows, int32_t numCols);

template <typename Element>
void EvalSquareMatMultRotateKeyGen(PrivateKey<Element>& secretKey, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalMultMatVec(MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t numCols,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(PrivateKey<Element>& secretKey,
                                      const Ciphertext<Element>& ciphertext,
                                      int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(const Ciphertext<Element>& ciphertext, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(const Ciphertext<Element>& ctVector, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ciphertext,
                                    int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPhi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(PrivateKey<Element>& secretKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t numCols,
                                    int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(const Ciphertext<Element>& ctVector, int32_t numCols, int32_t numRepeats);

template <typename Element>
Ciphertext<Element> EvalMatMulSquare(const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalTranspose(PrivateKey<Element>& secretKey,
                                  const Ciphertext<Element>& ciphertext,
                                  int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalTranspose(const Ciphertext<Element>& ciphertext, int32_t numCols);

template <typename Element>
Ciphertext<Element> EvalAddAccumulateRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

template <typename Element>
Ciphertext<Element> EvalAddAccumulateCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);

template <typename Element>
Ciphertext<Element> EvalSubAccumulateRows(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t numRows = 0,
                                          uint32_t slots   = 0);

template <typename Element>
Ciphertext<Element> EvalSubAccumulateCols(const Ciphertext<Element>& ciphertext,
                                          uint32_t numCols,
                                          uint32_t subringDim = 0);
}
#endif  // ENC_MATRIX_H