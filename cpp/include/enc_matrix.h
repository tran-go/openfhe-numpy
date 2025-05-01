#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

// #include "openfhe.h"
#include "config.h"
#include "helper.h"
#include "utils.h"

#define ENC_MATRIX_API
namespace fhemat {

using namespace lbcrypto;

template <typename Element>
void EvalLinTransKeyGenFromInt(CryptoContext<Element>& cryptoContext,
                               const KeyPair<Element>& keyPair,
                               int32_t rowSize,
                               int linTransTypeInt,
                               int32_t repeats = 0);

template <typename Element>
void EvalLinTransKeyGen(CryptoContext<Element>& cryptoContext,
                        const KeyPair<Element>& keyPair,
                        int32_t rowSize,
                        LinTransType linTransType,
                        int32_t nRepeats = 0);

template <typename Element>
void MulMatRotateKeyGen(CryptoContext<Element>& cryptoContext, const KeyPair<Element>& keyPair, int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalMultMatVec(CryptoContext<Element>& cryptoContext,
                                   MatKeys<Element> evalKeys,
                                   MatVecEncoding encodeType,
                                   int32_t rowSize,
                                   const Ciphertext<Element>& ctVector,
                                   const Ciphertext<Element>& ctMatrix);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(CryptoContext<Element>& cryptoContext,
                                      const PublicKey<Element>& publicKey,
                                      const Ciphertext<Element>& ctVector,
                                      int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransSigma(CryptoContext<Element>& cryptoContext,
                                      const KeyPair<Element>& keyPair,
                                      const Ciphertext<Element>& ctVector,
                                      int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalLinTransTau(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize);
template <typename Element>
Ciphertext<Element> EvalLinTransPhi(CryptoContext<Element>& cryptoContext,
                                    const PublicKey<Element>& publicKey,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats);
template <typename Element>

Ciphertext<Element> EvalLinTransPhi(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats);
template <typename Element>
Ciphertext<Element> EvalLinTransPsi(CryptoContext<Element>& cryptoContext,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats);

template <typename Element>
Ciphertext<Element> EvalLinTransPsi(CryptoContext<Element>& cryptoContext,
                                    const KeyPair<Element>& keyPair,
                                    const Ciphertext<Element>& ctVector,
                                    int32_t rowSize,
                                    int32_t nRepeats);

template <typename Element>

Ciphertext<Element> EvalMatMulSquare(CryptoContext<Element>& cryptoContext,
                                     const PublicKey<Element>& publicKey,
                                     const Ciphertext<Element>& matrixA,
                                     const Ciphertext<Element>& matrixB,
                                     int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalMatrixTranspose(CryptoContext<Element>& cryptoContext,
                                        const KeyPair<Element>& keyPair,
                                        const Ciphertext<Element>& ctMatrix,
                                        int32_t rowSize);

template <typename Element>
Ciphertext<Element> EvalMatrixTranspose(CryptoContext<Element>& cryptoContext,
                                        const PublicKey<Element>& publicKey,
                                        const Ciphertext<Element>& ctMatrix,
                                        int32_t rowSize);
// template <typename Element>
// Ciphertext<Element> EvalAddAccumulateRows(ConstCiphertext<Element> ciphertext,
//                                           uint32_t numRows,
//                                           const std::map<uint32_t, EvalKey<Element>>& evalSumKeys,
//                                           uint32_t subringDim = 0);

// template <typename Element>
// Ciphertext<Element> EvalAddAccumulateCols(ConstCiphertext<Element> ciphertext,
//                                           uint32_t numRows,
//                                           const std::map<uint32_t, EvalKey<Element>>& evalSumKeys,
//                                           uint32_t subringDim = 0);

}  // namespace fhemat
#endif  // ENC_MATRIX_H