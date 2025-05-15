#ifndef UTILS_H
#define UTILS_H

// load necessary library
#include "openfhe_numpy/config.h"
#include "openfhe_numpy/internal/helper.h"

using namespace lbcrypto;

uint32_t NextPow2(const uint32_t x);
bool IsPow2(uint32_t x);
// template <class Element>
// void Debug(CryptoContext<Element> cc, KeyPair<Element> keys, Ciphertext<Element> ct, std::string msg, int length = 16);
void Debug(CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keys, Ciphertext<DCRTPoly> ct, std::string msg, int length=16);
// void Debug(CryptoContext, KeyPair keys, Ciphertext, std::string msg, int length = 16);
std::vector<double> GenSigmaDiag(std::size_t rowsize, int32_t k);
std::vector<double> GenTauDiag(std::size_t total_slots, std::size_t rowsize, int32_t k);
std::vector<double> GenPhiDiag(std::size_t rowsize, int32_t k, int type);
std::vector<double> GenPsiDiag(std::size_t rowsize, int32_t k);
std::vector<double> GenTransposeDiag(std::size_t total_slots, std::size_t rowsize, int32_t i);
void RoundVector(std::vector<double>& vector);
#endif  // UTILS_H