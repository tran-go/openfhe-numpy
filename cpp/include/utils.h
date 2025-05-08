#ifndef UTILS_H
#define UTILS_H

// load necessary library
#include "config.h"
#include "helper.h"

using namespace lbcrypto;

uint32_t NextPow2(const uint32_t x);
template <class Element>
void Debug(CryptoContext<Element> cc, KeyPair<Element> keys, Ciphertext<Element> ct, std::string msg, int length = 16);
// void Debug(CryptoContext, KeyPair keys, Ciphertext, std::string msg, int length = 16);
std::vector<double> GenSigmaDiag(int32_t rowsize, int32_t k);
std::vector<double> GenTauDiag(int32_t total_slots, int32_t rowsize, int32_t k);
std::vector<double> GenPhiDiag(int32_t rowsize, int32_t k, int type);
std::vector<double> GenPsiDiag(int32_t rowsize, int32_t k);
std::vector<double> GenTransposeDiag(int32_t total_slots, int32_t rowsize, int32_t i);
void RoundVector(std::vector<double>& vector);
#endif  // UTILS_H