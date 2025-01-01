#ifndef UTILS_H
#define UTILS_H

// load necessary library
#include "config.h"

uint32_t NextPow2(const uint32_t x);
std::vector<double> GenSigmaDiag(const uint32_t row_size, const uint32_t k);
std::vector<double> GenTauDiag(const uint32_t row_size, const uint32_t k);
std::vector<double> GenPhiDiag(const uint32_t row_size, const uint32_t k, const int type);
std::vector<double> GenPsiDiag(const uint32_t row_size, const uint32_t k);
#endif // UTILS_H