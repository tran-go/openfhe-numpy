// Config.h
#ifndef CONFIG_H
#define CONFIG_H

/* load necessary library */
#include <vector>
#include <string>
#include "openfhe.h"


/*  define flags  */
#define DEBUG

/* define constants */
// define encoding style
enum encode_style
{
  // pack matrix row-wise and vector column-wise, result is column-wise
  MM_CRC = 0,
  // pack matrix column-wise and vector row-wise, result i row-wise
  MM_RCR = 1,
  // pack matrix diagonal
  MM_DIAG = 2
};

// define the type of permutation  
enum perm_style
{
  SIGMA = 0,
  TAU = 1,
  PHI = 2,
  PSI = 3
};

/* rename common types */
using CC = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>; // crypto contexts
using CT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;    // ciphertext
using PT = lbcrypto::Plaintext;                         // plaintext
using CryptoParams = lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>;
using KeyPair = lbcrypto::KeyPair<lbcrypto::DCRTPoly>;
using MatKeys = std::shared_ptr<std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>;

#endif