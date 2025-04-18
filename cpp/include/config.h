// Config.h
#ifndef CONFIG_H
#define CONFIG_H

// -----------------------------------------------------------------------------
// Required Libraries
// -----------------------------------------------------------------------------
#include <vector>
#include <string>
#include "openfhe.h"

// -----------------------------------------------------------------------------
// Debugging
// -----------------------------------------------------------------------------
// #define DEBUG  // Enable debug output (disable/comment this line in production)

// -----------------------------------------------------------------------------
// Encoding Style Enum
// -----------------------------------------------------------------------------
enum MatVecEncoding {
    MM_CRC  = 0,  // Matrix: row-wise, Vector: column-wise → Result: column-wise
    MM_RCR  = 1,  // Matrix: column-wise, Vector: row-wise → Result: row-wise
    MM_DIAG = 2   // Diagonal-style encoding
};

// -----------------------------------------------------------------------------
// Permutation Type Enum
// -----------------------------------------------------------------------------
enum LinTransType {
    SIGMA = 0,
    TAU   = 1,
    PHI   = 2,
    PSI   = 3,
    TRANSPOSE = 4
};

// -----------------------------------------------------------------------------
// Type Aliases for OpenFHE
// -----------------------------------------------------------------------------
using CC         = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;                     // Crypto context
using CT         = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;                        // Ciphertext
using PT         = lbcrypto::Plaintext;                                             // Plaintext
using CryptoParams = lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>;           // CKKS context parameters
using KeyPair    = lbcrypto::KeyPair<lbcrypto::DCRTPoly>;                           // Keypair
using PublicKey   = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
// automorphism keys for EvalSumRows/EvalSumCols; works only for packed encoding
using MatKeys    = std::shared_ptr<std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>; 

#endif  // CONFIG_H
