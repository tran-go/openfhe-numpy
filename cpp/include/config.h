// Config.h
#ifndef CONFIG_H
#define CONFIG_H

// -----------------------------------------------------------------------------
// Required Libraries
// -----------------------------------------------------------------------------
#include <string>
#include <vector>

#include "openfhe.h"

namespace openfhe_matrix {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
inline constexpr const std::string_view METADATA_ARRAYINFO_TAG = "arrayInfo";

// -----------------------------------------------------------------------------
// Encoding Style Enum
// -----------------------------------------------------------------------------
enum class MatVecEncoding : std::uint8_t {
    MM_CRC  = 0,  // Matrix: row-wise, Vector: column-wise → Result: column-wise
    MM_RCR  = 1,  // Matrix: column-wise, Vector: row-wise → Result: row-wise
    MM_DIAG = 2   // Diagonal-style encoding
};

// -----------------------------------------------------------------------------
// Permutation Type Enum
// -----------------------------------------------------------------------------
enum class LinTransType : std::uint8_t { SIGMA = 0, TAU = 1, PHI = 2, PSI = 3, TRANSPOSE = 4 };

// -----------------------------------------------------------------------------
// Array Encoding Type Enum
// -----------------------------------------------------------------------------
enum class ArrayEncodingType : std::uint8_t { ROW_MAJOR = 0, COL_MAJOR = 1, DIAG_MAJOR = 2 };

// -----------------------------------------------------------------------------
// Type Aliases for OpenFHE
// -----------------------------------------------------------------------------
// using DCRTPoly           = lbcrypto::DCRTPoly;
// using CryptoContext      = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;        // Crypto
// context using Ciphertext         = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;              //
// Ciphertext using ConstCiphertext    = lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>; using
// Plaintext          = lbcrypto::Plaintext;                                    // Plaintext
// using CryptoParams       = lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>;  // CKKS
// context parameters using KeyPair            = lbcrypto::KeyPair<lbcrypto::DCRTPoly>; //
// Keypair using PublicKey          = lbcrypto::PublicKey<lbcrypto::DCRTPoly>; using EvalKey =
// lbcrypto::EvalKey<lbcrypto::DCRTPoly>;
// // automorphism keys for EvalSumRows/EvalSumCols; works only for packed encoding
template <typename Element>
using MatKeys = std::shared_ptr<std::map<usint, lbcrypto::EvalKey<Element>>>;
}  // namespace openfhe_matrix
#endif  // CONFIG_H
