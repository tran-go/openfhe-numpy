// Config.h
#ifndef CONFIG_H
#define CONFIG_H

// -----------------------------------------------------------------------------
// Required Libraries
// -----------------------------------------------------------------------------
#include <string>
#include <vector>

#include "openfhe.h"

namespace openfhe_matrix
{
    // -----------------------------------------------------------------------------
    // Encoding Style Enum
    // -----------------------------------------------------------------------------
    enum class MatVecEncoding
    {
        MM_CRC = 0,  // Matrix: row-wise, Vector: column-wise → Result: column-wise
        MM_RCR = 1,  // Matrix: column-wise, Vector: row-wise → Result: row-wise
        MM_DIAG = 2  // Diagonal-style encoding
    };

    // -----------------------------------------------------------------------------
    // Permutation Type Enum
    // -----------------------------------------------------------------------------
    enum class LinTransType
    {
        SIGMA = 0,
        TAU = 1,
        PHI = 2,
        PSI = 3,
        TRANSPOSE = 4
    };

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
}
#endif  // CONFIG_H
