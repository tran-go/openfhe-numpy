#include "openfhe_numpy/utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

// utils implementation
using namespace lbcrypto;

void RoundVector(std::vector<double>& vector) {
    std::for_each(std::begin(vector), std::end(vector), [](double& e) { e = std::round(e); });
}

uint32_t NextPow2(uint32_t x) {
    return pow(2, ceil(log(double(x)) / log(2.0)));
};

uint32_t IsPow2(uint32_t x) {
    return sqrt(x)*sqrt(x) == x; 
};


void Debug(CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keys, Ciphertext<DCRTPoly> ct, std::string msg, int length) {
    Plaintext pt;
    cc->Decrypt(keys.secretKey, ct, &pt);
    pt->SetLength(length);
    std::vector<double> v = pt->GetRealPackedValue();
    std::cout << msg << std::endl;
    RoundVector(v);
    PrintVector(v);
    std::cout << std::endl;
};

/*
Compute diagonals for the permutation matrix Sigma.
B[i,j] = A[i, i +j]
*/
std::vector<double> GenSigmaDiag(int32_t numCols, int32_t k) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(n, 0);

    if (k >= 0) {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - numCols * k;
            if ((0 <= tmp) and (tmp < numCols - k)) {
                diag[i] = 1;
            }
        }
    }
    else {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - (numCols + k) * numCols;
            if ((-k <= tmp) and (tmp < numCols)) {
                diag[i] = 1;
            }
        }
    }
    return diag;
}

/*
Compute diagonals  for the permutation matrix Tau.
B[i,j] = A[i + j,i]
u_[d.k][k + d*i] = 1 for all 0 <= i < d
*/

std::vector<double> GenTauDiag(int32_t totalSlots, int32_t numCols, int32_t k) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);

    for (auto t = 0; t < totalSlots / n; t++) {
        for (auto i = 0; i < numCols; i++) {
            diag[(t * n) + k + numCols * i] = 1;
        }
    }
    return diag;
}

/**
 *Compute diagonals for the permutation matrix Phi (V).
 *B[i,j] = A[i,j+1]
 *There are two diagonals in the matrix Phi.
 *Type = 0 correspond for the k-th diagonal, and type = 1 is for the (k-d)-th
 *diagonal
 */
std::vector<double> GenPhiDiag(int32_t numCols, int32_t k, int type) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(n, 0);

    if (type == 0) {
        for (int32_t i = 0; i < n; i++)
            if ((i % numCols >= 0) and ((i % numCols) < numCols - k))
                diag[i] = 1;
        return diag;
    }
    for (int32_t i = 0; i < n; i++)
        if ((i % numCols >= numCols - k) and (i % numCols < numCols)) {
            diag[i] = 1;
        }

    return diag;
}

/**
 *Compute diagonals for the permutation Psi (W).
 *B[i,j] = A[i+1,j]
 */
std::vector<double> GenPsiDiag(int32_t numCols, int32_t k) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(n, 1);
    return diag;
}

std::vector<double> GenTransposeDiag(int32_t totalSlots, int32_t numCols, int32_t i) {
    int32_t n = numCols * numCols;
    std::vector<double> diag(totalSlots, 0);
    for (auto t = 0; t < totalSlots / n; t++) {
        if (i >= 0) {
            for (auto j = 0; j < numCols - i; j++) {
                auto idx = t * n + (numCols + 1) * j + i;
                if (idx < totalSlots)
                    diag[idx] = 1;
            }
        }
        else {
            for (auto j = -i; j < numCols; j++) {
                auto idx = t * n + (numCols + 1) * j + i;
                if (idx < totalSlots)
                    diag[idx] = 1;
            }
        }
    }
    return diag;
}
