#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

// utils implementation

void RoundVector(std::vector<double>& vector) {
    std::for_each(std::begin(vector), std::end(vector), [](double& e) { e = std::round(e); });
}

uint32_t NextPow2(uint32_t x) {
    return pow(2, ceil(log(double(x)) / log(2.0)));
};

void Debug(CryptoContext cc, KeyPair keys, Ciphertext ct, std::string msg, int length) {
    Plaintext pt;
    // std::cout << "length = "<< length<< std::endl;
    cc->Decrypt(keys.secretKey, ct, &pt);
    pt->SetLength(length);
    // std::cout << "DONE set length"<< std::endl;
    std::vector<double> v = pt->GetRealPackedValue();
    std::cout << msg << std::endl;
    RoundVector(v);
    PrintVector(v);
    std::cout << std::endl;
    // printf("HELLO WHAT IS WRONG!!!");
};

/*
Compute diagonals for the permutation matrix Sigma.
B[i,j] = A[i, i +j]
*/
std::vector<double> GenSigmaDiag(int32_t row_size, int32_t k) {
    int32_t n = row_size * row_size;
    std::vector<double> diag(n, 0);

    if (k >= 0) {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - row_size * k;
            if ((0 <= tmp) and (tmp < row_size - k)) {
                diag[i] = 1;
            }
        }
    } else {
        for (int32_t i = 0; i < n; i++) {
            int32_t tmp = i - (row_size + k) * row_size;
            if ((-k <= tmp) and (tmp < row_size)) {
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
// std::vector<double> GenTauDiag(const int32_t row_size, const int32_t k)
// {
//   int32_t n = row_size * row_size;
//   std::vector<double> diag(n, 0);

//   for (int32_t i = 0; i < row_size; i++)
//     diag[k + row_size * i] = 1;

//   return diag;
// }

std::vector<double> GenTauDiag(int32_t total_slots, int32_t row_size, int32_t k) {
    int32_t n = row_size * row_size;
    std::vector<double> diag(total_slots, 0);

    for (auto t = 0; t < total_slots / n; t++) {
        for (auto i = 0; i < row_size; i++) {
            diag[(t * n) + k + row_size * i] = 1;
        }
    }
    return diag;
}

/*
Compute diagonals for the permutation matrix Phi (V).
B[i,j] = A[i,j+1]
There are two diagonals in the matrix Phi.
Type = 0 correspond for the k-th diagonal, and type = 1 is for the (k-d)-th
diagonal
*/
std::vector<double> GenPhiDiag(int32_t row_size, int32_t k, int type) {
    int32_t n = row_size * row_size;
    std::vector<double> diag(n, 0);

    if (type == 0) {
        for (int32_t i = 0; i < n; i++)
            if ((i % row_size >= 0) and ((i % row_size) < row_size - k)) diag[i] = 1;
        return diag;
    }
    for (int32_t i = 0; i < n; i++)
        if ((i % row_size >= row_size - k) and (i % row_size < row_size)) {
            diag[i] = 1;
        }

    return diag;
}

/*
Compute diagonals for the permutation Psi (W).
B[i,j] = A[i+1,j]
*/
std::vector<double> GenPsiDiag(int32_t row_size, int32_t k) {
    int32_t n = row_size * row_size;
    std::vector<double> diag(n, 1);
    return diag;
}

std::vector<double> GenTransposeDiag(int32_t total_slots, int32_t row_size, int32_t i) {
    int32_t n = row_size * row_size;
    std::vector<double> diag(total_slots, 0);
    for (auto t = 0; t < total_slots / n; t++) {
        if (i >= 0) {
            for (auto j = 0; j < row_size - i; j++) {
                auto idx = t * n + (row_size + 1) * j + i;
                if (idx < total_slots) diag[idx] = 1;
            }
        } else {
            for (auto j = -i; j < row_size; j++) {
                auto idx = t * n + (row_size + 1) * j + i;
                if (idx < total_slots) diag[idx] = 1;
            }
        }
    }
    return diag;
}
