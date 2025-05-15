#ifndef HELPER_H
#define HELPER_H

#include "../config.h"

// Internal helper functions - NOT part of the public API
// Do not use directly in client code

/*
Displaying the 2D vector
*/
template <typename Element>
inline void PrintMatrix(const std::vector<std::vector<Element>>& mat) {
    for (uint32_t i = 0; i < mat.size(); i++) {
        for (uint32_t j = 0; j < mat[i].size(); j++)
            std::cout << mat[i][j] << " ";
        std::cout << std::endl;
    }
};

/*
Displaying the 1D vector
*/

template <typename Element>
inline void PrintVector(const std::vector<Element> vec) {
    std::cout.precision(2);
    for (uint32_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl << std::endl;
};

/*
Normal Matrix-Matrix Product
*/

inline std::vector<std::vector<double>> MulMats(std::vector<std::vector<double>>& A,
                                                std::vector<std::vector<double>>& B) {
    uint32_t nA = A.size();
    uint32_t mA = A[0].size();

    uint32_t nB = B.size();
    uint32_t mB = B[0].size();

    std::vector<std::vector<double>> result(nA, std::vector<double>(mB, 0));

    try {
        if (mA == nB) {
            for (uint32_t i = 0; i < nA; i++) {
                for (uint32_t j = 0; j < mB; j++) {
                    for (uint32_t k = 0; k < mA; k++) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
        else {
            throw("Error:: size problems!!!");
        }
    }
    catch (...) {
        OPENFHE_THROW("ERROR ::: Normal Matrix Multiplication ::: size is different");
    }
    return result;
};

/*
Normal Matrix Vector Multiplication
*/

inline std::vector<double> MulMatVec(std::vector<std::vector<double>> mat, std::vector<double> vec) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();
    uint32_t k = vec.size();

    std::vector<double> result(m, 0);

    try {
        if (m == k) {
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t j = 0; j < m; j++) {
                    result[i] += mat[i][j] * vec[j];
                }
            }
        }
        else {
            throw("Error:: size problems!!!");
        }
    }
    catch (...) {
        OPENFHE_THROW("ERROR ::: Normal Matrix Multiplication ::: size is different");
    }
    return result;
};

/*
Sample a rational random vector
*/

inline std::vector<double> RandVec(const int n, const int modulus = 5, const bool verbose = true) {
    std::vector<double> vec(n, 0);
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < n; i++)
        vec[i] = rand() % modulus;

    // Print the generated vector
    if (verbose) {
        std::cout << "Random Vector:" << std::endl;
        for (const auto& element : vec)
            std::cout << element << " ";
    }
    return vec;
};

/*
Sample a rational random matrix
*/

inline std::vector<std::vector<double>> RandMatrix(const int nrows,
                                                   const int numCols,
                                                   const double min_val = 0,
                                                   const double max_val = 10,
                                                   const bool verbose   = true) {
    std::vector<std::vector<double>> matrix(nrows, std::vector<double>(numCols));
    std::srand(static_cast<unsigned>(std::time(0)));

    // Fill the matrix with random numbers in the range [minVal, maxVal]
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix[i][j] = min_val + std::fmod(std::rand(), (max_val - min_val + 1));
        }
    }

    // Print the generated matrix
    if (verbose) {
        std::cout << "Random Matrix:" << std::endl;
        for (const auto& row : matrix) {
            for (const auto& element : row) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }
    }
    return matrix;
};

template <class Element>
std::vector<Element> EncodeMatrix(std::vector<std::vector<Element>> mat, const long total_slots) {
    uint32_t n = mat.size();
    uint32_t m = mat[0].size();

    uint32_t size   = n * m;
    uint32_t blocks = total_slots / size;

    std::vector<Element> vec(total_slots, 0);
    long k = 0;
    for (uint32_t t = 0; t < blocks; ++t) {
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < m; ++j) {
                vec[k] = mat[i][j];
                k += 1;
            }
        }
    }
    return vec;
}

// template <class Element>
// std::vector<Element> EncodeVector(std::vector<std::vector<Element>> mat, const long total_slots) {
//     uint32_t n = mat.size();
//     uint32_t m = mat[0].size();

//     uint32_t size   = n * m;
//     uint32_t blocks = total_slots / size;

//     std::vector<Element> vec(total_slots, 0);
//     long k = 0;
//     for (uint32_t t = 0; t < blocks; ++t) {
//         for (uint32_t i = 0; i < n; ++i) {
//             for (uint32_t j = 0; j < m; ++j) {
//                 vec[k] = mat[i][j];
//                 k += 1;
//             }
//         }
//     }
//     return vec;
// }


#endif  // HELPER_H
