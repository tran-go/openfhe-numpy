#include "utils.h"

// utils implementation
usint NextPow2(const usint x)
{
  return pow(2, ceil(log(double(x)) / log(2.0)));
};

/*
Compute diagonals for a matrix
u[l] = (U[0,l], U[1,l+1],  )
*/
// vector<double>


/*
Compute diagonals for the permutation matrix Sigma.
B[i,j] = A[i, i +j]
*/
std::vector<double> GenSigmaDiag(const uint32_t row_size, const uint32_t k)
{
  uint32_t n = row_size * row_size;
  std::vector<double> diag(n, 0);

  if (k >= 0){
    for (uint32_t i = 0; i < n; i++){
      uint32_t tmp = i - row_size * k;
      if ((0 <= tmp) and (tmp < row_size - k)){
        diag[i] = 1;
      }}}
  else{
    for (uint32_t i = 0; i < n; i++){
      uint32_t tmp = i - (row_size + k ) * row_size;
      if ((-k <= tmp) and (tmp < row_size)){
        diag[i] = 1;
      }}}
    return diag;
}

 
/*
Compute diagonals for the permutation matrix Tau.
B[i,j] = A[i + j,i]
*/
std::vector<double> GenTauDiag(const uint32_t row_size, const uint32_t k)
{
  uint32_t n = row_size * row_size;
  std::vector<double> diag(n, 0);

  for (uint32_t i = 0; i < row_size; i++)
      diag[k + row_size * i] = 1;

  return diag;
}


/*
Compute diagonals for the permutation matrix Phi (V).
B[i,j] = A[i,j+1]
There are two diagonals in the matrix Phi. 
Type = 0 correspond for the k-th diagonal, and type = 1 is for the (k-d)-th diagonal 
*/
std::vector<double> GenPhiDiag(const uint32_t row_size, const uint32_t k, const int type )
{
    uint32_t n = row_size * row_size;
    std::vector<double> diag(n, 0);

    if (type == 0){
      for (uint32_t i = 0; i < n; i++)
        if ((i % row_size >= 0) and ((i % row_size) < row_size - k ))   
          diag[i] = 1;
      return diag;
    }
    for (uint32_t i = 0; i < n; i++)
      if  ((i % row_size >= row_size - k ) and (i % row_size < row_size)){
        diag[i] = 1;
      }
        
    return diag;
}


/*
Compute diagonals for the permutation Psi (W).
B[i,j] = A[i+1,j]
*/
std::vector<double> GenPsiDiag(const uint32_t row_size, const uint32_t k)
{
  uint32_t n = row_size * row_size;
  std::vector<double> diag(n, 1);
  return diag;
}
