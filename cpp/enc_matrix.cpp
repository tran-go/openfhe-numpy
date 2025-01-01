#include "enc_matrix.h"

void test()
{
  std::cout << "Hello Boca Raton!!!\n";
}

/*
Compute ct_matrix * ct_vector
*/
CT EvalMultMatVec(
  CC& context,
  KeyPair keys,
  MatKeys eval_keys,
  int type,
  uint32_t row_size,
  const CT& ct_vec,
  const CT& ct_mat)
{
  CT ct_prod;
  auto ct_mult = context->EvalMult(ct_mat, ct_vec);

#ifdef DEBUG
  std::cout << "Debugging enabled" << std::endl;
  PT pt_mult;
  context->Decrypt(keys.secretKey, ct_mult, &pt_mult);
  pt_mult->SetLength(row_size * 4);
  std::vector<double> vec = pt_mult->GetRealPackedValue();
  PrintVector(vec);
#endif

  if (type == encode_style::MM_CRC)
  {
    std::cout << "Using encode_style = " << encode_style::MM_CRC
      << " with row_size = " << row_size << std::endl;
    ct_prod = context->EvalSumCols(ct_mult, row_size, *eval_keys);

#ifdef DEBUG
    PT pt_tmp;
    context->Decrypt(keys.secretKey, ct_prod, &pt_tmp);
    pt_tmp->SetLength(row_size * 4);
    std::vector<double> vec = pt_tmp->GetRealPackedValue();
    PrintVector(vec);
#endif
  }
  else if (type == encode_style::MM_RCR)
  {
    std::cout << "Using encode_style = " << encode_style::MM_RCR
      << " with row_size = " << row_size << std::endl;

    ct_prod = context->EvalSumRows(ct_mult, row_size, *eval_keys);
  }
  else
  {
    printf("To be continue ... \n");
  }
  return ct_prod;
};

/*
Generate the diagonals for the W permutation
*/
CT EvalLinTransShift(CC cc,
  KeyPair keys,
  const CT c_vec,
  const int opt,
  const uint32_t row_size)
{
  uint32_t n = row_size * row_size;
  CT c_result, c_tmp;
  std::vector<double> result(n, 0.0);
  PT p_result = cc->MakeCKKSPackedPlaintext(result);
  c_result = cc->Encrypt(keys.publicKey, p_result);

  if (opt == perm_style::SIGMA)
  {
    for (uint32_t k = -row_size; k < row_size; k++)
    {
      auto diag = GenSigmaDiag(row_size, k);
      auto p_diag = cc->MakeCKKSPackedPlaintext(diag);

      cc->EvalRotateKeyGen(keys.secretKey, { (int)k });
      auto c_rotated = cc->EvalRotate(c_vec, k);
      cc->EvalAddInPlace(c_result, cc->EvalMult(c_rotated, p_diag));
    }
  }
  else if (opt == perm_style::TAU)
  {
    for (uint32_t k = 0; k < row_size; k++)
    {
      auto diag = GenTauDiag(row_size, row_size * k);
      auto p_diag = cc->MakeCKKSPackedPlaintext(diag);

      cc->EvalRotateKeyGen(keys.secretKey, { (int)k });
      auto c_rotated = cc->EvalRotate(c_vec, row_size * k);
      cc->EvalAddInPlace(c_result, cc->EvalMult(c_rotated, p_diag));
    }
  }
  else {
    throw("Wrong !!!");
  }
  return c_result;
};

CT EvalLinTransPhi(CC cc,
  KeyPair keys,
  const CT c_vec,
  const uint32_t row_size,
  const uint32_t k)
{
  uint32_t n = row_size * row_size;
  CT c_result, c_tmp;
  std::vector<double> result(n, 0.0);
  PT p_result = cc->MakeCKKSPackedPlaintext(result);
  c_result = cc->Encrypt(keys.publicKey, p_result);
  for (uint32_t i = 0; i < 2; i++)
  {
    uint32_t idx = k + i * row_size;
    auto diag = GenPhiDiag(row_size, k, i);
    PT p_diag = cc->MakeCKKSPackedPlaintext(diag);
    cc->EvalRotateKeyGen(keys.secretKey, { (int)idx });
    CT c_rotated = cc->EvalRotate(c_vec, idx);
    cc->EvalAddInPlace(c_result, cc->EvalMult(c_rotated, p_diag));
  }
  return c_result;
};


CT EvalLinTransPsi(CC cc,
  KeyPair keys,
  const CT c_vec,
  const uint32_t row_size,
  const uint32_t k)
{
  cc->EvalRotateKeyGen(keys.secretKey, { (int)row_size * (int)k });
  CT c_result = cc->EvalRotate(c_vec, row_size * k);
  return c_result;
}

void Debug(CC cc, KeyPair keys, CT ct, std::string msg, int length = 16)
{
  PT pt;
  cc->Decrypt(keys.secretKey, ct, &pt);
  pt->SetLength(length);
  std::vector<double> v = pt->GetRealPackedValue();
  std::cout<<msg<<std::endl;
  PrintVector(v);
}

CT EvalMatMulSquare(CC cc,
  KeyPair keys,
  const CT cmat_A,
  const CT cmat_B,
  const uint32_t row_size)
{

  // Step 1 - 1
  CT cA = EvalLinTransShift(cc, keys, cmat_A, perm_style::SIGMA, row_size);
  Debug(cc, keys, cA, "cA");
  // Step 1 - 2
  CT cB = EvalLinTransShift(cc, keys, cmat_B, perm_style::TAU, row_size);
  Debug(cc, keys, cB, "cB");
  CT cAB = cc->EvalMult(cA, cB);
  Debug(cc, keys, cB, "cAB");
  // Step 2 and 3
  for (uint32_t k = 1; k < row_size; k++) {
    std::cout<<"Loop: "<<k<<std::endl;
    cA = EvalLinTransPhi(cc, keys, cA, row_size, k);
    Debug(cc, keys, cA, "cA");
    cB = EvalLinTransPsi(cc, keys, cB, row_size, k);
    Debug(cc, keys, cB, "cB");
    CT tmp = cc->EvalMult(cA, cB);
    cc->EvalAddInPlace(cAB, tmp);
  }
   std::cout<<"Finish MM "<<std::endl;
  return cAB;
}