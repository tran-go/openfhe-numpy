import random
# import pytest
from openfhe import *
from openfhe_matrix import *
from helper import *


def main():
  print("Hello")
  # precision = 2
  mult_depth = 9
  firstModSize = 59
  # dcrtBits = 59
  row_size = 4
  n_rows, n_cols = 4, 4
  
  
  parameters = CCParamsCKKSRNS()
  parameters.SetMultiplicativeDepth(mult_depth)
  parameters.SetScalingModSize(firstModSize)
  # parameters.SetBatchSize(batch_size)

  print("111111111111111111111")
  cc = GenCryptoContext(parameters)
  cc.Enable(PKESchemeFeature.PKE)
  cc.Enable(PKESchemeFeature.KEYSWITCH)
  cc.Enable(PKESchemeFeature.LEVELEDSHE)

  # parameters = CCParamsCKKSRNS()
  # parameters.SetMultiplicativeDepth(mult_depth)
  # parameters.SetScalingModSize(firstModSize)
      

  # cc = GenCryptoContext(parameters)
  # print("2222222222222")
  # cc.Enable(PKESchemeFeature.PKE)
  # print("333333333333333333333")
  # cc.Enable(PKESchemeFeature.KEYSWITCH)
  # print("444444444444444444")
  # cc.Enable(PKESchemeFeature.LEVELEDSHE)
  # print("22222222222222222")

  ring_dims = cc.GetRingDimension()
  num_slots = ring_dims // 2

  print("The CKKS scheme is using ring dimension: " 
          + str(cc.GetRingDimension()))

  keys = cc.KeyGen()   
  A = np.array([[1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0], [4, 4, 4, 0]])
  B = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [3, 0, 3, 0], [3, 0, 2, 0]])
    
  vA = SMat._pack_row_wise(A,4,16)
  vB = SMat._pack_row_wise(B,4,16)

  pA = cc.MakeCKKSPackedPlaintext(vA);
  pB = cc.MakeCKKSPackedPlaintext(vB);

  cA = cc.Encrypt(keys.publicKey, pA);
  cB = cc.Encrypt(keys.publicKey, pB);

  print('matrix product: \n')
  SMat._print_mat(A@B,4)

  ct_AB = EvalMatMulSquare(cc, keys, cA, cB, row_size);
  result = cc.Decrypt(ct_AB, keys.secretKey)
  result.SetLength(16)
  print("result = " + result.GetFormattedValues(precision))


if __name__ == "__main__":
    main()

