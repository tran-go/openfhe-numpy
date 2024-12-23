from openfhe import *
from matrix import *



def main():
    precision = 2
    mult_depth = 4
    firstModSize = 60
    dcrtBits = 59
    block_size = 4


    parameters = CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(mult_depth)
    parameters.SetScalingModSize(firstModSize)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    ring_dims = cc.GetRingDimension()
    num_slots = ring_dims // 2

    print("The CKKS scheme is using ring dimension: " + str(cc.GetRingDimension()))

    keys = cc.KeyGen()
    cc.EvalSumKeyGen(keys.secretKey)
    sum_row_keys = cc.EvalSumRowsKeyGen(keys.secretKey, None, block_size)



    matrix = [[1,2,3], [4,5,6]]
    vector = [1,2,3]

    e_mat = EMat(cc, keys.publicKey, matrix, num_slots, block_size, Pack_Type.RW)
    e_vec = EMat(cc, keys.publicKey, vector, num_slots, block_size, Pack_Type.CW)

    ct_mul = mul_mat(cc, sum_row_keys, e_mat, e_vec, num_slots, block_size)



    result = cc.Decrypt(ct_mul, keys.secretKey)
    result.SetLength(batch_size)
    print("result = " + result.GetFormattedValues(precision))
        
    real_result = np.array(matrix) @ np.array(vector)
    print(real_result) 


if __name__ == "__main__":
    main()
