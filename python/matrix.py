from openfhe import *
from helper import *

# vector v = [1,2,3] is a column vector in this context. 

class EMat(object):

	def __init__(self, cc, public_key, data, num_slots, block_size = None, type = None):
		
		self.context = context
		self.data = data
		self.type = type
		self.num_slots = num_slots

		matrix = np.array(data)
		self.dims  = matrix.shape

		if  len(dims) == 1:
			self.is_matrix = 0
		else: 
			self.is_matrix = 1


		if type == None: 
			if is_matrix == 1: 
				type = Pack_Type.RW
			else: 
				type = Pack_Type.CW



		if block_size == None: 
			if self.is_matrix:
				self.row_size = next_power2 (self.dims[1]) 
				self.col_size = num_slots // self.row_size
			else:
				self.row_size = 1
				self.col_size = next_power2 (self.dims[0])
		else: 
			self.row_size = block_size
			self.col_size = num_slots // self.row_size

		if type == Pack_Type.RW:
			if self.is_matrix:
				packed = MatrixHelper._pack_row_wise(data, self.row_size, num_slots)
			else: 
				packed = VectorHelper._pack_row_wise(data, self.row_size, num_slots) 	
			pt = self.context.MakeCKKSPackedPlaintext(packed)
			
		elif type ==  Pack_Type.CW:  
			if self.is_matrix:
				packed = MatrixHelper._pack_col_wise(data, self.col_size , num_slots)
			else: 
				packed = VectorHelper._pack_col_wise(data, self.col_size, num_slots) 	
							
			pt = self.context.MakeCKKSPackedPlaintext(packed)
		else:
			 raise Exception(f'Wrong type [{type}]')

		self.ct_data = context.Encrypt(public_key, pt)


	
	def decrypt(self, secret_key) -> List[float]:
		pt =  self.context.Decrypt(secret_key, self.ct_data)
		return pt.GetRealPackedValue()



	@staticmethod
	def mul_mat(cc, key_sum, ct_matrix, ct_vec, num_slots, block_size = None):
		
		if ct_matrix.type == CW and ct_vec.type == RW:
			# type == RCR using key_eval_sum_rows
			ct_mult = cc.EvalMult(ct_matrix, ct_vec)
		    ct_product = cc.EvalSumRows(ct_mult, row_size, key_sum, subring_dim)

		elif ct_matrix.type == RW and ct_vec.type == CW:
			# type == CRC using key_eval_sum_cols
			ct_mult = cc.EvalMult(ct_matrix, ct_vec)
		    ct_product = cc.EvalSumCols(ct_mult, row_size, key_sum)
		
		return ct_product
