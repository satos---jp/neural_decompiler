import pickle

import random

def sample_dataset():
	with open('dataset_asm_ast/c_trans_array.pickle','rb') as fp:
		c_syntax_arr = pickle.load(fp)
	with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
		asm_vocab = pickle.load(fp)
	print('start')
	data = []
	for i in range(12):
		with open('dataset_asm_ast/data_%d.pickle' % i,'rb') as fp:
			nd = pickle.load(fp)[1]
		
		print(len(nd))
		for d in nd:
			if random.randint(1,12)==1:
				data.append(d)
		print('done',i)
		print('nl',len(data))
		del nd
	print(len(data))
	print('start saving')
	with open('sampled_asm_ast_dataset.picikle','wb') as fp:
		print('opened')
		pickle.dump((data,asm_vocab,c_syntax_arr),fp)
		print('dumped')
		
sample_dataset()
