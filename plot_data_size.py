import pickle

tds = []
tdsi = 0
'''
def csize(d):
	res = 1
	_,_,cs = d
	for c in cs:
		res += csize(c)
	return res

lens = {i: 0 for i in range(2000)}
for i in range(13):
	print(i)
	with open('dataset_asm_ast/data_%d.pickle' % i,'rb') as fp:
		cs = pickle.load(fp)
		pass
	print('loaded')
	print(len(cs[1]))
	#continue
	for d in cs[1]:
		#print(d)
		#if lens[d[1]] >= 1000:
		#	continue
		lens[csize(d[2])] += 1
		#tds.append(d)
		"""
		if len(tds)>=100000:
			with open('dataset_asm_ast/data_cutoff_%d.pickle' % tdsi,'wb') as fp:
				print(len(tds))
				pickle.dump(tds,fp)
			tds = []
			tdsi += 1
		"""
	print('calced')

#with open('dataset_asm_ast/data_cutoff_%d.pickle' % tdsi,'wb') as fp:
#	print(len(tds))
#	pickle.dump(tds,fp)

#print(tdsi)
print(lens)
exit()
'''

csrc_lens = {0: 0, 1: 0, 2: 1652760, 3: 3545865, 4: 1981035, 5: 1615788, 6: 1320921, 7: 1039905, 8: 661185, 9: 595962, 10: 649809, 11: 418311, 12: 342486, 13: 285264, 14: 388629, 15: 293904, 16: 218043, 17: 223596, 18: 110493, 19: 178794, 20: 89109, 21: 124344, 22: 102294, 23: 115794, 24: 82278, 25: 112086, 26: 62838, 27: 75258, 28: 69777, 29: 87633, 30: 56466, 31: 73809, 32: 74691, 33: 44496, 34: 84852, 35: 72459, 36: 31932, 37: 50733, 38: 62739, 39: 35154, 40: 43632, 41: 28260, 42: 24885, 43: 21951, 44: 21825, 45: 22536, 46: 19134, 47: 17352, 48: 12312, 49: 23715, 50: 13851, 51: 28827, 52: 17100, 53: 16128, 54: 16758, 55: 15084, 56: 17973, 57: 16992, 58: 12924, 59: 12285, 60: 8973, 61: 16542, 62: 9855, 63: 17478, 64: 10836, 65: 13131, 66: 12600, 67: 13212, 68: 8100, 69: 13500, 70: 9090, 71: 11187, 72: 10107, 73: 9864, 74: 8973, 75: 7002, 76: 7578, 77: 13131, 78: 5274, 79: 6561, 80: 5382, 81: 6183, 82: 4869, 83: 5445, 84: 5859, 85: 5139, 86: 7929, 87: 4977, 88: 6363, 89: 7407, 90: 4374, 91: 4680, 92: 3708, 93: 4986, 94: 3159, 95: 4914, 96: 5751, 97: 4626, 98: 6507, 99: 5778, 100: 2997, 101: 3411, 102: 2961, 103: 3285, 104: 3240, 105: 3933, 106: 3186, 107: 3168, 108: 3132, 109: 4320, 110: 2943, 111: 3069, 112: 2826, 113: 3069, 114: 2934, 115: 4617, 116: 4437, 117: 2385, 118: 2898, 119: 2988, 120: 2178, 121: 3447, 122: 2682, 123: 3060, 124: 2592, 125: 2421, 126: 2493, 127: 2790, 128: 2259, 129: 3159, 130: 3051, 131: 2376, 132: 2412, 133: 2592, 134: 2277, 135: 2538, 136: 2502, 137: 2142, 138: 1836, 139: 9684, 140: 1908, 141: 2628, 142: 2493, 143: 2520, 144: 2061, 145: 2367, 146: 2034, 147: 2466, 148: 1890, 149: 3582, 150: 2439, 151: 3024, 152: 1557, 153: 1710, 154: 2214, 155: 2664, 156: 1737, 157: 1503, 158: 1017, 159: 1467, 160: 1332, 161: 2196, 162: 1467, 163: 1755, 164: 2277, 165: 1791, 166: 1647, 167: 1998, 168: 1458, 169: 1647, 170: 1305, 171: 1917, 172: 1503, 173: 1386, 174: 756, 175: 1278, 176: 972, 177: 990, 178: 810, 179: 1422, 180: 2025, 181: 1008, 182: 1566, 183: 1890, 184: 1071, 185: 1089, 186: 882, 187: 3591, 188: 3870, 189: 1620, 190: 873, 191: 2412, 192: 792, 193: 1341, 194: 981, 195: 4095, 196: 1188, 197: 855, 198: 1539, 199: 1161, 200: 1701, 201: 3699, 202: 0, 203: 0, 204: 0, 205: 0, 206: 0, 207: 0, 208: 0, 209: 0}


from plot_data_size_data import ast_lens


from matplotlib import pyplot as plt
lens = ast_lens
lens = [lens[i] for i in range(0,1000)]
plt.yscale('log')
plt.ylabel('number of flagments',fontsize=15)
plt.xlabel('AST size of C flagment',fontsize=15)
plt.plot(lens)
#plt.show()
plt.savefig('texs/ast_lens.png')
plt.clf()

lens = csrc_lens
lens = [lens[i] for i in range(0,210)]
plt.yscale('log')
plt.xlabel('token length of C flagment',fontsize=15)
plt.ylabel('number of flagments',fontsize=15)
plt.plot(lens)
#plt.show()
plt.savefig('texs/c_lens.png')
