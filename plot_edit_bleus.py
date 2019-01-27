from plot_edit_bleus_data import seq2seq,seq2seq_att,seq2tree
from matplotlib import pyplot as plt


def lap(i,v):
	return list(map(lambda x: x[i],v))

seq2seq = seq2seq[:21]

ts = ['edit distance','BLEU']
for t in range(2):
	plt.plot(lap(t,seq2seq),marker='o',label='Seq2seq')
	plt.plot(lap(t,seq2seq_att),marker='s',label='Seq2seqAtt')
	plt.plot(lap(t,seq2tree),marker='^',label='Seq2tree')
	plt.legend()
	plt.title(ts[t])
	plt.show()
