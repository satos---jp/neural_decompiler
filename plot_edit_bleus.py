from plot_edit_bleus_data import seq2seq,seq2seq_att,seq2tree_flatten,seq2tree
from matplotlib import pyplot as plt


def lap(i,v):
	return list(map(lambda x: x[i],v))

seq2seq = seq2seq[:21]

def xlas(x):
	x = x.split('e')
	return x[0] + '*10^' + x[1][-1]

ts = ['edit distance ratio','BLEU score']
ys = ['edit distance ratio','BLEU score']
sn = ['edit_dist.png','bleu.png']
for t in range(2):
	plt.plot(lap(t,seq2seq),marker='o',label='Seq2seq')
	plt.plot(lap(t,seq2seq_att),marker='s',label='Seq2seqAtt')
	plt.plot(lap(t,seq2tree_flatten),marker='^',label='Seq2tree')
	#plt.plot(lap(t,seq2tree),marker='*',label='Seq2tree_my')
	plt.legend(fontsize=15)
	#plt.title(ts[t])
	plt.xlabel('training data size',fontsize=15)
	plt.xticks([(i*1000.0/1024) for i in range(21)][::5],[xlas('%0.1e' % (100 * 1000 * i)) for i in range(21)][::5])
	plt.ylabel(ys[t],fontsize=15)
	plt.show()
	continue
	plt.savefig('texs/' + sn[t])
	plt.clf()
