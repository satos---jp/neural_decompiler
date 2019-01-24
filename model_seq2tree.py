import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer import cuda
from cuda_setting import isgpu
	
if isgpu:
	xp = cuda.cupy
else:
	xp = np

def sequence_embed(embed, xs):
	x_len = [len(x) for x in xs]
	x_section = np.cumsum(x_len[:-1])
	ex = embed(F.concat(xs, axis=0))
	exs = F.split_axis(ex, x_section, 0)
	return exs


from model_attention import GlobalGeneralAttention



class Seq2Tree(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, trans_data, n_units,v_eos_src,n_maxsize):
		super(Seq2Tree, self).__init__()

		# for each nodetype, for each move, the result array.
		self.trans_data = trans_data
		self.embed_idx = []
		ns = 0
		def inc():
			nonlocal ns
			ns += 1
			return ns-1
		self.embed_idx = [[[inc() for v in vs] for vs in moves] for moves in self.trans_data]
		self.embed_root_idx = ns
		self.embed_y_size = ns+1
		
		#self.choice_num_sum = sum(list(map(lambda d: len(d),self.trans_data)))
		
		self.choicerange = []
		self.choice_idx = []
		s = 0
		for d in self.trans_data:
			b = s
			s += len(d)
			self.choicerange.append((b,s))
			self.choice_idx.append(list(range(b,s)))
		
		self.type_size = len(self.embed_idx)
		self.n_all_choice = sum(map(lambda x: len(x),self.trans_data))
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
						
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units, n_units*2, 0.1)
			self.Wc = L.Linear(n_units*4, n_units)
			self.Ws = L.Linear(n_units, self.n_all_choice)
						
			#self.att = Attention(n_units)
			self.att = GlobalGeneralAttention(n_units)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.n_maxsize = n_maxsize
		self.rootidx = len(trans_data)-1
		
	def forward(self, xs, ys):
		batch = len(xs)
		xs = [xp.array(x[::-1]) for x in xs]
		exs = sequence_embed(self.embed_x, xs)
		# None represents a zero vector in an encoder.
		hx, cx, xs_states = self.encoder(None, None, exs)
		hx = F.reshape(F.transpose(hx,(1,0,2)),(batch,self.n_layers,self.n_units*2))
		cx = F.reshape(F.transpose(cx,(1,0,2)),(batch,self.n_layers,self.n_units*2))
		hx = [d for d in hx]
		cx = [d for d in cx]
		
		evs = [self.embed_y(xp.array([i])) for i in range(self.embed_y_size)]
		
		concat_ys_outs = [[] for _ in ys]
		#print(len(ys))

		att_os = []	
		for i,(y,hxs) in enumerate(zip(ys,xs_states)):
			concat_oss = []
			def rec_LSTM(node,eidx,nhxncx):
				nonlocal i
				(nhx,ncx) = nhxncx
				ntype,nchoice,children = node
				#eidx = self.embed_idx[ntype][ppos]
				ev = evs[eidx]
				#print(ev.shape)
				#print(nhx.shape,ncx.shape)
				thx,tcx,nos = self.decoder(nhx,ncx,[ev])
				nos = nos[0]
				#wnos = self.W[ntype](nos)
				if len(self.trans_data[ntype])>1:
					concat_oss.append(nos)
					concat_ys_outs[i].append(self.choice_idx[ntype][nchoice])
				# otherwise, we don't have to train.
				
				for j,ch in enumerate(children):
					#print(ntype,nchoice,i)
					teidx = self.embed_idx[ntype][nchoice][j]
					rec_LSTM(ch,teidx,(thx,tcx))
			
			nhx,ncx = hx[i],cx[i]
			ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
			nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
			
			assert y[0] == self.type_size-1
			ridx = self.embed_root_idx
			rec_LSTM(y,ridx,(nhx,ncx))
			
			#print(concat_oss[0].shape,len(concat_oss))
			yh = F.concat(concat_oss,axis=0)
			#print(yh.shape)
			ch = self.att(xs_states[i],yh)
			att_os.append(F.tanh(self.Wc(F.concat([ch,yh],axis=1))))
		
		concat_os = F.concat(att_os, axis=0)
		concat_ys_out = list(map(lambda d: xp.array(d),concat_ys_outs))
		concat_ys_out = F.concat(concat_ys_out, axis=0)
		loss = F.sum(F.softmax_cross_entropy(
			self.Ws(concat_os), concat_ys_out, reduce='no')) / batch
		
		chainer.report({'loss': loss}, self)
		#print(loss)
		return loss
	
	def translate(self, xs):
		EOS_DST = self.v_eos_dst
		batch = len(xs)
		
		#beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_states = self.encoder(None, None, exs)
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			ivs = [self.embed_y(self.xp.full(1,i,np.int32))[0] for i in range(self.embed_y_size)]
			v = ivs[EOS_DST]
			
			result = []

			
			nsize = None
			def expand_tree(ntype,eidx,nhxncx):
				nonlocal nsize
				(nhx,ncx) = nhxncx
				if nsize > self.n_maxsize:
					return (ntype,-1,[])
				nsize += 1
				#eidx = self.embed_idx[ntype][ppos]
				ev = self.embed_y(np.array([eidx]))[0]
				tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
				thx,tcx,nos = self.decoder(nhx,ncx,[ev])
				nos = nos[0]
				
				if len(self.trans_data[ntype])>1:
					wy = self.W[ntype](nos)
					#wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.vs_target_vocab[ntype])),axis=1),(self.vs_target_vocab[ntype],)).data
					nchoice = F.argmax(wy.data).data.astype(np.int32).item()
				else:
					nchoice = 0
				
				#print(ntype,nchoice)
				#print(c_cfg.idx2nodetype(ntype))
				ctypes = self.trans_data[ntype][nchoice]
				resv = []
				for i,ct in enumerate(ctypes):
					teidx = self.embed_idx[ntype][nchoice][i]
					resv.append(expand_tree(ct,teidx,(thx,tcx),ndepth+1))
				return (ntype,nchoice,resv)
			
			for i,hxs in enumerate(xs_states):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
		
				self.att.init_hxs(hxs)
				ridx = self.embed_root_idx
				
				# TODO(satos) What is the beam search for tree!?
				# now, beam search is ommited.
				nsize = 0
				tree = expand_tree(self.v_eos_dst,ridx,(nhx,ncx))
				self.att.reset()
				result.append(tree)
			
		return result
