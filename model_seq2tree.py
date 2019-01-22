import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers
import c_cfg
	
def sequence_embed(embed, xs):
	x_len = [len(x) for x in xs]
	x_section = np.cumsum(x_len[:-1])
	ex = embed(F.concat(xs, axis=0))
	exs = F.split_axis(ex, x_section, 0)
	return exs


from model_attention import Attention,DotAttention,LocalDotAttention



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
		
		self.type_size = len(self.embed_idx)
		
		self.n_max_choice = max(map(lambda x: len(x),self.trans_data))
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			#self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			self.embed_y_0 = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			self.embed_y_1 = L.EmbedID(self.type_size, n_units) # maybe mergable
			self.W = [None for _ in self.trans_data]
			for i,moves in enumerate(self.trans_data):
				if len(moves)>1:
					self.W[i] = L.Linear(n_units*2, len(moves))
			
			self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units*2, n_units, 0.1)
			#self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.W_flatten = L.Linear(n_units, self.n_max_choice)
			
			#self.att = Attention(n_units)
			self.att = DotAttention(n_units)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.n_maxsize = n_maxsize
		self.v_eos_dst = c_cfg.rootidx
		
		self.n_maxlen = 800
	
	#@profile
	def forward_tree(self, xs, ys):
		#print(ys)
		#exit()
		#xs = [x[::-1] for x in xs]
		
		#eos_dst = self.xp.array([self.v_eos_dst], np.int32)
		#ys_in = [F.concat([eos_dst, y], axis=0) for y in ys]
		#ys_out = [F.concat([y, eos_dst], axis=0) for y in ys]

		# Both xs and ys_in are lists of arrays.
		exs = sequence_embed(self.embed_x, xs)
		#eys = sequence_embed(self.embed_y, ys_in)
		#print(list(map(lambda x: len(x),exs)))
		#print(list(map(lambda x: len(x),eys)))
		#exit()
		print('embedded')
		batch = len(xs)
		# None represents a zero vector in an encoder.
		hx, cx, xs_states = self.encoder(None, None, exs)
		#_, _, os = self.decoder(hx, cx, eys)
		
		#return chainer.Variable(np.array([0],dtype=np.float32))
		
		hx = F.transpose(hx,axes=(1,0,2))
		cx = F.transpose(cx,axes=(1,0,2))
		#print(batch,hx.shape,cx.shape,(batch,xs_states[0].shape))
		
		concat_oss = [[] for _ in self.trans_data]
		concat_ys_outs = [[] for _ in self.trans_data]
		#@profile
		def rec_LSTM(node,eidx,nhxncx):
			(nhx,ncx) = nhxncx
			ntype,nchoice,children = node
			#eidx = self.embed_idx[ntype][ppos]
			ev = self.embed_y(np.array([eidx]))[0]
			#ctx = self.att(nhx)
			#tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
			tv = F.reshape(ev,(1,self.n_units))
			thx,tcx,nos = self.decoder(nhx,ncx,[tv])
			nos = nos[0]
			#wnos = self.W[ntype](nos)
			if len(self.trans_data[ntype])>1:
				#print(nos.shape,nhx.shape)
				ctx = self.att(nos)
				tos = F.reshape(F.tanh(F.concat([ctx,ev],axis=0)),(1,self.n_units*2))
				#print(tos.shape)
				concat_oss[ntype].append(tos)
				concat_ys_outs[ntype].append(nchoice)
			# otherwise, we don't have to train.
			for i,ch in enumerate(children):
				#print(ntype,nchoice,i)
				teidx = self.embed_idx[ntype][nchoice][i]
				rec_LSTM(ch,teidx,(thx,tcx))
		
		print('start_rec')
		for i,(y,hxs) in enumerate(zip(ys,xs_states)):
			nhx,ncx = hx[i],cx[i]
			ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
			nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
			
			assert y[0] == self.v_eos_dst
			self.att.init_hxs(hxs)
			ridx = self.embed_root_idx
			rec_LSTM(y,ridx,(nhx,ncx))
			self.att.reset()
		
		#loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
		
		loss_s = []
		for i,(concat_os,concat_ys_out) in enumerate(zip(concat_oss,concat_ys_outs)):
			assert len(concat_os) == len(concat_ys_out)
			if len(concat_os)==0:
				continue
			concat_os = F.concat(concat_os,axis=0)
			#print(concat_os.shape)
			#concat_os = np.array(concat_os)
			#print(concat_os)
			#print(concat_ys_out)
			d = F.softmax_cross_entropy(self.W[i](concat_os), np.array(concat_ys_out), reduce='no')
			loss_s.append(d)
		
		#print(loss_s)
		
		loss = F.sum(F.concat(loss_s,axis=0))/ batch
		
		chainer.report({'loss': loss}, self)
		#n_words = concat_ys_out.shape[0]
		#perp = self.xp.exp(loss.array * batch / n_words)
		#chainer.report({'perp': perp}, self)
		return loss


	#@profile
	def forward(self, xs, ys):
		#print(ys)
		#exit()
		#xs = [x[::-1] for x in xs]
		
		
		def flatten(y,pa=self.embed_root_idx):
			ty,ch,cs = y
			res = [((pa,ty),ch)]
			for i,c in enumerate(cs):
				eidx = self.embed_idx[ty][ch][i]
				res += flatten(c,eidx)
			return res
		
		#assert all(map(lambda x: unflatten(flatten(x),c_cfg.rootidx)[0]==x,ys))
		
		ys = list(map(flatten,ys))
		ys_in_0  = [np.array(list(map(lambda ab: ab[0][0],y))) for y in ys]
		ys_in_1  = [np.array(list(map(lambda ab: ab[0][1],y))) for y in ys]
		#ys_out = [np.array(list(map(lambda ab: self.choice_idx[ab[0][1]][ab[1]],y))) for y in ys]
		ys_out = [np.array(list(map(lambda ab: ab[1],y))) for y in ys]
		
		# Both xs and ys_in are lists of arrays.
		exs = sequence_embed(self.embed_x, xs)
		eys_0 = sequence_embed(self.embed_y_0, ys_in_0)
		eys_1 = sequence_embed(self.embed_y_1, ys_in_1)
		#print(eys_0[0].shape,eys_1[0].shape)
		eys = [F.concat([y0,y1],axis=1) for (y0,y1) in zip(eys_0,eys_1)]
		#print(eys[0].shape)
		#print('embedded')
		batch = len(xs)
		# None represents a zero vector in an encoder.
		hx, cx, xs_states = self.encoder(None, None, exs)
		#print('encode')
		_, _, os = self.decoder(hx, cx, eys)
		#print('decode')
		
		ctxs = [self.att(xh,yh) for (xh,yh) in zip(xs_states,os)]
		#print('attentioned')
		att_os = [F.tanh(self.Wc(F.concat([ch,yh]))) for (ch,yh) in zip(ctxs,os)]
		
		concat_os = F.concat(att_os, axis=0)
		concat_ys_out = F.concat(ys_out, axis=0)
		loss = F.sum(F.softmax_cross_entropy(
			self.W_flatten(concat_os), concat_ys_out, reduce='no')) / batch
		#print('lossed')
		chainer.report({'loss': loss}, self)
		#n_words = concat_ys_out.shape[0]
		#perp = self.xp.exp(loss.array * batch / n_words)
		#chainer.report({'perp': perp}, self)
		return loss
		
	def translate_tree(self, xs):
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
				ctx = self.att(nhx)
				tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
				thx,tcx,nos = self.decoder(nhx,ncx,[tv])
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
	


	def translate(self, xs):
		EOS_DST = self.v_eos_dst
		batch = len(xs)
		
		def unflatten(v,ty=c_cfg.rootidx):
			#print(v,ty)
			if len(v)==0:
				return (ty,-1,[]),[]
			ch = v[0]
			v = v[1:]
			cts = self.trans_data[ty][ch]
			#print(cts)
			rcs = []
			for ct in cts:
				ac,v = unflatten(v,ct)
				rcs.append(ac)
	
			return (ty,ch,rcs),v
		
		def get_frontier_type(v,ty=c_cfg.rootidx,pa=self.embed_root_idx):
			if len(v)==0:
				return (pa,ty),[]
			ch = v[0]
			v = v[1:]
			cts = self.trans_data[ty][ch]
			for i,ct in enumerate(cts):
				r,v = get_frontier_type(v,ct,self.embed_idx[ty][ch][i])
				if r is not None:
					return r,v
	
			return None,v
		
		beam_with = 1
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			xs = [x[::-1] for x in xs]
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_outputs = self.encoder(None, None, exs)
			print('encoded')
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			result = []
			
			for i in range(len(xs)):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
				beam_data = [(0.0,([],nhx,ncx))]
				
				print(self.n_maxsize,self.n_maxlen)
				for j in range(self.n_maxsize):
					to_beam = []
					for r,(kd,nhx,ncx) in beam_data:
						#print(kd)
						#print(unflatten(kd))
						paty,_ = get_frontier_type(kd)
						if paty is None:
							to_beam.append((r,(kd,nhx,ncx)))
							continue
						pa,ty = paty
						v = F.concat([self.embed_y_0(np.array([pa])),self.embed_y_1(np.array([ty]))],axis=1)
						#print(v.shape)
						thx,tcx,ys = self.decoder(nhx,ncx,[v])
						choice_num = len(self.trans_data[ty])
						
						wy = self.W_flatten(self.att(xs_outputs[i],ys[0])).data[0]
						#print(wy.shape,wy)
						wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.n_max_choice)),axis=1),(self.n_max_choice,)).data
						#print(wy.shape)
						to_beam += [(r+nr,(kd + [i],thx,tcx)) for i,nr in list(enumerate(wy))[:choice_num]]
					
					#print(to_beam[0][0])
					#print(list(map(lambda a: a[0],to_beam[:10])))
					beam_data = sorted(to_beam)[::-1][:beam_with]
					#print(list(map(lambda a: a[0],beam_data[:10])))
					
				result.append(beam_data[0][1][0])
		# Remove EOS taggs
		outs = []
		for y in result:
			outs.append(unflatten(y)[0])
		
		#print(outs)
		return outs
		
		
"""
		def unflatten(v,ty):
			#print(v,ty)
			_,ch = v[0]
			v = v[1:]
			cts = self.trans_data[ty][ch]
			#print(cts)
			rcs = []
			for ct in cts:
				ac,v = unflatten(v,ct)
				rcs.append(ac)
			
			return (ty,ch,rcs),v
"""
