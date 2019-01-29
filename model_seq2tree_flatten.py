import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers
#import c_cfg

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


from model_attention import Attention,DotAttention,GlobalGeneralAttention

class Seq2Tree_Flatten(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, trans_data, n_units,v_eos_src,n_maxsize):
		super(Seq2Tree_Flatten, self).__init__()

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
		
		self.choicerange = []
		self.choice_idx = []
		self.is_trivial = []
		s = 0
		for d in self.trans_data:
			ist = len(d)<=1
			self.is_trivial.append(ist)
			#if ist:
			#	self.choicerange.append(None)
			#	self.choice_idx.append([0])
			#	continue
			b = s
			s += len(d)
			self.choicerange.append((b,s))
			self.choice_idx.append(list(range(b,s)))
		#self.choice_num_sum = sum(list(map(lambda d: len(d),self.trans_data)))
		
		self.type_size = len(self.embed_idx)
		self.n_all_choice = sum(map(lambda x: len(x),self.trans_data))
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			#self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			self.embed_y_0 = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			self.embed_y_1 = L.EmbedID(self.type_size, n_units) # maybe mergable
			
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units*2, n_units*2, 0.1)
			self.Wc = L.Linear(n_units*4, n_units)
			self.Ws = L.Linear(n_units, self.n_all_choice)
			
			#self.att = Attention(n_units)
			self.att = GlobalGeneralAttention(n_units)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.n_maxsize = n_maxsize
		self.rootidx = len(trans_data)-1
		


	#@profile
	def forward(self, xs, ys):
		xs = [xp.array(x[::-1]) for x in xs]
		
		def flatten(y,pa=self.embed_root_idx):
			ty,ch,cs = y
			res = [((pa,ty),ch)]
			for i,c in enumerate(cs):
				eidx = self.embed_idx[ty][ch][i]
				res += flatten(c,eidx)
			return res
		
		#assert all(map(lambda x: unflatten(flatten(x),c_cfg.rootidx)[0]==x,ys))
		
		ys = list(map(flatten,ys))
		ys_in_0  = [xp.array(list(map(lambda ab: ab[0][0],y))) for y in ys]
		ys_in_1  = [xp.array(list(map(lambda ab: ab[0][1],y))) for y in ys]
		ys_out = [xp.array(list(map(lambda ab: self.choice_idx[ab[0][1]][ab[1]],y))) for y in ys]
		ys_conds = [xp.array(list(map(lambda ab: self.is_trivial[ab[0][1]],d)),dtype=xp.bool) for d in ys]
		#print(ys_conds)
		#ys_out = [xp.array(list(map(lambda ab: ab[1],y))) for y in ys]
		
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
		
		#print(hx.shape,cx.shape,eys[0].shape)
		#print(hx[:,:,1],hx[:,:,1].shape)
		#exit()
		
		hx = F.transpose(F.reshape(F.transpose(hx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
		cx = F.transpose(F.reshape(F.transpose(cx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
		# CPU :: (8, 4, 128) (8, 4, 128) (length , 256)
		# GPU :: (8, 4, 128) (8, 4, 128) (length , 256)
		_, _, os = self.decoder(hx, cx, eys)
		#print('decode')
		
		ctxs = [self.att(xh,yh) for (xh,yh) in zip(xs_states,os)]
		#print('attentioned')
		att_os = [F.tanh(self.Wc(F.concat([ch,yh],axis=1))) for (ch,yh) in zip(ctxs,os)]
		#print(ys_out[0].shape,att_os[0].shape)
		
		concat_os = F.concat(att_os, axis=0)
		concat_ys_out = F.concat(ys_out, axis=0)
		concat_cond = F.concat(ys_conds, axis=0)
		sxe = F.softmax_cross_entropy(self.Ws(concat_os), concat_ys_out, reduce='no')
		sxec = F.where(concat_cond,xp.zeros(sxe.shape,dtype=xp.float32),sxe)

		loss = F.sum(sxec) / batch
		#print('lossed')
		chainer.report({'loss': loss}, self)
		#n_words = concat_ys_out.shape[0]
		#perp = self.xp.exp(loss.array * batch / n_words)
		#chainer.report({'perp': perp}, self)
		return loss


	def translate(self, xs):
		
		batch = len(xs)
		
		def unflatten(v,ty=self.rootidx):
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
		
		def get_frontier_type(v,ty=self.rootidx,pa=self.embed_root_idx):
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
		
		
		#self.n_maxsize = 10000
		beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			xs = [xp.array(x[::-1]) for x in xs]
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_outputs = self.encoder(None, None, exs)
			hx = F.transpose(F.reshape(F.transpose(hx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
			cx = F.transpose(F.reshape(F.transpose(cx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
			#print('encoded')
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			result = []
			"""
			self.n_maxsize = 800
			beam_with = 3
			vs = [xp.array([EOS_DST]) for _ in xs_outputs]
			kds = [[] for _ in xs_outputs]
			rs = [0.0 for _ in xs_outputs]
			beam_data = [(rs,kds,hx,cx,vs)]
			for j in range(self.n_maxlen):
				print(j)
				to_beam = [[] for _ in range(batch)]
				for rs,kds,nhx,ncx,vs in beam_data:
					if type(nhx) is list:
						#print(nhx[0].shape)
						nhx = F.stack(nhx,axis=1)
						ncx = F.stack(ncx,axis=1)
						vs = list(map(lambda d: xp.array(d),vs))
						#print(nhx.shape)
					#print(rs,kds,nhx,ncx,vs)
					evs = sequence_embed(self.embed_y,vs)
					thx,tcx,ys = self.decoder(nhx,ncx,evs)
					ctxs = [self.att(xh,yh) for (xh,yh) in zip(xs_outputs,ys)]
					att_ys = [F.tanh(self.Wc(F.concat([ch,yh],axis=1))) for (ch,yh) in zip(ctxs,ys)]
					wy = self.Ws(F.concat(att_ys, axis=0))
					#print(wy.shape)
					wy = F.log_softmax(wy).data
					#print(thx.shape)
					thx = F.separate(thx, axis=1)
					tcx = F.separate(tcx, axis=1)
					#print(thx[0].shape)
					for i in range(batch):
						ahx,acx = thx[i],tcx[i]
						for t,nr in enumerate(wy[i]):
							to_beam[i].append((rs[i]+nr,kds[i] + [t],ahx,acx,[t]))
				
				to_beam = list(map(lambda x: sorted(x)[::-1][:beam_with],to_beam))
				beam_data = [tuple([[to_beam[i][k][s] for i in range(batch)] for s in range(5)]) for k in range(beam_with)]
		
		result = beam_data[0][1] # for i in range(batch)
		
			"""
			for i in range(len(xs)):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
				beam_data = [(0.0,([],nhx,ncx))]
				
				#print(self.n_maxsize,self.n_maxlen)
				for j in range(self.n_maxsize):
					print(j)
					to_beam = []
					for ci,(r,(kd,nhx,ncx)) in enumerate(beam_data):
						#print(kd)
						#print(unflatten(kd))
						paty,_ = get_frontier_type(kd)
						if paty is None:
							to_beam.append((r,(kd,nhx,ncx)))
							if ci==0:
								break
							continue
						pa,ty = paty
						v = F.concat([self.embed_y_0(xp.array([pa])),self.embed_y_1(xp.array([ty]))],axis=1)
						#print(v.shape)
						thx,tcx,ys = self.decoder(nhx,ncx,[v])
						
						yh = ys[0]
						ctx = self.att(xs_outputs[i],yh)
						att_yh = F.tanh(self.Wc(F.concat([ctx,yh],axis=1)))
						#print(ty)
						choice_from,choice_to = self.choicerange[ty]
						#if ty==8 and j>=200:
						#	choice_to = choice_from + 1
						cl = choice_to - choice_from
						wy = self.Ws(att_yh).data[0][choice_from:choice_to]
						#print(wy.shape,wy)
						wy = F.reshape(F.log_softmax(F.reshape(wy,(1,cl))),(cl,)).data
						#print(wy.shape)
						to_beam += [(r+nr,(kd + [i],thx,tcx)) for i,nr in list(enumerate(wy))]
					
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



