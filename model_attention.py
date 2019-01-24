import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

# from https://arxiv.org/pdf/1409.0473.pdf. Global attention.
class Attention(chainer.Chain):
	def __init__(self, n_units):
		super(Attention, self).__init__()
		with self.init_scope():
			self.att_W = L.Linear(n_units, n_units)
			self.att_U = L.Linear(n_units*2, n_units)
			self.att_v = L.Linear(n_units, 1)
		
		self.n_units = n_units
		self.Uhxs = None
	
	#@profile
	def init_hxs(self,hxs):
		self.hxs = hxs
		self.Uhxs = self.att_U(hxs)
		self.Uhxs_len = hxs.shape[0]
	
	#@profile
	def forward(self,nhx):
		nhx_for_att = nhx[-1]
		Wnhx = self.att_W(nhx_for_att)
		#print(Wnhx.shape,Uhxs.shape)
		#print((Wnhx + Uhxs).shape)
		
		att_rates = F.softmax(F.reshape(self.att_v(
				F.tanh(
					Wnhx + self.Uhxs
				)
		),(1,self.Uhxs_len)))
		#print(hxs.shape,att_rates.shape)
		#print(att_rates)
		#print(F.matmul(att_rates,hxs).shape)
		ctx = F.matmul(att_rates,self.hxs)[0]
		#print(ctx.shape)
		return ctx
		
	def reset(self):
		self.hxs = None
		self.Uhxs = None


class DotAttention(chainer.Chain):
	def __init__(self, n_units):
		super(DotAttention, self).__init__()
		self.n_units = n_units
		self.hxs = None
		
	#@profile
	def forward(self,xh,yh):
		#print(xh.shape,yh.shape)
		
		att_rates = F.softmax(F.matmul(yh,xh.T))
		#print(hxs.shape,att_rates.shape)
		#print(att_rates.shape)
		ds = F.sum(att_rates,axis=1)
		#print(ds[:10],ds.shape)
		#print(F.matmul(att_rates,hxs).shape)
		ctx = F.tanh(F.matmul(att_rates,xh))
		#print(ctx.shape)
		return ctx

class GlobalGeneralAttention(chainer.Chain):
	def __init__(self, n_units):
		super(GlobalGeneralAttention, self).__init__()
		self.n_units = n_units
		self.hxs = None
		with self.init_scope():
			self.Wa = L.Linear(n_units*2, n_units*2)
		
	#@profile
	def forward(self,xh,yh):
		#print(xh.shape,yh.shape)
		
		# O([a,b] * [b,c]) = a*b*c
		# a*b*c+a*c*d = a*c*(b+d)
		# a*b*d+b*c*d = (a+c)*b*d
		
		att_rates = F.softmax(F.matmul(self.Wa(yh),xh.T))
		#print(hxs.shape,att_rates.shape)
		#print(att_rates.shape)
		ds = F.sum(att_rates,axis=1)
		#print(ds[:10],ds.shape)
		#print(F.matmul(att_rates,hxs).shape)
		ctx = F.tanh(F.matmul(att_rates,xh))
		#print(ctx.shape)
		return ctx

class LocalDotAttention(chainer.Chain):
	def __init__(self, n_units):
		super(DotAttention, self).__init__()
		self.n_units = n_units
		self.hxs = None
		with self.init_scope():
			self.Wp = L.Linear(n_units, n_units)
			self.vp = L.Linear(n_units, 1)		
	
	#@profile
	def forward(self,xh,yh):
		#print(xh.shape,yh.shape)
		
		p_t = xh.shape[0] * F.sigmoid(self.vp(F.tanh(self.Wp(yh))))		
		#p_att = F.exp(
		att_rates = F.softmax(F.matmul(yh,xh.T)*p_att)
		
		#print(hxs.shape,att_rates.shape)
		#print(att_rates.shape)
		ds = F.sum(att_rates,axis=1)
		#print(ds[:10],ds.shape)
		#print(F.matmul(att_rates,hxs).shape)
		ctx = F.tanh(F.matmul(att_rates,xh))
		#print(ctx.shape)
		return ctx



			
