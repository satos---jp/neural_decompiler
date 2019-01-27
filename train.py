#! /usr/bin/python3
#coding: utf-8

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import cuda

from model_seq2seq_pure import Seq2seq
from model_rnn_with_att import Seq2seq_with_Att
from model_rnn_with_globalatt import Seq2seq_with_GlobalAtt
from model_seq2tree import Seq2Tree
from model_seq2tree_flatten import Seq2Tree_Flatten

#from precision import prec_seq2tree

def convert(batch, device):
	def to_device_batch(batch):
		return batch

	return {'xs': to_device_batch([x for x,_ in batch]),
			'ys': to_device_batch([y for _,y in batch])}

import datetime
def getstamp():
	return datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

import code
#import edit_distance
import random
import pickle

from chainer import cuda
from cuda_setting import isgpu
if isgpu:
	xp = cuda.cupy
else:
	xp = np


class Asm_csrc_dataset(chainer.dataset.DatasetMixin):
	def __init__(self):
		super(Asm_csrc_dataset,self).__init__()
		
		self.len = 100000 * 150
		#self.len = 3200
		self.cache = []
		self.data_idx = []
		
		self.get_example = self.data_range(0,150)
		self.get_test = self.data_range(150,180)
		self.useddatanum = 0
		
	def __len__(self):
		return self.len
	
	def data_range(self,a,b):
		def gex(_):
			self.useddatanum += 1
			if self.cache == []:
				if self.data_idx == []:
					self.data_idx =list(range(a,b))
					random.shuffle(self.data_idx)
			
				fn = self.data_idx.pop()
				with open('dataset_asm_csrc/data_%d.pickle' % fn,'rb') as fp:
					self.cache = pickle.load(fp)[1]
					#self.cache = list(map(lambda xy: (np.array(xy[0],np.int32),xy[2]),self.cache))
					random.shuffle(self.cache)
					
			res = self.cache.pop()
			p,q = res
			#if size >= 100:
			#	return gex(-1)
			#print(size)
			return (xp.array(p,xp.int32),xp.array(q,xp.int32))
			#return (np.array([0],np.int32),(c_cfg.rootidx,0,[(0,0,[])]))
		
		return gex


def init_seq2seq():
	global train_iter_step
	n_layer = 4
	n_unit = 128
	n_maxlen = 200
	
	with open('dataset_asm_csrc/vocab_csrc.pickle','rb') as fp:
		c_vocab = pickle.load(fp)
	with open('dataset_asm_csrc/vocab_asm.pickle','rb') as fp:
		asm_vocab = pickle.load(fp)
	
	dataset = Asm_csrc_dataset()
	print('load dataset')
	print(len(dataset))
	
	v_eos_src = len(asm_vocab)
	asm_vocab += ['__EOS_ASM_']
	src_vocab_len = len(asm_vocab)
	
	v_eos_dst = len(c_vocab)
	c_vocab += ['__EOS_C_']
	dst_vocab_len = len(c_vocab)
 	
	if modelname == 'seq2seq':
		model = Seq2seq(n_layer, src_vocab_len, dst_vocab_len , n_unit, v_eos_dst, n_maxlen)
	elif modelname == 'seq2seq_att':
		model = Seq2seq_with_GlobalAtt(n_layer, src_vocab_len, dst_vocab_len , n_unit, v_eos_dst, n_maxlen)
	else:
		assert False
	#serializers.load_npz('iter1000_seq_att.npz',model)
	#serializers.load_npz('save_models/models_prestudy_edit_dist_0.65/iter_47600__edit_dist_0.691504__time_2018_12_28_01_36_30.npz',model)
	if not isgpu:
		#serializers.load_npz('iter1000.npz',model)
		pass
	if isgpu:
		model.to_gpu(0)

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	logt = 0
	train_iter_step = 100

	@chainer.training.make_extension()
	def translate(trainer):
		nonlocal logt,model,optimizer
		if isgpu:
			model.to_cpu()
		serializers.save_npz(modelname + '/models/iter_%s_time_%s.npz' % (logt,getstamp()),model)
		serializers.save_npz(modelname + '/optimizers/iter_%s_time_%s.npz' % (logt,getstamp()),optimizer)
		if isgpu:
			model.to_gpu(0)
		logt += train_iter_step
		
	return (model,optimizer,dataset,translate,asm_vocab)


def csize(d):
	res = 1
	_,_,cs = d
	for c in cs:
		res += csize(c)
	return res

class Asm_ast_dataset(chainer.dataset.DatasetMixin):
	def __init__(self):
		super(Asm_ast_dataset,self).__init__()
		
		#self.len = 1000190
		self.len = 2 * 100000 # 77385
		self.cache = []
		self.data_idx = []
		
		if modelname == 'seq2tree_cutoff':
			self.get_example = self.data_range(0,2)
			self.get_test = self.data_range(2,3)
		else:
			self.get_example = self.data_range(0,100)
			self.get_test = self.data_range(100,120)
		self.useddatanum = 0
		
	def __len__(self):
		return self.len
	
	
	# stop until 10^3 
	def data_range(self,a,b):
		def gex(_):
			self.useddatanum += 1
			if self.cache == []:
				if self.data_idx == []:
					self.data_idx =list(range(a,b))
					random.shuffle(self.data_idx)
			
				fn = self.data_idx.pop()
				if modelname == 'seq2tree_cutoff':
					fp = open('dataset_asm_ast/data_cutoff_%d.pickle' % fn,'rb')
				else:
					fp = open('dataset_asm_ast/data_sampled_%d.pickle' % fn,'rb')
				self.cache = pickle.load(fp)
				#self.cache = list(map(lambda xy: (np.array(xy[0],np.int32),xy[2]),self.cache))
				random.shuffle(self.cache)
			
			res = self.cache.pop()
			p,size,q = res
			#if size >= 100:
			#	return gex(-1)
			#print(size)
			return (np.array(p,np.int32),q)
			#return (np.array([0],np.int32),(c_cfg.rootidx,0,[(0,0,[])]))
		
		return gex

def load_dataset():
	
	with open('dataset_asm_ast/c_trans_array.pickle','rb') as fp:
		c_syntax_arr = pickle.load(fp)
	with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
		asm_vocab = pickle.load(fp)
	"""
	data = []
	for i in range(13):
		with open('dataset_asm_ast/data_%d.pickle' % i,'rb') as fp:
			data += pickle.load(fp)
	"""
	#with open('sampled_asm_ast_dataset.pickle','rb') as fp:
	#	res =  pickle.load(fp)
	
	res = (Asm_ast_dataset(),asm_vocab,c_syntax_arr)
	return res


train_iter_step = None


def init_seq2tree():
	global train_iter_step,modelname
	n_layer = 4
	n_unit = 128
	
	(data,asm_vocab,c_syntax_arr) = load_dataset()
	print('load dataset')
	print(len(data))
	
	n_maxsize = 800
	
	v_eos_src = len(asm_vocab)
	asm_vocab += ['__EOS__']
	src_vocab_len = len(asm_vocab)
 
	if modelname == 'seq2tree_flatten':
		model = Seq2Tree_Flatten(n_layer, src_vocab_len, c_syntax_arr, n_unit, v_eos_src, n_maxsize)
	elif modelname in ['seq2tree','seq2tree_cutoff']:
		model = Seq2Tree(n_layer, src_vocab_len, c_syntax_arr, n_unit, v_eos_src, n_maxsize)
	else:
		assert False
	#print(model.translate([data.get_test(-1)[0]]))
	#print(data.get_test(-1)[1])
	if not isgpu:
		#serializers.load_npz('iter1000.npz',model)
		#serializers.load_npz('iter2200_s2t.npz',model)
		pass
	if isgpu:
		model.to_gpu(0)
	
	logt = 0
	train_iter_step = 100
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	
	@chainer.training.make_extension()
	def translate(trainer):
		nonlocal logt,model,optimizer
		if isgpu:
			model.to_cpu()
		serializers.save_npz(modelname + '/models/iter_%s_time_%s.npz' % (logt,getstamp()),model)
		serializers.save_npz(modelname + '/optimizers/iter_%s_time_%s.npz' % (logt,getstamp()),optimizer)
		if isgpu:
			model.to_gpu(0)
		logt += train_iter_step
		"""
		nonlocal data
		ds = [data.get_test(-1) for _ in range(3)]
		ts = list(map(lambda x: model.xp.array(x[0]),ds))
		results = model.translate(ts)
		
		data = list(zip(ts,results))
		with open('log/iter_%s_time_%s.txt' % (logt,getstamp()),'w') as fp:
			pickle.dump(data,fp)
		"""
	return (model,optimizer,data,translate,asm_vocab)


modelname = 'seq2seq'

import os
def main():
	import sys
	sys.setrecursionlimit(10000)

	if isgpu:
		print('running in gpu')
		cuda.get_device(0).use()

	global train_iter_step 
	global modelname
	if modelname in ['seq2seq','seq2seq_att']:
		model,optimizer,dataset,translate,asm_vocab = init_seq2seq()
	elif modelname in ['seq2tree_flatten','seq2tree']:
		model,optimizer,dataset,translate,asm_vocab = init_seq2tree()
	else:
		assert False
	"""
	from precision import prec_seq2tree
	serializers.load_npz('iter6400_seq2tree.npz',model)
	prec_seq2tree(model,dataset,asm_vocab,'o4.pickle')
	exit()
	"""
	
	dns = os.listdir(modelname + '/models/')
	for i in range(21):
		pn = 'iter_%d_' % (i*100)
		nfn = list(filter(lambda x: pn in x,dns))
		assert len(nfn)==1
		nfn = nfn[0]
		serializers.load_npz(modelname + '/models/' + nfn,model)
		
		sfn = 'translate_data/' + modelname + '/' + pn + 'transdata.pickle'
		wds = []
		for t in range(1):
			#print(t)
			ds = [dataset.get_test(-1) for _ in range(1000)]
			results = model.translate(list(map(lambda x: x[0],ds)))
			wds += list(zip(ds,results))
		print('write to',sfn)
		with open(sfn,'wb') as fp:
			pickle.dump(wds,fp)
	exit()
	
	train_iter = chainer.iterators.SerialIterator(dataset, (1024 if isgpu else 6))
	updater = training.updaters.StandardUpdater(train_iter, optimizer,converter=convert)
	trainer = training.Trainer(updater, (10000000, 'epoch'), out='result')

	trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/loss', 
		 'validation/main/datanum',
		 'elapsed_time']),
		trigger=(1, 'iteration'))
	
	trainer.extend(translate, trigger=(train_iter_step, 'iteration'))
	
	class Reportdatanum(chainer.training.Extension):
		trigger = 1, 'iteration'
		priority = chainer.training.PRIORITY_WRITER
		def __call__(self,trainer):
			#print(dataset.useddatanum)
			chainer.report({'validation/main/datanum': dataset.useddatanum})
	trainer.extend(Reportdatanum(), trigger=(1, 'iteration'))
	
	trainer.run()

main()


"""
128 ::
0           1           252.627                                                                                   4.75998       
0           2           277.566                                                                                   5.89728       
0           3           323.39                                                                                    7.71957       
0           4           190.173                                                                                   8.84619       
0           5           237.659                                                                                   10.5173       
0           6           187.587                                                                                   11.7923       
0           7           193.1                                                                                     12.8792       
0           8           229.474                                                                                   14.6554       
0           9           222.666                                                                                   16.1893       
0           10          196.27                                                                                    17.6905       
0           11          207.201                                                                                   22.2253       
0           12          217.263                                                                                   23.5221       
0           13          174.542                                                                                   24.5908       
0           14          211.686                                                                                   26.2417       
0           15          167.472                                                                                   27.2718       
0           16          145.681                                                                                   28.6305       
0           17          180.189                                                                                   30.0821       
0           18          177.517                                                                                   31.4316       
0           19          149.224                                                                                   33.1968       
0           20          150.17                                                                                    34.9044       
0           21          183.886                                                                                   38.8256       
0           22          156.197                                                                                   39.9867       
0           23          149.079                                                                                   41.2111       
0           24          155.636                                                                                   42.5831       
0           25          190.169                                                                                   44.1674       
0           26          133.625                                                                                   45.1342       
0           27          141.91                                                                                    46.2559       
0           28          198.27                                                                                    47.9311       
0           29          153.244                                                                                   49.1867       
0           30          158.331                                                                                   50.6079       
0           31          145.206                                                                                   54.566        
0           32          158.711                                                                                   56.0405       
0           33          162.192                                                                                   57.2195       
0           34          137.014                                                                                   58.5285       
0           35          170.638                                                                                   59.9924       
0           36          213.641                                                                                   61.3654       
0           37          174.058                                                                                   62.6967       
0           38          121.399                                                                                   63.7692       
0           39          144.677                                                                                   64.928        
0           40          147.012                                                                                   66.2577       
0           41          170.172                                                                                   70.4797       
0           42          175.914                                                                                   72.0402       
0           43          160.253                                                                                   73.3652       
0           44          193.004                                                                                   75.0236       
0           45          172.668                                                                                   76.3752       
0           46          148.645                                                                                   77.7094       
0           47          141.254                                                                                   78.8417       
0           48          153.402                                                                                   80.5113       
0           49          186.067                                                                                   81.9098       
0           50          156.022                                                                                   83.3042       
0           51          152.798                                                                                   87.4779       
0           52          160.544                                                                                   88.8702       
0           53          119.213                                                                                   90.0091       
0           54          192.119                                                                                   91.423        
0           55          155.721                                                                                   92.5918       
0           56          134.298                                                                                   93.83         
0           57          118.475                                                                                   95.2126       
0           58          142.825                                                                                   96.5735       
0           59          155.614                                                                                   97.9724       
0           60          165.244                                                                                   99.3755       
0           61          119.995                                                                                   103.196       
0           62          158.002                                                                                   104.737       
0           63          127.247                                                                                   105.943       
0           64          130.589                                                                                   107.283       
0           65          147.574                                                                                   108.966       
0           66          124.79                                                                                    110.705       
0           67          137.226                                                                                   112.385       
0           68          136.006                                                                                   113.826       
0           69          141.285                                                                                   115.23        
0           70          151.784                                                                                   116.818       
0           71          143.042                                                                                   120.941       
0           72          148.702                                                                                   122.735       
0           73          175.556                                                                                   124.249       
0           74          135.399                                                                                   125.589       
0           75          127.809                                                                                   127.552       
0           76          103.204                                                                                   128.611       
0           77          169.652                                                                                   130.329       
0           78          129.758                                                                                   131.821       
0           79          117.23                                                                                    133.039       
0           80          88.0942                                                                                   134.384       
0           81          134.573                                                                                   138.492       
0           82          144.195                                                                                   139.771       
0           83          83.7688                                                                                   140.783       
0           84          97.0111                                                                                   142.61        
0           85          173.903                                                                                   144.365       
0           86          121.531                                                                                   145.946       
0           87          110.397                                                                                   147.371       
0           88          120.391                                                                                   148.675       
0           89          150.364                                                                                   150.3         
0           90          105.004                                                                                   151.706       
0           91          118.322                                                                                   155.888       
0           92          95.4582                                                                                   157.33        
0           93          118.496                                                                                   158.549       
0           94          155.907                                                                                   160.435       
0           95          96.9999                                                                                   162.266       
0           96          104.311                                                                                   163.558       
0           97          100.086                                                                                   165.148       
0           98          110.893                                                                                   166.565       
0           99          109.177                                                                                   167.956       
0           100         91.0065                                                                                   169.303       
0           101         79.0897                                                                                   173.153       
0           102         118.513                                                                                   174.935       
0           103         79.9844                                                                                   176.289       
0           104         98.2802                                                                                   177.493       
0           105         102.095                                                                                   178.925       
0           106         96.9165                                                                                   180.369       
0           107         131.165                                                                                   182.073       
0           108         98.4408                                                                                   183.323       
0           109         97.2777                                                                                   184.615       
0           110         118.91                                                                                    186.046       
0           111         99.9964                                                                                   190.053       
0           112         102.685                                                                                   191.463       
0           113         81.4883                                                                                   192.56        
0           114         95.7424                                                                                   194.102       
0           115         83.7703                                                                                   195.477       
0           116         65.2937                                                                                   196.612       
0           117         130.413                                                                                   197.997       
0           118         63.8102                                                                                   199.114       
0           119         74.727                                                                                    200.306       
0           120         106.112                                                                                   201.96        
0           121         89.4531                                                                                   206.063       
0           122         129.986                                                                                   207.428       
0           123         87.7359                                                                                   209.012       
0           124         83.8749                                                                                   210.73        
0           125         91.1136                                                                                   212.333       
0           126         83.1234                                                                                   213.967       
0           127         81.9057                                                                                   215.57        
0           128         85.47                                                                                     216.948       
0           129         83.4988                                                                                   218.366       
0           130         81.5499                                                                                   219.774       
0           131         63.3845                                                                                   223.948       
0           132         90.6956                                                                                   225.415       
0           133         101.35                                                                                    227.283       
0           134         91.5048                                                                                   228.542       
0           135         82.3857                                                                                   229.81        
0           136         76.0119                                                                                   231.21        
0           137         76.0103                                                                                   232.564       
0           138         66.6741                                                                                   233.694       
0           139         108.572                                                                                   235.32        
0           140         62.6313                                                                                   236.461       
0           141         71.2345                                                                                   240.299       
0           142         63.0012                                                                                   241.777       
0           143         87.968                                                                                    243.288       
0           144         63.5002                                                                                   244.339       
0           145         77.029                                                                                    245.903       
0           146         71.9717                                                                                   247.257       
0           147         64.76                                                                                     248.331       
0           148         61.5841                                                                                   249.697       
0           149         65.6348                                                                                   251.137       
0           150         75.5919                                                                                   252.975       
0           151         72.487                                                                                    256.714       
0           152         76.7178                                                                                   257.879       
0           153         59.0207                                                                                   259.195       
0           154         58.8694                                                                                   260.934       
0           155         57.3286                                                                                   261.869       
0           156         73.4312                                                                                   263.128       
0           157         100.846                                                                                   264.609       
0           158         69.0954                                                                                   266.451       
0           159         77.047                                                                                    267.848       
0           160         55.5447                                                                                   268.973       
0           161         57.557                                                                                    273.203       
0           162         68.7164                                                                                   274.464       
0           163         70.7515                                                                                   275.904       
0           164         64.0804                                                                                   277.293       
0           165         58.9836                                                                                   278.43        
0           166         62.9821                                                                                   279.711       
0           167         59.7598                                                                                   280.984       
0           168         56.9575                                                                                   282.25        
0           169         57.8422                                                                                   283.506       
0           170         54.1295                                                                                   285.085       
0           171         77.5898                                                                                   289.156       
0           172         66.4932                                                                                   290.388       
0           173         60.928                                                                                    291.893       
0           174         58.9986                                                                                   293.625       
0           175         52.792                                                                                    294.694       
0           176         63.0873                                                                                   295.854       
0           177         72.8561                                                                                   297.603       
0           178         60.609                                                                                    299.077       
0           179         58.4466                                                                                   300.404       
0           180         54.37                                                                                     301.806       
0           181         55.9891                                                                                   305.805       
0           182         50.1484                                                                                   306.993       
0           183         59.8921                                                                                   308.142       
0           184         56.9963                                                                                   309.615       
0           185         51.7971                                                                                   310.74        
0           186         66.4193                                                                                   312.119       
0           187         65.4606                                                                                   313.676       
0           188         63.5139                                                                                   315.009       
0           189         66.3956                                                                                   316.44        
0           190         48.7913                                                                                   317.836       
0           191         52.0756                                                                                   321.828       
0           192         55.9184                                                                                   323.323       
0           193         45.9439                                                                                   324.984       
0           194         55.0736                                                                                   326.501       
0           195         46.1799                                                                                   327.692       
0           196         48.4027                                                                                   329.743       
0           197         54.907                                                                                    330.958       
0           198         46.0646                                                                                   332.019       
0           199         46.3956                                                                                   333.189       
0           200         48.1586                                                                                   334.62        
0           201         55.3131                                                                                   338.926       
0           202         60.5457                                                                                   340.753       
0           203         46.7629                                                                                   342.522       
0           204         57.9261                                                                                   344.213       
0           205         42.6738                                                                                   345.293       
0           206         56.393                                                                                    346.602       
0           207         53.8654                                                                                   347.786       
0           208         64.7243                                                                                   349.474       
0           209         37.9634                                                                                   350.484       
0           210         57.3904                                                                                   351.949       
0           211         44.9759                                                                                   355.902       
0           212         68.6911                                                                                   357.709       
0           213         38.7147                                                                                   358.974       
0           214         49.5313                                                                                   360.463       
0           215         54.5073                                                                                   361.671       
0           216         47.5766                                                                                   362.861       
0           217         38.0438                                                                                   363.928       
0           218         41.2975                                                                                   365.425       
0           219         41.8397                                                                                   366.492       
0           220         48.8108                                                                                   367.929       
0           221         42.5263                                                                                   371.821       
0           222         46.6479                                                                                   373.659       
0           223         44.8487                                                                                   375.16        
0           224         54.5946                                                                                   376.668       
0           225         41.3815                                                                                   377.707       
0           226         58.781                                                                                    379.653       
0           227         45.5343                                                                                   381.144       
0           228         52.35                                                                                     382.443       
0           229         58.3683                                                                                   383.691       
0           230         46.1631                                                                                   384.889       
0           231         42.4301                                                                                   388.656       
0           232         40.6203                                                                                   389.968       
0           233         55.3652                                                                                   391.214       
0           234         47.7521                                                                                   392.801       
0           235         48.9761                                                                                   393.954       
0           236         52.9952                                                                                   395.529       
0           237         34.6115                                                                                   397.078       
0           238         47.8925                                                                                   398.573       
0           239         37.0497                                                                                   399.831       
0           240         40.0364                                                                                   401.078       
0           241         39.0714                                                                                   405.261       
0           242         40.7163                                                                                   406.874       
0           243         46.4605                                                                                   408.328       
0           244         46.0559                                                                                   409.912       
0           245         41.3949                                                                                   410.986       
0           246         42.8677                                                                                   412.756       
0           247         43.76                                                                                     414.281       
0           248         39.5166                                                                                   415.599       
0           249         46.3428                                                                                   417.253       
0           250         45.5168                                                                                   418.516       
0           251         40.022                                                                                    422.692       
0           252         34.9433                                                                                   423.845       
0           253         51.9581                                                                                   425.06        
0           254         45.6496                                                                                   426.443       
0           255         42.6863                                                                                   427.644       
0           256         46.3202                                                                                   428.863       
0           257         51.3629                                                                                   430.238       
0           258         59.7657                                                                                   431.943       
0           259         36.3449                                                                                   432.981       
0           260         36.2088                                                                                   434.112       
0           261         45.0007                                                                                   438.868       
0           262         50.1237                                                                                   440.226       
0           263         49.6206                                                                                   441.634       
0           264         43.9667                                                                                   443.218       
0           265         50.0119                                                                                   444.61        
0           266         54.1881                                                                                   445.928       
0           267         41.6175                                                                                   447.131       
0           268         39.6065                                                                                   448.431       
0           269         49.0601                                                                                   449.988       
0           270         44.5073                                                                                   451.345       
0           271         38.8319                                                                                   455.211       
0           272         43.0757                                                                                   456.642       
0           273         41.8302                                                                                   457.788       
0           274         35.0168                                                                                   458.804       
0           275         44.4992                                                                                   460.168       
0           276         38.0951                                                                                   461.52        
0           277         42.6944                                                                                   462.836       
0           278         49.6327                                                                                   464.566       
0           279         45.8477                                                                                   465.772       
0           280         43.5211                                                                                   466.918       
0           281         44.7476                                                                                   470.997       
0           282         52.5502                                                                                   472.36        
0           283         44.2101                                                                                   473.57        
0           284         38.0695                                                                                   474.83        
0           285         35.9484                                                                                   475.984       
0           286         49.6738                                                                                   477.85        
0           287         35.6277                                                                                   479.049       
0           288         42.6163                                                                                   480.526       
0           289         54.9567                                                                                   481.741       
0           290         43.9474                                                                                   482.948       
0           291         39.2589                                                                                   487.185       
0           292         43.2582                                                                                   488.835       
0           293         44.6923                                                                                   490.49        
0           294         45.7851                                                                                   491.621       
0           295         45.409                                                                                    493.068       
0           296         38.9114                                                                                   494.599       
0           297         39.8875                                                                                   495.804       
0           298         39.4636                                                                                   496.86        
0           299         46.276                                                                                    498.113       
0           300         46.917                                                                                    499.51        
0           301         41.7756                                                                                   503.425       
0           302         43.9648                                                                                   504.741       
0           303         52.6431                                                                                   506.355       
0           304         50.2347                                                                                   508.139       
0           305         34.1143                                                                                   509.446       
0           306         36.3351                                                                                   510.665       
0           307         41.6895                                                                                   512.208       
0           308         39.7641                                                                                   513.561       
0           309         42.2575                                                                                   514.913       
0           310         41.1203                                                                                   516.528       
0           311         41.2665                                                                                   520.412       
0           312         39.0183                                                                                   521.769       
0           313         40.6884                                                                                   522.984       
0           314         35.743                                                                                    524.262       
0           315         41.3599                                                                                   525.816       
0           316         31.0343                                                                                   526.981       
0           317         32.3325                                                                                   528.175       
0           318         59.1408                                                                                   529.532       
0           319         35.3639                                                                                   530.978       
0           320         47.0927                                                                                   532.332       
0           321         40.955                                                                                    536.178       
0           322         42.8168                                                                                   537.61        
0           323         50.6913                                                                                   539.215       
0           324         49.5531                                                                                   540.734       
0           325         37.3004                                                                                   541.933       
0           326         31.6539                                                                                   543.536       
0           327         32.4347                                                                                   544.721       
0           328         32.1234                                                                                   545.999       
0           329         32.8211                                                                                   547.147       
0           330         39.4566                                                                                   548.364       
0           331         30.9744                                                                                   552.144       
0           332         45.0875                                                                                   553.869       
0           333         38.5619                                                                                   555.114       
0           334         45.1244                                                                                   556.733       
0           335         37.0398                                                                                   558.19        
0           336         35.5945                                                                                   559.496       
0           337         30.7964                                                                                   560.894       
0           338         34.4199                                                                                   562.047    
"""


"""

1024::
0           1           242.283     1024                     12.5317       
0           2           264.449     2048                     20.0429       
0           3           257.43      3072                     27.2964       
0           4           242.043     4096                     34.78         
0           5           229.177     5120                     42.3159       
0           6           200.764     6144                     49.621        
0           7           195.56      7168                     57.0314       
0           8           174.355     8192                     64.1096       
0           9           207.034     9216                     72.2297       
0           10          224.524     10240                    79.962        
0           11          207.014     11264                    90.043        
0           12          200.528     12288                    97.8202       
0           13          178.77      13312                    105.089       
0           14          196.957     14336                    112.377       
0           15          197.855     15360                    119.818       
0           16          175.568     16384                    126.653       
0           17          195.984     17408                    134.835       
0           18          181.628     18432                    142.374       
0           19          169.43      19456                    149.544       
0           20          166.733     20480                    156.666       
0           21          171.365     21504                    166.476       
0           22          165.479     22528                    173.64        
0           23          178.659     23552                    180.721       
0           24          177.381     24576                    187.911       
0           25          168.694     25600                    195.663       
0           26          155.647     26624                    202.884       
0           27          163.096     27648                    210.216       
0           28          158.378     28672                    217.044       
0           29          169.354     29696                    224.436       
0           30          163.607     30720                    231.394       
0           31          159.726     31744                    241.189       
0           32          147.127     32768                    248.001       
0           33          164.369     33792                    255.785       
0           34          180.917     34816                    262.997       
0           35          179.161     35840                    270.075       
0           36          186.512     36864                    277.368       
0           37          187.996     37888                    284.632       
0           38          182.038     38912                    291.545       
0           39          216.779     39936                    298.986       
0           40          193.069     40960                    305.925       
0           41          168.048     41984                    316.748       
0           42          196.895     43008                    324.191       
0           43          174.718     44032                    331.077       
0           44          184.004     45056                    338.166       
0           45          174.729     46080                    345.459       
0           46          175.796     47104                    352.723       
0           47          201.072     48128                    359.94        
0           48          183.063     49152                    367.092       
0           49          173.547     50176                    375.548       
0           50          191.162     51200                    383.188       
0           51          177.983     52224                    393.19        
0           52          172.625     53248                    400.491       
0           53          165.348     54272                    407.618       
0           54          168.207     55296                    414.888       
0           55          181.554     56320                    422.468       
0           56          176.673     57344                    430.068       
0           57          159.652     58368                    437.812       
0           58          151.71      59392                    444.85        
0           59          166.718     60416                    452.308       
0           60          158.725     61440                    459.278       
0           61          148.705     62464                    469.089       
0           62          157.629     63488                    476.29        
0           63          146.31      64512                    483.606       
0           64          155.74      65536                    490.746       
0           65          156.203     66560                    497.901       
0           66          150.014     67584                    505.942       
0           67          156.474     68608                    513.535       
0           68          145.791     69632                    521.056       
0           69          144.685     70656                    528.343       
0           70          147.003     71680                    535.774       
0           71          145.461     72704                    545.371       
0           72          148.717     73728                    552.735       
0           73          134.987     74752                    559.878       
"""

