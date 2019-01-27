import pickle

def prec_seq2seq(model):
	def normalize_sentense(s):
		ts = []
		names = {}
		for d in s:
			if d[:5]=='__ID_':
				if not sd in names.keys():
					names[d] = '__NORMALIZE_TMP_ID_%d__' % len(names.keys())
				d = names[d]
			ts.append(d)
		return ts
	
	logt = 0
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = random.sample(test_data, 100)
		for fr,to in ds:
			result = model.translate([model.xp.array(fr)])[0]
			result = list(result)
			result = normalize_sentense(result)
			to = normalize_sentense(to)
			ratio += 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
		#print(ratio)
		ratio /= len(ds)
		return ratio
	
	@chainer.training.make_extension()
	def translate(trainer):
		eds = calcurate_edit_distance()
		print('edit distance:',eds)
		nonlocal logt
		logt += train_iter_step
		with open('log/iter_%s_editdist_%f_time_%s.txt' % (logt,eds,getstamp()),'w') as fp:
			res = ""
			for _ in range(10):
				source, target = test_data[np.random.choice(len(test_data))]
				result = model.translate([model.xp.array(source)])[0]
				
				#target = normalize_sentense(target)
				#result = normalize_sentense(result)
				
				#source_sentence = ' '.join([asm_vocab[x] for x in source])
				source_sentence = ' '.join([str(x) for x in source])
				target_sentence = ' '.join(csrc2ast.ast2token_seq(target))
				result_sentence = ' '.join(csrc2ast.ast2token_seq(result))
				res += ('# source : ' + source_sentence) + "\n"
				res += ('# expect : ' + target_sentence) + "\n"
				#print('# result_array : ' + ' '.join(["%d" % y for y in result]))
				res += ('# result : ' + result_sentence) + "\n"
			fp.write(res)
		
		serializers.save_npz('models/iter_%s_editdist_%f_time_%s.npz' % (logt,eds,getstamp()),model)
		
		#import code
		#code.interact(local={'d':model.embed_x})
		#print('embed data : ', model.embed_x.W)
	
	return translate


import csrc2ast
import edit_distance

def prec_seq2tree(model,data,asm_vocab,fn):
	
	def translate():
		ds = [data.get_test(-1) for _ in range(100)]
		results = model.translate(list(map(lambda x: x[0],ds)))
		wds = list(zip(ds,results))
		print('write to',fn)
		with open(fn,'wb') as fp:
			pickle.dump(wds,fp)
		
	translate()

