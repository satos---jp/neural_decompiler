import code
import re

def get_line_from_list(asm,objd):
	asm = list(map(lambda x: x.split('#')[0].strip(),asm.split('\n')))
	asm_with_loc = list(filter(lambda x: x!='' and x[0]!='#' and x[-1]!=':' and (x[0] != '.' or x[:5]=='.loc '),asm))
	asm_noloc = list(filter(lambda x: x!='' and x[0]!='#' and x[-1]!=':' and x[0]!='.',asm))

	"""
	while asm_with_loc[-1][-1]==':':
		asm_with_loc = asm_with_loc[:-1]
	while asm_noloc[-1][-1]==':':
		asm_noloc = asm_noloc[:-1]
	"""
	#exit()
	objd = objd.split('\n')[5:]
	objd = list(filter(lambda x: x!='',objd))
	objd = list(map(lambda x: x.split('#')[0].strip(),objd))
	objd_asm_len = len(list(filter(lambda x: x[-1]!=':',objd)))
	"""
	# only for clang
	tobjd = []
	for i,v in enumerate(objd[:-1]):
		if objd[i+1][-1]==':':
			continue
		tobjd.append(v)
	objd = tobjd + [objd[-1]]
	"""
	#print(objd)
	#print(len(asm2),len(objd))
	#assert len(objd) == len(asm2)
	#print(len(objd),len(asm_noloc))
	"""
	print('\n'.join(asm_with_loc))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	if objd_asm_len != len(asm_noloc):
		"""
		print('\n'.join(asm_noloc))
		print('-' * 100)
		print('\n'.join(objd))
		print('nanka hen',objd_asm_len,len(asm_noloc))
		exit()
		
		asm volatile があるので諦める
		"""
		raise ParseErr("nanka hen")
	
	"""
	print(len(asm),len(objd))
	print('\n'.join(asm_with_loc))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	
	addrs = set([])
	for d in objd:
		if d[-1]==':':
			continue
		op = d.split('\t')[2]
		if op[0]=='j' or op[:4] == 'call':
			nad = list(filter(lambda a: a!='',op.split(' ')))[1]
			addrs |= set([nad])
			#print(op,nad)
	
	addrs = list(addrs)
	tobjd = []
	for d in objd:
		if d[-1]==':':
			if d[:-1] in addrs:
				tobjd.append(['LABEL_%d' % addrs.index(d[:-1]),':'])
			continue
		naddr = d.split('\t')[0][:-1].strip()
		if naddr in addrs:
			tobjd.append(['LABEL_%d' % addrs.index(naddr),':'])
		op = d.split('\t')[2]
		if op[0]=='j' or op[:4] == 'call':
			nad = list(filter(lambda a: a!='',op.split(' ')))[1]
			tobjd.append([op.split(' ')[0],'LABEL_%d' % addrs.index(nad)])
		else:
			nops = []
			be = ''
			op += ' '
			for c in op:
				if c.isalnum():
					be += c
				else:
					if be != '':
						if be != 'PTR':
							nops.append(be)
						be = ''
					if c != ' ':
						nops.append(c)
			
			tobjd.append(nops)
	#exit()
	
	i = 0
	res = []
	nld = 0
	for s in asm_with_loc:
		if s[:5]=='.loc ':
			# nld = s.split('\t')[1].split(' ')[1] # for clang -S
			nld = s.split(' ')[2] # for gcc -S
		else:
			#print(nld,tobjd[i])
			while tobjd[i][-1]==':':
				res.append((int(nld)-1,tobjd[i])) #1-indexed!!
				i += 1
			res.append((int(nld)-1,tobjd[i])) #1-indexed!!
			i += 1
	return res
