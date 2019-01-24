import sys

isgpu = False
if len(sys.argv)>=2 and sys.argv[1]=='gpu':
	isgpu = True

