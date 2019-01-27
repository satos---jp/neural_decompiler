cp ../../tex-srcs/iscs-thesis.cls ./
sed '/__NAMES__/r ../../names.txt' paper_base.tex | sed 's/__NAMES__//g' > paper.tex
extractbb bleu.png && extractbb edit_dist.png && extractbb ast_lens.png && extractbb c_lens.png && 
platex paper.tex && pbibtex paper && platex paper.tex && platex paper.tex && platex paper.tex && dvipdfm paper.dvi
