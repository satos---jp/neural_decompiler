cp ../../tex-srcs/iscs-thesis.cls ./
sed '/__NAMES__/r ../../names.txt' paper_base.tex | sed 's/__NAMES__//g' > paper.tex
platex paper.tex && pbibtex paper && platex paper.tex && dvipdfm paper.dvi
