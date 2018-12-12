cp ../../tex-srcs/iscs-thesis.cls ./
sed '/__NAMES__/r ../../names.txt' abstract_base.tex | sed 's/__NAMES__//g' > abstract.tex
platex abstract.tex && dvipdfm abstract.dvi
