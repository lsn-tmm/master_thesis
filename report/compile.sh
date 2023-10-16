rm mainNotes.bib main.aux main.out main.dvi main.log main.ps

latex main.tex 
latex main.tex 
latex main.tex 
bibtex main
latex main.tex 
latex main.tex
latex main.tex
dvips main.dvi
ps2pdf main.ps

rm mainNotes.bib main.aux main.out main.dvi main.log main.ps
