cd ~/Documents/open_source/abc2018/project
pandoc -o output/report.tex -s --biblatex -F pandoc-citeproc report.md
cp refs.bib output/

cd output
latexmk -pdf -silent report.tex
cd ..
cp output/report.pdf .