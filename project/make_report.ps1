cd ~/Documents/dev/abc2018/project
pandoc -o output/report.tex -s --biblatex `
    -F pandoc-eqnos `
    -F pandoc-fignos `
    -F pandoc-tablenos `
    -F pandoc-citeproc `
    report.md

# Copy necessary files
cp refs.bib output/
md -Force output/figures
cp figures/* output/figures/

cd output
latexmk -pdfxe -silent -f report.tex
cd ..
cp output/report.pdf .