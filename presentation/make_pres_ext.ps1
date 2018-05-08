cd ~/Documents/dev/abc2018/presentation

pandoc -t beamer -s `
    -o output/ep_abc_pres_ext.tex `
    -F pandoc-citeproc `
    ep_abc_pres.md

latexmk -pdfxe -silent -f `
    -jobname=output/ep_abc_pres_ext `
    output/ep_abc_pres.tex

cp output/ep_abc_pres.pdf .