#!/bin/bash

pandoc -t beamer -s \
    -o output/ep_abc_pres.tex \
    -F pandoc-citeproc \
    ep_abc_pres.md

latexmk -pdfxe -silent -f \
    -jobname=output/ep_abc_pres \
    output/ep_abc_pres.tex

cp output/ep_abc_pres.pdf .
