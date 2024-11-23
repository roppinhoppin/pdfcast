#!/bin/bash
git pull
conda activate pdfcast
for pdf_file in pdf/*.pdf; do
    magic-pdf -p "$pdf_file" -o pdf/
done
git add pdf/
git commit -m "update pdf using minerU on popos"
git push
