#!/bin/bash
git pull
conda activate pdfcast
python figure_extract.py
git add pdf/
git commit -m "update pdf using minerU on popos"
git push
