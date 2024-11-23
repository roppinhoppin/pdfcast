#!/bin/bash
python figure_extract.py
git pull
git add pdf/
git commit -m "update pdf using minerU on popos"
git push
