#!/bin/zsh
python retrieve.py --update
git add .
git commit -m "Update after running retrieve.py"
git push
ssh popos 'cd pdfcast && ./popos.sh'
python retrieve.py --update
git add .
git commit -m "Update after running popos.sh and retrieve.py"
git push
