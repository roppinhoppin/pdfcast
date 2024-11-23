#!/bin/zsh
git pull
python retrieve.py
git add .
git commit -m "Update after running retrieve.py"
git push
echo "Done retrieve.py"
ssh -t popos 'cd pdfcast && ./popos.sh'
echo "Done popos.sh"
git pull
python retrieve.py
git add .
git commit -m "Update after running popos.sh and retrieve.py"
git push
