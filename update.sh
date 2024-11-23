#!/bin/zsh
python retrieve.py --update
git add .
git commit -m "Update after running retrieve.py"
git push 
