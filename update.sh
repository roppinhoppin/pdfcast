#!/bin/zsh
python retrieve.py
git add .
git commit -m "Update after running retrieve.py"
git push 
