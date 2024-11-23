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
# copy magic-pdf/{num}/auto/images to image/{num}/
for num in $(ls magic-pdf); do
  if [ -d "magic-pdf/$num/auto/images" ]; then
    mkdir -p "images/$num"
    cp -r "magic-pdf/$num/auto/images/" "images/$num/"
  fi
done
python retrieve.py
git add .
git commit -m "Update after running popos.sh and retrieve.py"
git push
