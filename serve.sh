#!/bin/bash

# copy magic-pdf/{num}/auto/images to image/{num}/
for num in $(ls magic-pdf); do
  if [ -d "magic-pdf/$num/auto/images" ]; then
    mkdir -p "images/$num"
    cp -r "magic-pdf/$num/auto/images/" "images/$num/"
  fi
done

python retrieve.py --update

# This script is expected to used after running bundle install github-pages
bundle exec jekyll serve --watch --host 0.0.0.0
