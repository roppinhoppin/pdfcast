#!/bin/bash

export magicpdf="docker run --rm -it --gpus=all -v $(pwd)/pdf:/pdf mineru:latest magic-pdf"
git pull
for pdf_file in pdf/*.pdf; do
    pdf_filename=$(basename "$pdf_file" .pdf)
    echo "pdf_filename: $pdf_filename"
    if [ -f "pdf/${pdf_filename}/auto/${pdf_filename}.md" ]; then
        echo "File pdf/${pdf_filename}/auto/${pdf_filename}.md already exists. Skipping conversion."
        continue
    fi
    $magicpdf -p "$pdf_file" -o pdf/
done
# sudo chown -R $(whoami):$(whoami) pdf/
git add pdf/
git commit -m "update pdf using minerU on popos"
git push
