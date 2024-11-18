import os
from datetime import datetime
import hashlib 

# Define the directory to search for folders
directory = "/Users/kaoru/Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/pdfpod/"

# Define the template for the markdown file
template = """---
actor_ids:
  - alice
  - bob
audio_file_path: /audio/{folder}.wav
transcript_path: /transcript/{folder}.txt
pdffile_path: /pdf/{folder}.pdf
date: {date}
description: Auto-generated podcast article for {folder}.
layout: article
title: {folder}
---

"""

# Get the list of folders in the directory
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Create output directory if it doesn't exist
output_dir = "/Users/kaoru/Desktop/podcast-gen/pdfcast/_posts/"
os.makedirs(output_dir, exist_ok=True)

# Process each folder
for i,folder in enumerate(folders):
    folder_path = os.path.join(directory, folder)
    # Write the markdown file
    num = hashlib.md5(folder.encode()).hexdigest()
    # Check if there's a file that contains num in the _posts directory
    existing_files = [f for f in os.listdir(output_dir) if str(num) in f]
    if existing_files:
        print(f"Skipping existing podcast article for folder: {folder}")
        continue
    markdown_file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{num}.md"
    markdown_file_path = os.path.join(output_dir, markdown_file_name)
    
    if os.path.exists(markdown_file_path):
        print(f"Skipping existing podcast article: {markdown_file_path}")
        continue
    audio_file_path = folder_path + "/" + folder + ".wav"  # Replace spaces with underscores for file names
    pdf_file_path = folder_path + "/" +  folder + ".pdf"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0900")
    
    # Generate markdown content
    markdown_content = template.format(
        folder=folder,
        date=date
    )
    
    with open(markdown_file_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    # Copy the audio file to the audio directory
    audio_output_dir = "audio/"
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_output_path = os.path.join(audio_output_dir, os.path.basename(audio_file_path))

    if os.path.exists(audio_file_path):
        with open(audio_file_path, "rb") as src_file:
            with open(audio_output_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        print(f"Copied audio file to: {audio_output_path}")
    else:
        print(f"Audio file not found: {audio_file_path}")

    # Copy the pdf file to the pdf directory
    pdf_output_dir = "pdf/"
    os.makedirs(pdf_output_dir, exist_ok=True)
    pdf_output_path = os.path.join(pdf_output_dir, os.path.basename(pdf_file_path))
    
    if os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as src_file:
            with open(pdf_output_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        print(f"Copied pdf file to: {pdf_output_path}")
    else:
        print(f"PDF file not found: {pdf_file_path}")
    
    print(f"Created podcast article: {markdown_file_path}")