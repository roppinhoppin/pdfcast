import os
from datetime import datetime

# Define the directory to search for folders
directory = "/Users/kaoru/Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/pdfpod/"

# Define the template for the markdown file
template = """---
actor_ids:
  - alice
  - bob
audio_file_path: /audio/{audio_file_name}.wav
transcript_path: /transcript/{audio_file_name}.txt
pdffile_path: /pdf/{pdf_file_name}.pdf
date: {date}
description: Auto-generated podcast article for {audio_file_name}.
layout: article
title: {audio_file_name}
---

## 関連リンク
"""

# Get the list of folders in the directory
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Create output directory if it doesn't exist
output_dir = "/Users/kaoru/Desktop/podcast-gen/pdfcast/_posts/"
os.makedirs(output_dir, exist_ok=True)

# Process each folder
for folder in folders:
    folder_path = os.path.join(directory, folder)
    audio_file_name = folder + ".wav"  # Replace spaces with underscores for file names
    pdf_file_name = folder + ".pdf"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0900")
    
    # Generate markdown content
    markdown_content = template.format(
        audio_file_name=audio_file_name,
        pdf_file_name=pdf_file_name,
        date=date
    )
    
    # Write the markdown file
    markdown_file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{folder}.md"
    markdown_file_path = os.path.join(output_dir, markdown_file_name)
    
    with open(markdown_file_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)
    
    print(f"Created podcast article: {markdown_file_path}")