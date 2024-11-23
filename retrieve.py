import argparse
import hashlib
import os
from datetime import datetime

import figure_extract

# Define the directory to search for folders
directory = "/Users/kaoru/Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/pdfpod/"

# Define the template for the markdown file
template = """---
audio_file_path: /audio/{num}.wav
transcript_path: /transcript/{num}.txt
pdffile_path: /pdf/{num}.pdf
date: {date}
images: {images}
description: AI-generated podcast from the PDF file {folder}
layout: article
title: {folder}
---

## 文字起こし
{transcription}

"""

parser = argparse.ArgumentParser(description=f"Retrieve podcast data from directory {directory} and turn the data into podcast")
parser.add_argument("--update", action="store_true", help="Update existring article when .")
args = parser.parse_args()

# Get the list of folders in the directory
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Create output directory if it doesn't exist
output_dir = "/Users/kaoru/Desktop/podcast-gen/pdfcast/_posts/"
os.makedirs(output_dir, exist_ok=True)

# Process each folder
for i,folder in enumerate(folders):
    is_wav = False
    is_pdf = False
    is_txt = False

    folder_path = os.path.join(directory, folder)
    # Write the markdown file
    num = hashlib.md5(folder.encode()).hexdigest()
    # Check if there's a file that contains num in the _posts directory
    existing_files = [f for f in os.listdir(output_dir) if str(num) in f]
    markdown_file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{num}.md"
    if existing_files:
        print(f"Skipping existing podcast article for folder: {folder}")
        if args.update:
            markdown_file_name = existing_files[0]
        else:
            continue
    markdown_file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{num}.md"
    markdown_file_path = os.path.join(output_dir, markdown_file_name)
    
    audio_file_path = folder_path + "/" + folder + ".wav"  # Replace spaces with underscores for file names
    pdf_file_path = folder_path + "/" +  folder + ".pdf"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S +0900")
    

    # Copy the audio file to the audio directory
    audio_output_dir = "audio/"
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_output_path = os.path.join(audio_output_dir, str(num) + ".wav")

    if os.path.exists(audio_file_path):
        with open(audio_file_path, "rb") as src_file:
            with open(audio_output_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        print(f"Copied audio file to: {audio_output_path}")
        is_wav = True
    else:
        print(f"Audio file not found: {audio_file_path}")

    # Copy the pdf file to the pdf directory
    pdf_output_dir = "pdf/"
    os.makedirs(pdf_output_dir, exist_ok=True)
    pdf_output_path = os.path.join(pdf_output_dir, str(num) + ".pdf")
    
    if os.path.exists(pdf_file_path):
        with open(pdf_file_path, "rb") as src_file:
            with open(pdf_output_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        print(f"Copied pdf file to: {pdf_output_path}")
        is_pdf = True
    else:
        print(f"PDF file not found: {pdf_file_path}")

    # Copy the transcript file to the transcript directory
    transcript_file_path = folder_path + "/" + folder + ".txt"
    transcript_output_dir = "transcript/"
    os.makedirs(transcript_output_dir, exist_ok=True)
    transcript_output_path = os.path.join(transcript_output_dir, str(num) + ".txt")

    if os.path.exists(transcript_file_path):
        with open(transcript_file_path, "rb") as src_file:
            with open(transcript_output_path, "wb") as dst_file:
                dst_file.write(src_file.read())
        print(f"Copied transcript file to: {transcript_output_path}")
        is_txt = True
    else:
        print(f"Transcript file not found: {transcript_file_path}")

    # Read the description from the transfript_file_path
    transcription = ""
    if os.path.exists(transcript_file_path):
        with open(transcript_file_path, "r") as file:
            transcription = file.read()
    else:
        print(f"Transcript file not found: {transcript_file_path}")

    # Extract images from the PDF file
    image_paths = []
    pdf_image_folder = os.path.join("pdf", str(num), "auto", "images")
    if os.path.exists(pdf_image_folder):
        image_paths = [os.path.join(pdf_image_folder, f) for f in os.listdir(pdf_image_folder) if f.endswith(".png")]
    else:
        print(f"Figure folder not found: {pdf_image_folder}")

    # Generate markdown content
    markdown_content = template.format(
        folder=folder,
        date=date,
        num=num,
        images=image_paths,
        transcription=transcription
    )
    
    if is_wav and is_pdf:
        with open(markdown_file_path, "w", encoding="utf-8") as file:
            file.write(markdown_content)
        print(f"Created podcast article: {markdown_file_path}")
    else:
        print(f"Skipping podcast article creation for folder: {folder}")
