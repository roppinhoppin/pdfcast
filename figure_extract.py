import os
import shutil
import subprocess

magicpdfbin = "/Users/kaoru/anaconda3/envs/mineru/bin/magic-pdf"


def convert_from_path(folder_path, update=False):
    folder_path = os.path.normpath(folder_path)
    folder = os.path.basename(folder_path)
    pdf_file_path = folder_path + "/" + folder + ".pdf"
    main_pdf_path = os.path.join(folder_path, "main.pdf")

    # Copy the pdf_file_path to main.pdf
    shutil.copy(pdf_file_path, main_pdf_path)

    mineru_folder_path = folder_path + "/main"
    image_folder = os.path.join(mineru_folder_path, "auto/images/")
    # print(mineru_folder_path)
    if os.path.exists(mineru_folder_path) and not update:
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files
    else:
        cp = subprocess.run(
            "whereis magic-pdf",
            shell=False,
            capture_output=True,
            check=False,
            text=True,
        )
        response = cp.stdout.strip().split()
        if cp.returncode or len(response) != 2:  # if not 2 tokens: no tesseract-ocr
            raise Exception("magic-pdf is not installed")
        output = subprocess.run(
            f"magic-pdf -p {pdf_file_path} -o {folder_path}",
            shell=1,
            capture_output=1,
            check=0,
            text=True,
        )

        # command = [
        #     f"{magicpdfbin}",
        #     "-s",
        #     "12",
        #     "-e",
        #     "12",
        #     "-p",
        #     "main.pdf",
        #     "-o",
        #     "./",
        # ]
        # output = subprocess.run(
        #     command,
        #     shell=False,
        #     capture_output=True,
        #     check=False,
        #     text=True,
        #     cwd=folder_path,
        # )
        print(output)
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files


def convert_from_pdf(pdf_path, update=False):
    folder_path = os.path.dirname(pdf_path)
    image_folder = os.path.join(folder_path, "auto/images/")
    if os.path.exists(folder_path) and not update:
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files
    else:
        command = f"{magicpdfbin} -p {pdf_path} -o {folder_path}"
        output = subprocess.run(command, shell=1, capture_output=1, check=0, text=True)
        print(output)
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files


if __name__ == "__main__":
    # directory = """/Users/kaoru/Library/Mobile\ Documents/iCloud\~is\~workflow\~my\~workflows/Documents/pdfpod/Jenson\ et\ al.\ -\ 2024\ -\ Transformer\ Neural\ Processes\ --\ Kernel\ Regression_EN"""
    pdffile = "pdf/0ca879f8d33bba748f84ed82e0200541.pdf"
    # print(convert_from_path(directory, True))
    print(convert_from_pdf(pdffile, True))
