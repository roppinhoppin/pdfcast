import os
import shutil
import subprocess
import platform

system = platform.system()

if system == "Linux":
    magicpdfbin = "docker run --rm -it --gpus=all -v $(pwd)/pdf:/pdf --user $(id -u):$(id -g) mineru:latest magic-pdf"
elif system == "Darwin":
    magicpdfbin = "/Users/kaoru/anaconda3/envs/mineru/bin/magic-pdf"
else:
    raise Exception("Unsupported operating system")

# def convert_from_path(folder_path, update=False):
#     folder_path = os.path.normpath(folder_path)
#     folder = os.path.basename(folder_path)
#     pdf_file_path = folder_path + "/" + folder + ".pdf"
#     main_pdf_path = os.path.join(folder_path, "main.pdf")

#     # Copy the pdf_file_path to main.pdf
#     shutil.copy(pdf_file_path, main_pdf_path)

#     mineru_folder_path = folder_path + "/main"
#     image_folder = os.path.join(mineru_folder_path, "auto/images/")
#     # print(mineru_folder_path)
#     if os.path.exists(mineru_folder_path) and not update:
#         image_files = [image_folder + f for f in os.listdir(image_folder)]
#         return image_files
#     else:
#         cp = subprocess.run(
#             "whereis magic-pdf",
#             shell=False,
#             capture_output=True,
#             check=False,
#             text=True,
#         )
#         response = cp.stdout.strip().split()
#         if cp.returncode or len(response) != 2:  # if not 2 tokens: no tesseract-ocr
#             raise Exception("magic-pdf is not installed")
#         output = subprocess.run(
#             f"magic-pdf -p {pdf_file_path} -o {folder_path}",
#             shell=1,
#             capture_output=1,
#             check=0,
#             text=True,
#         )

#         # command = [
#         #     f"{magicpdfbin}",
#         #     "-s",
#         #     "12",
#         #     "-e",
#         #     "12",
#         #     "-p",
#         #     "main.pdf",
#         #     "-o",
#         #     "./",
#         # ]
#         # output = subprocess.run(
#         #     command,
#         #     shell=False,
#         #     capture_output=True,
#         #     check=False,
#         #     text=True,
#         #     cwd=folder_path,
#         # )
#         print(output)
#         image_files = [image_folder + f for f in os.listdir(image_folder)]
#         return image_files


def convert_from_pdf(pdf_path, update=True):
    pdf_name = os.path.basename(pdf_path)[:-4]
    folder_path = os.path.join(os.path.dirname(pdf_path), pdf_name)
    image_folder = os.path.join(folder_path, "auto/images/")
    is_done = os.path.exists(os.path.join(folder_path, f"auto/{pdf_name}.md"))
    if is_done and update is False:
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files
    else:
        command = f"{magicpdfbin} -p {pdf_path} -o {os.path.dirname(pdf_path)}"
        output = subprocess.run(
            command, shell=True, capture_output=1, check=0, text=True
        )
        # print(output)
        image_files = [image_folder + f for f in os.listdir(image_folder)]
        return image_files


if __name__ == "__main__":
    # directory = """/Users/kaoru/Library/Mobile\ Documents/iCloud\~is\~workflow\~my\~workflows/Documents/pdfpod/Jenson\ et\ al.\ -\ 2024\ -\ Transformer\ Neural\ Processes\ --\ Kernel\ Regression_EN"""
    # print(convert_from_path(directory, True))
    # pdffile = "pdf/0e26fdaa9a90f19ad8d796aca3b26351.pdf"
    # print(convert_from_pdf(pdffile))

    def iterate_and_convert(pdf_directory):
        for file in os.listdir(pdf_directory):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, file)
                print(f"Converting: {pdf_path}")
                convert_from_pdf(pdf_path, update=False)
                print(f"Done: {file}")

    pdf_directory = "pdf/"
    iterate_and_convert(pdf_directory)

    # import concurrent.futures

    # def iterate_and_convert(pdf_directory):
    #     pdf_files = []
    #     for root, dirs, files in os.walk(pdf_directory):
    #         for file in files:
    #             if file.endswith(".pdf"):
    #                 pdf_path = os.path.join(root, file)
    #                 pdf_files.append(pdf_path)

    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         executor.map(convert_from_pdf, pdf_files)

    # pdf_directory = "pdf/"
    # iterate_and_convert(pdf_directory)

