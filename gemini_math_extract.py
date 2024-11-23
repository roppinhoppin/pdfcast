import os
import subprocess
import google.generativeai as genai
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("--postnum", "-p", type=str, required=True)

# parser.add_argument("--update", type=bool, required=False, default=False)
parser.add_argument("--update", action="store_true")
args = parser.parse_args()
postnum = args.postnum
update = args.update

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

flash_model_info = genai.get_model('models/gemini-1.5-flash')
pro_model_info = genai.get_model('models/gemini-1.5-pro')
flash_model = genai.GenerativeModel('models/gemini-1.5-flash')
pro_model = genai.GenerativeModel('models/gemini-1.5-pro')

markdown_content = ""
if os.path.exists(f"pdf/{postnum}/auto/{postnum}.md"):
    with open(f"pdf/{postnum}/auto/{postnum}.md", "r") as f:
        markdown_content = f.read()
else:
    print(f"File pdf/{postnum}/auto/{postnum}.md does not exist.")
    exit(1)

prompt = f"""
You are professional mathematician. 
Your job is to extract all mathematical statements (like definitions, propositions, lemmas, theorems etc...) in a given markdown source and list them all without modifying their statements but you need to produce them in a proper markdown format so that it can be rendered by KaTeX (e.g. fix "\b" etc...) and fix typographical errors caused by OCR. 
Just list these statements and do not add conclusions, titles, comments, or other unnecessary information.

Source: {markdown_content}
"""

num_token = flash_model.count_tokens(prompt).total_tokens
if num_token > pro_model_info.input_token_limit:
    model = None
elif num_token > flash_model_info.input_token_limit:
    model = pro_model
else:
    model = flash_model

if model is None:
    print(f"Token limit exceeded: {num_token}")
else:
    print(f"Using {model} model with {num_token} tokens")

if os.path.exists(f"pdf/{postnum}/auto/math_extract.md") and update is False:
    print(f"Already done: {postnum}")
else:
    response = model.generate_content(prompt)

    with open(f"math/{postnum}.md", "w") as f:
        f.write(response.text)

# print(response.choices[0].message)

