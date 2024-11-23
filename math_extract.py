import os
import subprocess
import google.generativeai as genai
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("--postnum", type=str, required=True)
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

system_prompt = """
You are professional mathematician. 
Your job is to extract all mathematical statements (like definitions, propositions, lemmas, theorems etc...) in a given LaTeX source code and list them all without modifying their statements but you need to produce them in a proper markdown format so that it can be rendered by KaTeX. 
Just list these statements and do not add conclusions, titles, comments, or other unnecessary information.
"""
markdown_content = ""
with open(f"pdf/{postnum}/auto/{postnum}.md", "r") as f:
    markdown_content = f.read()

if os.path.exists(f"pdf/{postnum}/auto/math_extract.md") and update is False:
    print(f"Already done: {postnum}")
else:
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        n=1,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": markdown_content
            }
        ]
    )

    with open(f"pdf/{postnum}/auto/math_extract.md", "w") as f:
        # print(response.choices)
        f.write(response.choices[0].message.content)

# print(response.choices[0].message)

