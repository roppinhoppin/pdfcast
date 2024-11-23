import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--postnum", "-p", type=str, required=True)
# parser.add_argument("--update", type=bool, required=False, default=False)
parser.add_argument("--update", action="store_true")
args = parser.parse_args()
postnum = args.postnum
update = args.update

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

system_prompt = """
You are professional mathematician. Your job is to extract important mathematical statements (not just formulas) like definitions, theorems etc... in a given markdown source and list them all but you need to produce them in a proper markdown format  so that it can be rendered by mathjax (e.g. delimiters should be "$$" instead of "$" and fix "\b", "{{" etc...) and fix typographical errors caused by OCR. Just list these statements and do not add conclusions, titles, comments, or other unnecessary information. Markdown code below is an example of an ideal output.
"""
markdown_content = ""
with open(f"magic-pdf/{postnum}/auto/{postnum}.md", "r") as f:
    markdown_content = f.read()

if os.path.exists(f"math/{postnum}.md") and update is False:
    print(f"Already done: {postnum}")
else:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": markdown_content},
        ],
    )

    with open(f"math/{postnum}.md", "w") as f:
        # print(response.choices)
        f.write(response.choices[0].message.content)

# print(response.choices[0].message)
