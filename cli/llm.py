import os
from dotenv import load_dotenv
from lib.search_utils import PROMPT_PATH

load_dotenv()
ai_gateway_api_key = os.environ.get("AI_GATEWAY_API_KEY")
if not ai_gateway_api_key:
    raise RuntimeError("AI_GATEWAY_API_KEY environment variable not set")


model = "gpt-4o-mini-search-preview"
from openai import OpenAI

client = OpenAI(
    api_key=ai_gateway_api_key,
    base_url="https://ai-gateway.vercel.sh/v1",
)

# Gemini provider (disabled)
# from google import genai
# gemini_api_key = os.environ.get("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise RuntimeError("GEMINI_API_KEY environment variable not set")
# model = "gemini-2.0-flash-001"
# client = genai.Client(api_key=gemini_api_key)

def generate_content(prompt, query):
    rendered_prompt = prompt.format(query=query)
    response = client.responses.create(model=model, input=rendered_prompt)
    return response.output_text

def correct_spelling(query):
    with open(PROMPT_PATH / "spelling.md", "r", encoding="utf-8") as f:
        prompt = f.read()
    return generate_content(prompt, query)

def rewrite_query(query):
    with open(PROMPT_PATH / "rewrite.md", "r", encoding="utf-8") as f:
        prompt = f.read()
    return generate_content(prompt, query)
