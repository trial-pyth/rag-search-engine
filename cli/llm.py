import os
import math
import signal
from dotenv import load_dotenv
from lib.search_utils import PROMPT_PATH

load_dotenv()
ai_gateway_api_key = os.environ.get("AI_GATEWAY_API_KEY")
if not ai_gateway_api_key:
    raise RuntimeError("AI_GATEWAY_API_KEY environment variable not set")

timeout_seconds = float(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))

model = "gpt-4o-mini-search-preview"
from openai import OpenAI

client = OpenAI(
    api_key=ai_gateway_api_key,
    base_url="https://ai-gateway.vercel.sh/v1",
    timeout=timeout_seconds,
    max_retries=0,
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
    def _alarm_handler(signum, frame):
        raise TimeoutError(f"LLM request exceeded {timeout_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(max(1, int(math.ceil(timeout_seconds))))
    try:
        print("Calling LLM provider...", flush=True)
        response = client.responses.create(model=model, input=rendered_prompt)
        return response.output_text
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def augment_prompt(query, type):
    with open(PROMPT_PATH /f"{type}.md", "r", encoding="utf-8") as f:
        prompt = f.read()
    return generate_content(prompt, query)

def correct_spelling(query):
    return augment_prompt(query, "spelling")

def rewrite_query(query):
    return augment_prompt(query, "rewrite")

def expand_query(query):
    return augment_prompt(query, "expand")
    
