import json
import time
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

def individual_rerank(query, documents):
    with open(PROMPT_PATH/'individual_rerank.md') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        _prompt = prompt.format(
            query=query,
            title= doc['title'],
            description= doc['description']
        )
        response = client.responses.create(model=model, input=_prompt)
        clean_response_text = (response.output_text or "").strip()
        try:
            clean_response_text = int(clean_response_text)
        except:
            print(f"Failed to case {response.output_text} to int for {doc['title']}")
            clean_response_text = 0
        results.append({**doc, 'rerank_response':clean_response_text})
        time.sleep(3)
        print({response.output_text, int(response.output_text)})

    results = sorted(results, key=lambda x: x['rerank_response'], reverse=True)
    return results


def batch_rerank(query, documents):
    with open(PROMPT_PATH/'batch_rerank.md') as f:
        prompt = f.read()
    _mtemp = '''<movie id={idx}=>{title}:\n{desc}\n</movie>\n'''    
    doc_list_str =  ''
    for idx, doc in enumerate(documents):
        doc_list_str == _mtemp.format(idx=idx, title=doc['title'], desc=doc['description'])
    _prompt = prompt.format(
        query=query,
        doc_list_str=doc_list_str)
    response = client.responses.create(model=model, input = _prompt)
    print(response.text)
    response_parsed = json.loads(response.output_text)
    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, 'rerank_score':response_parsed.index(idx)})
    results = sorted(results, key = lambda x: x['rerank_score'], reverse=True)
    return results  
