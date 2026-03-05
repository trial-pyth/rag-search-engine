import os
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from google import genai
    from google.genai import errors
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    errors = None  # type: ignore[assignment]

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")
provider = os.environ.get("LLM_PROVIDER", "auto").strip().lower()

model = "gemini-2.0-flash-001"
gemini_client = genai.Client(api_key=gemini_api_key) if (genai and gemini_api_key) else None
openai_client = OpenAI(api_key=openai_api_key) if (OpenAI and openai_api_key) else None

def generate_content():
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    if provider in ("openai", "auto") and openai_client is not None:
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        try:
            response = openai_client.responses.create(model=model_name, input=prompt)
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            print("Prompt tokens: 0")
            print("Response tokens: 0")
            return 0

        text = getattr(response, "output_text", None)
        if text is None:
            text = ""
        print(text)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        response_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Response tokens: {response_tokens}")
        return 0

    if provider in ("gemini", "auto") and gemini_client is not None:
        try:
            response = gemini_client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            label = "Gemini API error" if (errors and isinstance(e, errors.ClientError)) else "Gemini request failed"
            print(f"{label}: {e}")
            print("Prompt tokens: 0")
            print("Response tokens: 0")
            return 0

        print(response.text or "")
        prompt_tokens = 0
        response_tokens = 0
        if response.usage_metadata is not None:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            response_tokens = response.usage_metadata.response_token_count or 0

        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Response tokens: {response_tokens}")
        return 0

    if provider == "openai" and openai_client is None:
        print("LLM_PROVIDER=openai but OPENAI_API_KEY/openai SDK is not available.")
    elif provider == "gemini" and gemini_client is None:
        print("LLM_PROVIDER=gemini but GEMINI_API_KEY/google-genai SDK is not available.")
    else:
        print("No LLM API key found; skipping API call.")
    print("Prompt tokens: 0")
    print("Response tokens: 0")
    return 0

if __name__ == "__main__":
    raise SystemExit(generate_content())
