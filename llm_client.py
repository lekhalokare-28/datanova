import os
import time
from openai import OpenAI, APIConnectionError

MODEL_NAME = "openai/gpt-oss-120b"

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def call_llm(prompt: str, model: str = MODEL_NAME, max_retries: int = 3) -> str:
    """Calls the LLM with automatic retry on transient network errors
    (e.g. flaky WiFi) instead of crashing the whole pipeline on one bad
    connection moment - this matters a lot for live demos."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except APIConnectionError as e:
            last_error = e
            wait_time = attempt * 2
            print(f"  [Network hiccup] Attempt {attempt}/{max_retries} failed, "
                  f"retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to reach the LLM API after {max_retries} attempts. "
        f"Check your internet connection. Last error: {last_error}"
    )
