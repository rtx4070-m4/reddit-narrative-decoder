from dotenv import load_dotenv
from openai import OpenAI
import os
import sys

load_dotenv(".env", override=False)

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not configured.")
    sys.exit(1)

try:
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Reply with exactly: API connection successful."
    )
    print("API request completed.")
    print("Model:", response.model)
    print("Response:", response.output_text.strip())
except Exception as exc:
    print("API connection failed.")
    print("Error type:", type(exc).__name__)
    print("Details:", str(exc)[:1000])
    print("\nNo research examples were sent.")
    sys.exit(1)
