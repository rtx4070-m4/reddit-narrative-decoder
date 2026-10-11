from dotenv import load_dotenv
import os

load_dotenv(".env", override=False)
key = os.getenv("OPENAI_API_KEY", "")

if key and key != "your_actual_api_key_here":
    print("API key detected. Value hidden.")
else:
    print("API key not configured. Check your local .env file.")

