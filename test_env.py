import os
from dotenv import load_dotenv
import duckdb
import talib
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Environment Loaded Successfully!")
print(f"Alpaca Key Found: {'Yes' if os.getenv('ALPACA_API_KEY') else 'No'}")
print(f"DuckDB Version: {duckdb.__version__}")
print(f"TA-Lib Version: {talib.__version__}")
