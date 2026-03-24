import google.generativeai as genai
import os

def troubleshooting():
    with open(".streamlit/secrets.toml", "r") as f:
        content = f.read()
        key = content.split('GEMINI_API_KEY = "')[1].split('"')[0]
        
    genai.configure(api_key=key)
    
    with open("model_list.txt", "w") as out:
        try:
            for m in genai.list_models():
                out.write(f"{m.name}\n")
        except Exception as e:
            out.write(f"Error: {e}")

if __name__ == "__main__":
    troubleshooting()
