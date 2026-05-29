import os
from groq import Groq

def get_response(userinput: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in the project .env before creating ChatGroq")
    
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": userinput,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content