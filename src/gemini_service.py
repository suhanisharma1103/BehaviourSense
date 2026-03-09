import os
from google import genai


def get_gemini_analysis(text: str):

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini key missing."

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)

        # Call Gemini model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
Analyze the following message and briefly comment on:
1. Phishing intent
2. Stress indicators
3. Toxicity

Message:
{text}

Respond in 2-3 lines.
"""
        )

        # Return Gemini explanation
        return response.text

    except Exception as e:
        # Handle quota / API errors safely
        return f"Gemini error: {str(e)}"