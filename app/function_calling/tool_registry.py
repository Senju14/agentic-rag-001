import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langdetect import detect
from search import retrieve_and_rerank

# ---------------- WEATHER ----------------
def weather_tool(city: str, format: str = None):
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.lower() == city.lower():
            return f"Sorry, I could not retrieve the weather for {city}."
        return text
    except Exception as e:
        return f"Weather tool error: {str(e)}"


# ---------------- WEB SEARCH ----------------
def web_search_tool(query: str):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not set. Set TAVILY_API_KEY env variable."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "search_depth": "basic", "max_results": 1}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            return data["results"][0].get("content") or data["results"][0].get("url")
        return "No results found."
    except Exception as e:
        return f"Web search error: {str(e)}"


# ---------------- SEND MAIL ----------------
def send_mail_tool(to_email: str, subject: str, body: str):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        return "Email credentials not set. Please configure SENDER_EMAIL and SENDER_PASSWORD."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return f"Email sent successfully to {to_email}"
    except Exception as e:
        return f"Send mail error: {str(e)}"


# ---------------- TRANSLATE ----------------
def translate_tool(text: str, target_lang: str = "en", source_lang: str = None):
    url = "https://api.mymemory.translated.net/get"
    try:
        if not source_lang:
            # Detect language 
            source_lang = detect(text)

        response = requests.get(
            url,
            params={"q": text, "langpair": f"{source_lang}|{target_lang}"}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("responseData", {}).get("translatedText") or "Translation failed."
    except Exception as e:
        return f"Translation error: {str(e)}"


# ---------------- DATABASE SEARCH ----------------
def search_db_tool(query: str, top_k: int = 5):
    try:
        results = retrieve_and_rerank(query, top_k=top_k)
        if not results:
            return "No matching results found in database."
        
        formatted = []
        for hit in results:
            text = hit.get("metadata", {}).get("chunk_text") or hit.get("text", "")
            semantic_score = hit.get("semantic_score", 0.0)
            rerank_score = hit.get("rerank_score", 0.0)
            formatted.append({
                "text": text,
                "semantic_score": semantic_score,
                "rerank_score": rerank_score
            })

        return "\n".join(formatted)
    except Exception as e:
        return f"Database search error: {str(e)}"


# ---------------- TOOL REGISTRY ----------------
tool_registry = {
    "weather": weather_tool,
    "web_search": web_search_tool,
    "send_mail": send_mail_tool,
    "translate": translate_tool,
    "search_db": search_db_tool,
}

custom_functions = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather for a city using wttr.in",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Tavily API and return the first result.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_mail",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_email": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email content"}
                },
                "required": ["to_email", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text into a target (auto-detects source language) language using MyMemory API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_lang": {"type": "string", "description": "Target language code (e.g. en, vi, fr, zh, ru, ja, ko)"}
                },
                "required": ["text", "target_lang"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_db",
            "description": "Search the database using semantic search and cross-encoder rerank.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}, 
                    "top_k": {"type": "integer", "description": "Number of results to return"}
                },
                "required": ["query"]
            }
        }
    },
]
