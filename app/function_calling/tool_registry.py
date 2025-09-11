import requests
import os

def weather_tool(city: str, format: str = None):
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.lower() == city.lower() or len(text) < 5:
            return f"Sorry, I could not retrieve the weather for {city}."
        return text
    except Exception as e:
        return f"Weather tool error: {str(e)}"

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

tool_registry = {
    "weather": weather_tool,
    "web_search": web_search_tool,
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
]
