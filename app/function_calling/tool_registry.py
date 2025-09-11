import requests

def weather_tool(city: str, format: str = None):
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text.strip()
        # Nếu text là rỗng hoặc chỉ là tên thành phố, trả về lỗi
        if not text or text.lower() == city.lower() or len(text) < 5:
            return f"Sorry, I could not retrieve the weather for {city}."
        return text
    except Exception as e:
        return f"Weather tool error: {str(e)}"

tool_registry = {
    "weather": weather_tool,
    # Thêm tool mới ở đây
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
    # Thêm function cho tool mới ở đây
]
