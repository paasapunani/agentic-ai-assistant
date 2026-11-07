"""External functions the agent can call via LangChain tool calling."""
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from langchain_core.tools import tool

HEADERS = {"User-Agent": "agentic-assistant/1.0 (learning project)"}

# ---------- safe calculator ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '(12.5 * 4) / 3 + 2**5'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time for an IANA timezone, e.g. 'Asia/Kolkata'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime(f"%A, %d %B %Y, %H:%M:%S ({timezone})")
    except Exception as e:
        return f"Time error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city name (uses the free Open-Meteo API)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, headers=HEADERS, timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find city: {city}"
        place = geo["results"][0]
        cur = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            },
            headers=HEADERS, timeout=10,
        ).json()["current"]
        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{cur['temperature_2m']}°C, humidity {cur['relative_humidity_2m']}%, "
            f"wind {cur['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather error: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary."""
    try:
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            headers=HEADERS, timeout=10,
        ).json()
        hits = search["query"]["search"]
        if not hits:
            return f"No Wikipedia results for: {query}"
        title = hits[0]["title"]
        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            headers=HEADERS, timeout=10,
        ).json()
        return f"{title}: {summary.get('extract', 'No summary available.')}"
    except Exception as e:
        return f"Wikipedia error: {e}"


TOOLS = [calculator, get_current_time, get_weather, wikipedia_search]
