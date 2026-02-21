"""
Aviation weather data fetched from aviationweather.gov API.
All functions return a JSON string or a plain-English error message.
Functions never raise exceptions to the caller.
"""

import json
import requests

BASE_URL = "https://aviationweather.gov/api/data"
NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "FlightBriefing/1.0 (personal aviation planning tool)",
    "Accept": "application/geo+json",
}
TIMEOUT = 15  # seconds


def _get(endpoint: str, params: dict) -> str:
    """Make a GET request and return raw response text or an error string."""
    try:
        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return json.dumps([])
        return text
    except requests.exceptions.Timeout:
        return f"Error: Request to {endpoint} timed out after {TIMEOUT} seconds."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} from {endpoint}: {e.response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to aviationweather.gov. Check network connectivity."
    except Exception as e:
        return f"Error: Unexpected error fetching {endpoint}: {str(e)}"


def get_metar(stations: list[str]) -> str:
    """
    Fetch METARs for a list of ICAO station identifiers.

    Args:
        stations: List of ICAO airport codes, e.g. ["KBDR", "KHVN", "KJFK"]

    Returns:
        JSON string of METAR records, or a plain-English error message.
    """
    if not stations:
        return "Error: No station identifiers provided for METAR request."
    ids = ",".join(s.upper().strip() for s in stations)
    return _get("metar", {"ids": ids, "format": "json", "hours": 2})


def get_taf(stations: list[str]) -> str:
    """
    Fetch TAFs for a list of ICAO station identifiers.

    Args:
        stations: List of ICAO airport codes, e.g. ["KBDR", "KHVN"]

    Returns:
        JSON string of TAF records, or a plain-English error message.
    """
    if not stations:
        return "Error: No station identifiers provided for TAF request."
    ids = ",".join(s.upper().strip() for s in stations)
    return _get("taf", {"ids": ids, "format": "json"})


def get_pireps(station: str, distance_nm: int = 100) -> str:
    """
    Fetch PIREPs within a specified distance of a station.

    Args:
        station: ICAO code of the center station, e.g. "KBDR"
        distance_nm: Search radius in nautical miles (default 100)

    Returns:
        JSON string of PIREP records, or a plain-English error message.
    """
    if not station:
        return "Error: No station identifier provided for PIREP request."
    return _get("pirep", {
        "id": station.upper().strip(),
        "format": "json",
        "distance": distance_nm
    })


def get_sigmets() -> str:
    """
    Fetch all active international and domestic SIGMETs.

    Returns:
        JSON string of SIGMET records, or a plain-English error message.
    """
    return _get("sigmet", {"format": "json"})


def get_airmets() -> str:
    """
    Fetch all active AIRMETs (Sierra, Tango, Zulu).

    Returns:
        JSON string of AIRMET records, or a plain-English error message.
    """
    return _get("airmet", {"format": "json"})


def get_winds_aloft(stations: list[str]) -> str:
    """
    Fetch winds aloft forecast for a list of stations (low-level, 6-hour forecast).

    Args:
        stations: List of station identifiers, e.g. ["BDR", "HVN", "GON"]

    Returns:
        JSON string of winds aloft data, or a plain-English error message.
    """
    if not stations:
        return "Error: No station identifiers provided for winds aloft request."
    ids = ",".join(s.upper().strip() for s in stations)
    return _get("windtemp", {
        "region": "all",
        "level": "low",
        "fcst": "06",
        "format": "json",
        "ids": ids
    })


def _nws_get(url: str):
    """GET from the NWS API; returns parsed dict or an error string."""
    try:
        resp = requests.get(url, headers=NWS_HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return "Error: NWS API request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} from NWS API: {e.response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to api.weather.gov. Check network connectivity."
    except Exception as e:
        return f"Error: Unexpected error from NWS API: {str(e)}"


def get_area_forecast_discussion(office: str = "OKX") -> str:
    """
    Fetch the latest NWS Area Forecast Discussion (AFD) for a forecast office.

    The AFD is written by NWS meteorologists and covers the multi-day synoptic
    pattern, including frontal timing, precipitation chances, and aviation-relevant
    hazards several days out. Essential for planning flights beyond 24 hours.

    Args:
        office: NWS forecast office ID, e.g. "OKX" (Upton NY, covers CT/NY),
                "BOX" (Boston MA), "PHI" (Mount Holly NJ). Default: "OKX".

    Returns:
        Full text of the most recent AFD, or a plain-English error message.
    """
    office = office.upper().strip()
    product_list = _nws_get(f"{NWS_BASE}/products/types/AFD/locations/{office}")
    if isinstance(product_list, str):
        return product_list
    graph = product_list.get("@graph", [])
    if not graph:
        return f"No Area Forecast Discussion found for NWS office {office}."
    product_id = graph[0]["id"]
    product = _nws_get(f"{NWS_BASE}/products/{product_id}")
    if isinstance(product, str):
        return product
    text = product.get("productText")
    if not text:
        return f"AFD retrieved for {office} but contained no text."
    return text


def get_extended_forecast(icao: str) -> str:
    """
    Fetch the NWS 7-day extended point forecast for an airport location.

    Looks up the airport's coordinates from its METAR record, then queries
    the NWS gridpoint forecast. Returns plain-language forecast periods
    (day/night) covering roughly 7 days. Useful for identifying favorable
    weather windows for multi-day trip planning.

    Args:
        icao: ICAO airport code used to look up location, e.g. "KBDR" or "KHVN"

    Returns:
        JSON array of NWS forecast periods, or a plain-English error message.
    """
    icao = icao.upper().strip()

    # Step 1 — get lat/lon from a recent METAR
    metar_raw = _get("metar", {"ids": icao, "format": "json", "hours": 1})
    try:
        metars = json.loads(metar_raw)
    except Exception:
        return f"Error: Could not parse METAR data to obtain coordinates for {icao}."
    if not metars:
        return f"Error: No METAR records found for {icao}; cannot determine location coordinates."
    lat = metars[0].get("lat")
    lon = metars[0].get("lon")
    if lat is None or lon is None:
        return f"Error: METAR for {icao} did not include coordinates."

    # Step 2 — resolve NWS gridpoint for that lat/lon
    points = _nws_get(f"{NWS_BASE}/points/{lat},{lon}")
    if isinstance(points, str):
        return points
    forecast_url = points.get("properties", {}).get("forecast")
    if not forecast_url:
        return f"Error: NWS did not return a forecast URL for {icao} ({lat}, {lon})."

    # Step 3 — fetch 7-day forecast
    forecast = _nws_get(forecast_url)
    if isinstance(forecast, str):
        return forecast
    periods = forecast.get("properties", {}).get("periods", [])
    if not periods:
        return f"No extended forecast periods returned for {icao}."
    return json.dumps(periods)


# Tool definitions for Claude
TOOL_DEFINITIONS = [
    {
        "name": "get_metar",
        "description": (
            "Fetch current METAR (Meteorological Aerodrome Report) observations for one or more "
            "ICAO airport stations. Returns raw JSON from aviationweather.gov including sky cover, "
            "visibility, temperature, dewpoint, altimeter, and wind. Always call this before "
            "stating any current weather conditions at an airport."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ICAO airport codes, e.g. [\"KBDR\", \"KHVN\", \"KJFK\"]"
                }
            },
            "required": ["stations"]
        }
    },
    {
        "name": "get_taf",
        "description": (
            "Fetch Terminal Aerodrome Forecasts (TAF) for one or more ICAO airport stations. "
            "Returns forecast wind, visibility, sky cover, and weather for the next 24-30 hours. "
            "Always call this before stating any forecast conditions at an airport."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ICAO airport codes, e.g. [\"KBDR\", \"KHVN\"]"
                }
            },
            "required": ["stations"]
        }
    },
    {
        "name": "get_pireps",
        "description": (
            "Fetch Pilot Weather Reports (PIREPs) within a specified distance of a station. "
            "Returns reports of actual in-flight conditions including turbulence, icing, and "
            "cloud tops from pilots who recently flew in the area. Call this before discussing "
            "enroute conditions, turbulence, or icing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "station": {
                    "type": "string",
                    "description": "ICAO code of the center station, e.g. \"KBDR\""
                },
                "distance_nm": {
                    "type": "integer",
                    "description": "Search radius in nautical miles (default 100)",
                    "default": 100
                }
            },
            "required": ["station"]
        }
    },
    {
        "name": "get_sigmets",
        "description": (
            "Fetch all currently active SIGMETs (Significant Meteorological Information). "
            "SIGMETs cover severe or extreme turbulence, severe icing not associated with "
            "thunderstorms, widespread dust or sandstorms, and volcanic ash. Always call this "
            "as part of any complete weather briefing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_airmets",
        "description": (
            "Fetch all currently active AIRMETs (Airmen's Meteorological Information). "
            "AIRMETs include Sierra (IFR conditions / mountain obscuration), Tango (turbulence, "
            "strong surface winds), and Zulu (icing, freezing level). Always call this as part "
            "of any complete weather briefing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_winds_aloft",
        "description": (
            "Fetch winds aloft forecast data for a list of stations. Returns forecast wind "
            "direction, speed, and temperature at various altitudes (low-level, 6-hour forecast). "
            "Call this when the pilot needs winds aloft information for cruise altitude planning "
            "or fuel burn calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of station identifiers, e.g. [\"BDR\", \"HVN\", \"GON\"]"
                }
            },
            "required": ["stations"]
        }
    },
    {
        "name": "get_area_forecast_discussion",
        "description": (
            "Fetch the latest NWS Area Forecast Discussion (AFD) — a detailed meteorologist-written "
            "narrative covering the multi-day synoptic weather pattern, frontal timing, precipitation "
            "chances, and aviation hazards for the region. This is the single best source for "
            "understanding weather patterns 1–5 days out. Use this for any planning question beyond "
            "the TAF horizon (>30 hours). Default office OKX covers Connecticut and New York. "
            "Always call this before giving extended planning advice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "office": {
                    "type": "string",
                    "description": (
                        "NWS forecast office ID. OKX = Upton NY (covers CT/NY, default), "
                        "BOX = Boston MA (covers eastern New England), "
                        "PHI = Mount Holly NJ (covers NJ/DE/PA). Default: 'OKX'."
                    ),
                    "default": "OKX"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_extended_forecast",
        "description": (
            "Fetch the NWS 7-day plain-language extended forecast for an airport location. "
            "Looks up the airport coordinates from its METAR record, then retrieves the NWS "
            "gridpoint forecast covering roughly 7 days in day/night periods. Useful for "
            "identifying favorable weather windows and go/no-go decisions for trips several "
            "days in advance. Complements the AFD with location-specific detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "icao": {
                    "type": "string",
                    "description": "ICAO airport code to forecast for, e.g. \"KBDR\" or \"KHVN\""
                }
            },
            "required": ["icao"]
        }
    }
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_metar": get_metar,
    "get_taf": get_taf,
    "get_pireps": get_pireps,
    "get_sigmets": get_sigmets,
    "get_airmets": get_airmets,
    "get_winds_aloft": get_winds_aloft,
    "get_area_forecast_discussion": get_area_forecast_discussion,
    "get_extended_forecast": get_extended_forecast,
}
