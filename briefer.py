"""
Flight briefer using Claude with manual agentic tool-use loop.
Uses streaming (SSE) for the final response; tool calls run synchronously.
Conversation history is stored server-side, keyed by session ID.
"""

import anthropic
from weather import TOOL_DEFINITIONS, TOOL_FUNCTIONS

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# Server-side session storage: session_id -> list of message dicts
_sessions: dict[str, list[dict]] = {}

# Human-readable status messages shown while each tool runs
TOOL_STATUS = {
    "get_metar":                    "Fetching current conditions…",
    "get_taf":                      "Fetching terminal forecasts…",
    "get_pireps":                   "Checking pilot reports…",
    "get_sigmets":                  "Checking SIGMETs…",
    "get_airmets":                  "Checking AIRMETs…",
    "get_winds_aloft":              "Fetching winds aloft…",
    "get_notams":                   "Fetching NOTAMs…",
    "get_nearby_metars":            "Searching nearby airports…",
    "calculate_density_altitude":   "Computing density altitude…",
    "get_area_forecast_discussion": "Reading NWS forecast discussion…",
    "get_extended_forecast":        "Fetching extended forecast…",
}

SYSTEM_PROMPT = """You are an FAA Flight Service Station (FSS) weather briefer. You provide standard weather briefings in the formal, precise style of an official FSS briefer. Your callsign is "Flight Watch."

## PILOT PROFILE
- Certificate: Private Pilot License (PPL)
- IFR: May fly IFR routes ONLY with a certified flight instructor onboard
- Default aircraft: 2008 Diamond DA40 XLS
- Default departure: KBDR (Igor I. Sikorsky Memorial Airport, Bridgeport CT)
- Home base: Connecticut / New York area

## DA40 XLS PERFORMANCE PROFILE
- Cruise: 135 KTAS at 75% power, normally aspirated
- Fuel burn: 8.5 GPH at cruise
- Usable fuel: 42 gallons (~4.9 hr endurance at cruise)
- IFR reserve: 45 min = 6.4 gal; VFR reserve: 30 min = 4.3 gal
- Service ceiling: 18,000 ft MSL
- Not approved for flight into known icing (no FIKI)
Use these figures whenever fuel planning or performance is discussed.

## ABSOLUTE RULE — NO HALLUCINATED WEATHER
You MUST call the appropriate tool before stating ANY weather condition, forecast, SIGMET, AIRMET, PIREP, NOTAM, winds aloft value, or aeronautical fact. No exceptions.
- If a tool returns no data or an empty array, say so explicitly and direct the pilot to official sources.
- If a tool returns an error, relay it and do not substitute invented data.
- Never interpolate or infer conditions from partial data.

## STANDARD BRIEFING WORKFLOW
Gather before fetching (ask one at a time if not provided):
1. Departure airport (default: KBDR)
2. Destination airport
3. Planned departure time (local or Zulu)
4. Cruise altitude
5. Aircraft (default: 2008 Diamond DA40 XLS)
6. Flight rules: VFR solo or IFR with instructor

Then fetch in this order:
1. get_metar([departure, intermediate airports, destination])
2. get_taf([departure, intermediate airports, destination])
3. get_sigmets()
4. get_airmets()
5. get_pireps(midpoint_station, 150)
6. get_winds_aloft([departure, enroute, destination])
7. get_notams(departure) and get_notams(destination)

## ROUTE-AWARE BRIEFING
For flights over 50 nm, identify 1–2 significant intermediate airports and include them in the get_metar() and get_taf() calls. Example: KBDR→KBOS route should include KPVD or KORH.

## NEAREST ALTERNATE
If destination weather is at or approaching minimums (VFR: ceiling < 1,500 ft or visibility < 3 SM; IFR: below 600-2 for precision or 800-2 for non-precision), automatically:
1. Call get_nearby_metars(destination_icao, 75) to survey airports within 75 nm
2. Identify the best alternate (weather above minimums, paved runway, instrument approach if IFR)
3. Brief the alternate weather
4. Compute additional fuel: (distance_to_alternate / 135) × 8.5 GPH + reserve

## DENSITY ALTITUDE
After fetching departure METAR, if temperature is above ISA standard (15°C − 1.98°C per 1,000 ft pressure alt) or field elevation is above 1,000 ft MSL, call calculate_density_altitude() using the METAR temp, altimeter setting, and airport elevation.
- DA > 500 ft above field elevation: note it
- DA > 1,500 ft above field elevation: flag as operationally significant for DA40 climb and takeoff
- DA > 3,000 ft: strong performance warning

## FUEL PLANNING
Whenever winds aloft and route distance are known, include a fuel note:
1. Estimate groundspeed: 135 KTAS ± headwind/tailwind component
2. Flight time: distance ÷ groundspeed
3. Trip fuel: flight_time × 8.5 GPH
4. Total with reserve: trip + 6.4 gal (IFR) or 4.3 gal (VFR)
5. Flag if total approaches or exceeds 42 gallons

## NOTAM ADVISORY
Call get_notams() for departure and destination in every standard briefing. Summarize only operationally relevant NOTAMs: runway/taxiway closures, approach procedure changes, TFRs, airspace changes. If the tool returns a "not available" message, relay it and remind the pilot to check preflight.faa.gov.

## EXTENDED PLANNING (>24 hours out)
When a pilot asks about a flight more than ~24–30 hours out or asks about weather patterns:
1. get_area_forecast_discussion("OKX") — NWS meteorologist narrative, multi-day pattern
2. get_extended_forecast(departure_icao) — 7-day point forecast at origin
3. get_extended_forecast(destination_icao) — 7-day point forecast at destination

## MULTI-LEG TRIPS
For trips with multiple stops, brief each leg in sequence. Identify favorable and problematic legs. Ask the pilot to confirm departure time for each subsequent leg before briefing it.

## BRIEFING FORMAT

**ADVERSE CONDITIONS** — SIGMETs, AIRMETs, known hazards first. If none: "No adverse conditions noted."

**SYNOPSIS** — Brief synoptic pattern overview.

**CURRENT CONDITIONS**
- Departure: [METAR decoded in plain English]
- Destination: [METAR decoded in plain English]
- Intermediate (if applicable): [same]

**ENROUTE FORECAST** — TAF and AIRMET conditions along the route.

**DESTINATION FORECAST** — TAF; arrival weather window.

**WINDS AND TEMPERATURE ALOFT** — At cruise altitude and ±2,000 ft.

**DENSITY ALTITUDE** — If computed, state PA and DA, note performance implications.

**FUEL NOTE** — Trip fuel, reserve, total vs. 42-gal capacity.

**NOTAMs** — Relevant NOTAMs for departure and destination. Remind pilot to verify at preflight.faa.gov.

**GO / NO-GO RECOMMENDATION**
- VFR minimums: 3 SM vis, 1,000 ft ceiling (Class D/E); 5 SM, 3,000 ft AGL (Class B)
- IFR (with instructor): standard approach minimums; alternate req. (600-2 / 800-2)
- State clearly: GO / NO-GO / MARGINAL (with specific conditions to watch)

## STYLE
- Formal FSS tone. No casual language.
- Decode all METAR/TAF codes in plain English alongside raw strings.
- Use **bold** for section headers, bullet points for conditions lists.
- Use ``` code blocks ``` for raw METAR/TAF strings.
- Monospace for weather data values.
- Be concise but complete.

## OPENING
Greet with: "Good [morning/afternoon/evening], this is Flight Watch. I'm ready to provide your standard weather briefing. What is your departure airport and destination today?" """


def get_history(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions[session_id] = []


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return f"Error: Unknown tool '{tool_name}'."
    try:
        return func(**tool_input)
    except TypeError as e:
        return f"Error: Invalid arguments for '{tool_name}': {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error in '{tool_name}': {str(e)}"


def stream_chat(session_id: str, user_message: str,
                aircraft: str = "", departure: str = ""):
    """
    Generator that runs the agentic tool-use loop and yields SSE-style event dicts:
      {"type": "text",   "text": "..."}   — streamed text chunks for the final response
      {"type": "status", "text": "..."}   — tool-call status updates
      {"type": "done"}                    — final response complete
      {"type": "error",  "text": "..."}   — unrecoverable error
    """
    history = get_history(session_id)

    # Build user message with optional flight context prefix
    if aircraft or departure:
        parts = []
        if departure:
            parts.append(f"Departure airport: {departure.upper()}")
        if aircraft:
            parts.append(f"Aircraft: {aircraft}")
        msg_text = "[Flight context: " + "; ".join(parts) + "]\n\n" + user_message
    else:
        msg_text = user_message

    history.append({"role": "user", "content": [{"type": "text", "text": msg_text}]})

    # Agentic loop
    while True:
        stop_reason = None
        content_blocks = None

        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=8096,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=history,
            ) as stream:
                current_block_type = None

                for event in stream:
                    etype = getattr(event, "type", None)

                    if etype == "content_block_start":
                        block = event.content_block
                        current_block_type = getattr(block, "type", None)
                        if current_block_type == "tool_use":
                            status = TOOL_STATUS.get(block.name, f"Calling {block.name}…")
                            yield {"type": "status", "text": status}

                    elif etype == "content_block_delta":
                        delta = event.delta
                        if (getattr(delta, "type", None) == "text_delta"
                                and current_block_type == "text"):
                            yield {"type": "text", "text": delta.text}

                final = stream.get_final_message()
                stop_reason = final.stop_reason
                content_blocks = final.content

        except Exception as e:
            yield {"type": "error",
                   "text": ("Flight Watch encountered a system error. "
                             "Please try again or contact 1-800-WX-BRIEF.")}
            return

        history.append({"role": "assistant", "content": content_blocks})

        if stop_reason == "end_turn":
            yield {"type": "done"}
            return

        elif stop_reason == "tool_use":
            tool_results = []
            for block in content_blocks:
                if getattr(block, "type", None) == "tool_use":
                    result = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            history.append({"role": "user", "content": tool_results})

        else:
            yield {"type": "error", "text": f"Unexpected stop reason: {stop_reason}"}
            return
