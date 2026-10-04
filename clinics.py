"""Find dermatologists and clinics near a place, using OpenStreetMap.

Nominatim turns a typed place into coordinates; Overpass returns tagged
healthcare sites around them. Neither needs an API key. Nothing is logged.
"""

import math
import re

import httpx

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Nominatim's usage policy requires an identifying User-Agent.
HEADERS = {"User-Agent": "Dermassist/0.1 (research prototype)"}

DERM_RADIUS_M = 25_000    # dermatologists are rarer, so search wider
GENERAL_RADIUS_M = 5_000  # GP practices, clinics and hospitals
MAX_DERM = 8
MAX_TOTAL = 12

# Coordinates are rounded to ~1 km before leaving the app.
COORD_DECIMALS = 2

_COORDS = re.compile(r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$")


class ClinicSearchError(Exception):
    """A place could not be found or a map service failed; the message is user-facing."""


def parse_coords(text):
    """Return (lat, lon) if text is "lat, lon", else None."""
    m = _COORDS.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None


def geocode(query):
    """Return (lat, lon, label) for a typed place or "lat, lon" string."""
    coords = parse_coords(query)
    if coords:
        return (*coords, "your location")
    try:
        r = httpx.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": 1},
                      headers=HEADERS, timeout=10)
        r.raise_for_status()
        hits = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ClinicSearchError("The place search is not responding. Try again in a moment.") from e
    if not hits:
        raise ClinicSearchError(f"Could not find “{query}”. Try a city, suburb or postcode.")
    hit = hits[0]
    return float(hit["lat"]), float(hit["lon"]), hit.get("display_name") or query


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def _kind(tags):
    if "dermatolog" in tags.get("healthcare:speciality", "").lower():
        return "Dermatology"
    values = {tags.get("amenity"), tags.get("healthcare")}
    if "hospital" in values:
        return "Hospital"
    if values & {"doctors", "doctor"}:
        return "Doctor"
    return "Clinic"


def _safe_url(url):
    url = (url or "").strip()
    return url if url.startswith(("https://", "http://")) else None


def to_place(element, lat0, lon0):
    """Normalise one Overpass element into a place dict, or None if unusable."""
    tags = element.get("tags") or {}
    name = tags.get("name")
    centre = element.get("center") or {}
    lat = element.get("lat", centre.get("lat"))
    lon = element.get("lon", centre.get("lon"))
    if not name or lat is None or lon is None:
        return None
    street = " ".join(filter(None, [tags.get("addr:housenumber"), tags.get("addr:street")]))
    town = tags.get("addr:suburb") or tags.get("addr:city")
    return {
        "id": f'{element.get("type")}/{element.get("id")}',
        "name": name,
        "kind": _kind(tags),
        "lat": lat,
        "lon": lon,
        "km": distance_km(lat0, lon0, lat, lon),
        "address": ", ".join(filter(None, [street, town])),
        "phone": tags.get("phone") or tags.get("contact:phone"),
        "website": _safe_url(tags.get("website") or tags.get("contact:website")),
    }


def select_places(elements, lat0, lon0):
    """Dermatology first (nearest up to MAX_DERM), then the nearest other sites, MAX_TOTAL in all."""
    places = {}
    for el in elements:
        place = to_place(el, lat0, lon0)
        if place:
            places[place["id"]] = place
    # The same site is often mapped twice (a point and its building): keep the first within 300 m.
    unique = []
    for p in sorted(places.values(), key=lambda p: (p["kind"] != "Dermatology", p["km"])):
        if not any(q["name"].lower() == p["name"].lower()
                   and distance_km(q["lat"], q["lon"], p["lat"], p["lon"]) < 0.3 for q in unique):
            unique.append(p)
    derm = [p for p in unique if p["kind"] == "Dermatology"]
    other = [p for p in unique if p["kind"] != "Dermatology"]
    derm = derm[:MAX_DERM]
    return derm + other[:MAX_TOTAL - len(derm)]


def find_nearby(lat, lon):
    """Return a list of place dicts near (lat, lon), dermatology first."""
    lat, lon = round(lat, COORD_DECIMALS), round(lon, COORD_DECIMALS)
    query = f"""
[out:json][timeout:25];
nwr["healthcare:speciality"~"dermatolog",i](around:{DERM_RADIUS_M},{lat},{lon});
out center tags 60;
(
  nwr["amenity"~"^(doctors|clinic|hospital)$"](around:{GENERAL_RADIUS_M},{lat},{lon});
  nwr["healthcare"~"^(doctor|clinic|hospital)$"](around:{GENERAL_RADIUS_M},{lat},{lon});
);
out center tags 150;
"""
    try:
        r = httpx.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=30)
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except (httpx.HTTPError, ValueError) as e:
        raise ClinicSearchError("The clinic search is busy or not responding. Try again in a moment.") from e
    return select_places(elements, lat, lon)
