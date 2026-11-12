import httpx
import pandas as pd

from app.routers.crop import FEATURES, model as crop_model
from app.routers.rag import collection, embedder


def recommend_crop(params: dict) -> dict:
    missing = [f for f in FEATURES if params.get(f) is None]
    if missing:
        return {"error": f"Missing values: {', '.join(missing)}"}
    row = pd.DataFrame([[float(params[f]) for f in FEATURES]], columns=FEATURES)
    probs = crop_model.predict_proba(row)[0]
    top3 = sorted(zip(crop_model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
    return {"top_3": [{"crop": c, "confidence": round(float(p), 3)} for c, p in top3]}


def search_docs(question: str) -> dict:
    if collection.count() == 0:
        return {"error": "Knowledge base is empty"}
    emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=emb, n_results=4)
    return {
        "chunks": res["documents"][0],
        "sources": sorted({m["source"] for m in res["metadatas"][0]}),
    }


def get_weather(city: str) -> dict:
    if not city:
        return {"error": "Please tell me which city"}
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15,
        ).json()
        if not geo.get("results"):
            return {"error": f"City not found: {city}"}
        place = geo["results"][0]
        w = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=15,
        ).json()
        d = w["daily"]
        days = [
            {
                "date": d["time"][i],
                "max_c": d["temperature_2m_max"][i],
                "min_c": d["temperature_2m_min"][i],
                "rain_mm": d["precipitation_sum"][i],
            }
            for i in range(len(d["time"]))
        ]
        return {"city": place["name"], "country": place.get("country"), "forecast": days}
    except httpx.HTTPError as e:
        return {"error": f"Weather service failed: {e}"}