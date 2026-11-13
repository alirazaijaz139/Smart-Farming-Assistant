# Smart Farming Assistant

Smart Farming Assistant is a web app for asking farming questions, comparing crop options from soil and weather inputs, and screening tomato and potato leaf photos.

| Feature | What it does |
| --- | --- |
| Ask | Routes farming questions through LangGraph. Uses local documents for wheat, cotton, rice and pest management, and Open-Meteo for city weather. |
| Crop | Ranks three crops from nitrogen, phosphorus, potassium, temperature, humidity, soil pH and rainfall. |
| Leaf disease | Classifies a JPG or PNG as healthy, early blight or late blight for tomato or potato. |
| Accounts and farms | Uses JWT authentication and stores farm records in SQLite by default. Farm records are available through the API. |

See [ARCHITECTURE.md](ARCHITECTURE.md) for request flows and model files.

## Run locally on Windows

Open PowerShell in the project root, where `app/`, `rag/` and `web/` are located. These commands use Python 3.11 and a virtual environment named `venv`. If your project already has this environment, use it.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency list should contain separate entries for `joblib`, `torch` and `torchvision`. If your copy has `joblibtorch`, replace that entry with `joblib`.

Create a local `.env` in the project root. Use a randomly generated secret and your own Groq key.

```dotenv
SECRET_KEY=replace-with-a-random-secret
GROQ_API_KEY=replace-with-your-groq-key
```

The application reads optional `GROQ_MODEL` and `DATABASE_URL` overrides. Keep `.env`, the virtual environment and the local user database out of commits.

Verify these files exist before starting the backend:

- `ml/crop_model.joblib`
- `ml/disease_model.pt` and `ml/disease_classes.json`
- `data/farming_docs/` with the wheat, cotton, rice and pest management notes

The embedding model may need to download on the first run. Stop FastAPI before building or refreshing the index. Ingestion replaces the Chroma collection, and a running server can retain its old ID.

```powershell
.\venv\Scripts\python.exe .\rag\ingest.py
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Wait for `Application startup complete`. In a second PowerShell window, also in the project root, start the static frontend.

```powershell
.\venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory web
```

Open [the frontend](http://127.0.0.1:5500) to register and log in, or [Swagger UI](http://127.0.0.1:8000/docs) to explore the API.

## Local verification

| Check | Observed result |
| --- | --- |
| Knowledge base rebuild | 27 chunks indexed across four documents in the reported local run. |
| Wheat maturity in Faisalabad | Passed. The Ask tab returned 148 days for the Galaxy-13 field trial. |
| Cotton sowing | Passed. The Ask tab returned the general Punjab March–May window and directed the farmer to district and variety guidance. |
| Cotton pink bollworm threshold | Passed. The Ask tab returned 5 larvae per 100 bolls. |
| Rice white heads | Partial pass. The reply identified stem borer as a possibility and suggested inspecting stems, then added treatment suggestions outside the supplied notes. |
| Crop and leaf disease | Both tabs returned predictions locally. The screenshots establish that the flows responded. Field accuracy was not measured in these checks. |

The leaf model was trained on six PlantVillage classes. A model score reflects its ranking among those classes. Field photos and unsupported crops can still receive a prediction.

## Project layout

| Path | Role |
| --- | --- |
| `web/index.html` | Static HTML and JavaScript frontend |
| `app/` | FastAPI routes, authentication and farm records |
| `agent/` | LangGraph routing and tool calls |
| `ml/` | Model artifacts and training scripts |
| `data/farming_docs/` | Source notes for farming Q&A |
| `rag/ingest.py` | Builds the local Chroma index |

Public deployment is pending. The static frontend currently uses a placeholder for its hosted backend URL.
