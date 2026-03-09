from fastapi import FastAPI
from pydantic import BaseModel
from risk_engine import calculate_risk
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(data: TextInput):
    return calculate_risk(data.text)
