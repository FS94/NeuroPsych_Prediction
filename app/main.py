import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class DataInput(BaseModel):
    Age: Optional[float] = None
    Edu: Optional[float] = None
    Pho_Fluency: Optional[float] = None
    Sem_Fluency: Optional[float] = None
    Rey_Copy: Optional[float] = None
    Rey_DelayedRecall: Optional[float] = None
    RavensPM: Optional[float] = None
    TMT_A: Optional[float] = None
    TMT_B: Optional[float] = None


class PredictionOut(BaseModel):
    diagnostic: str

@app.get("/")
def home():
    return {'health_check': 'OK',
            'model_version': model_version}
    
    
@app.post("/predict", response_model=PredictionOut)
def predict(payload: DataInput):
    input_df = pd.DataFrame([payload.dict()])
    
    diagnostic = predict_pipeline(input_df)
    return {'diagnostic': diagnostic}  # needs to match names

