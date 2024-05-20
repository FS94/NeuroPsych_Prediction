import pickle as pk
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/svm_model-{__version__}.pkl", "rb") as f:
    model = pk.load(f)
    
    
classes = ['AD', 'Healthy', 'VaE']

def predict_pipeline(input):
    pred = model.predict(input)
    return str(classes[pred[0]])
    
