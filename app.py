from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]
app = FastAPI(title = 'Breast Cancer Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/breast_cancer_model.joblib'))

class InputData(BaseModel):
    diagnosis: float = 569.000000
    radius_mean: float = 17.99
    texture_mean: float = 10.38
    perimeter_mean: float = 122.80
    area_mean: float = 1001.0
    smoothness_mean: float = 0.11840
    compactness_mean: float = 0.27760
    concavity_mean: float = 0.3001
    concave_points_mean: float = 0.14710
    symmetry_mean: float = 0.2419
    fractal_dimension_mean: float = 0.07864
    radius_se: float = 1.095
    texture_se: float = 0.9053
    perimeter_se: float = 8.589
    area_se: float = 153.4
    smoothness_se: float = 0.00645
    compactness_se: float = 0.04904
    concavity_se: float = 0.05373
    concave_points_se: float = 0.01587
    symmetry_se: float = 0.03022
    fractal_dimension_se: float = 0.00615
    radius_worst: float = 25.38
    texture_worst: float = 17.33
    perimeter_worst: float = 184.60
    area_worst: float = 2019.0
    smoothness_worst: float = 0.1622
    compactness_worst: float = 0.6656
    concavity_worst: float = 0.7119
    concave_points_worst: float = 0.2654
    symmetry_worst: float = 0.4601
    fractal_dimension_worst: float = 0.11890

class OutputData(BaseModel):
    score:float=0.80318881046519

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]

    return {'score':result}
