from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from minio import Minio

import os
from dotenv import load_dotenv
load_dotenv()

ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')

app = FastAPI()

# with open('src/model.pkl', 'rb') as f:
#     model = pickle.load(f)

# upload from s3 (minio)
# model = pickle.load(urllib.request.urlopen("http://s3:9000/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDk5L2NyZWRpdC1tb2RlbC9tb2RlbC5wa2w_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD0zSDZaQ0QyU1hQMFlFNUhJUFYzSiUyRjIwMjUwNTA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUwNFQxNTI3MjJaJlgtQW16LUV4cGlyZXM9NDMyMDAmWC1BbXotU2VjdXJpdHktVG9rZW49ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmhZMk5sYzNOTFpYa2lPaUl6U0RaYVEwUXlVMWhRTUZsRk5VaEpVRll6U2lJc0ltVjRjQ0k2TVRjME5qTTRNemt3TlN3aWNHRnlaVzUwSWpvaWEyOXVjMmx1TVRrNE9DSjkuSHNuUXZFcEVzNWN1NUY5Ym5BZ2pNTVQ4dlBGTERjZlhGYmR6bEwyMkUwakszRkNUSVhWaUw4aVVLUkhIMXh6WWF3emtjdXJWdUhuRWJPOHhjamxSRlEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnZlcnNpb25JZD1udWxsJlgtQW16LVNpZ25hdHVyZT1hNzc0MjgyYzFkNTk4ODhiODRmOTZmNzhjMzcxNWVlOTE5MjJkZTViNmZiMTkwOGU4YWNlMWJlM2FlYmY5ZjMz"))

client = Minio("s3:9099",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
)
model = pickle.load(client.get_object('credit-model', 'model.pkl'))

class PredictionInput(BaseModel):
    age: int
    sex: str
    job: str
    housing: str
    credit_amount: float
    duration: int


@app.get('/health')
def health_check():
    return {"status": 'healthy'}

@app.post('/predict')
def predict(input_data: PredictionInput):
    data_to_predict = pd.DataFrame({
        'age': [input_data.age],
        'sex': [input_data.sex],
        'job': [input_data.job],
        'housing': [input_data.housing],
        'credit_amount': [input_data.credit_amount],
        'duration': [input_data.duration]
    })

    prediction = model.predict(data_to_predict)[0]
    result = 'Good client' if prediction == 1 else 'Bad client'

    return {'prediction': result}