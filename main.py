## Configuration
import uvicorn
from typing import Optional
from fastapi import FastAPI

from joblib import dump, load
import numpy as np
import pickle
import sklearn

## Initialize App
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/about")
def read_root():
    return {"api_name": "test_api",
            "author": 'chitxxx'}


@app.get("/get_by_id/{id}")
def read_item(id: int, q: Optional[str] = None):
    return {"id": id, "q": q}

@app.get("/model/get_wage/{age}")
def predict_wage(age: int):
    age = int(age)
    model = load('models/model.joblib')
    # test model
    pred_wage = model.predict((np.array([[age]])))[0]
    return {'pred_wage': pred_wage}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8888)
