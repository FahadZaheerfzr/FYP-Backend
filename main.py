from fastapi import FastAPI
from keras.models import load_model

app = FastAPI()

model = load_model('model/Conv-LSTM.h5')

print(model.summary())
@app.get("/")
def read_root():
    result = model.summary()
    return {"result":result}