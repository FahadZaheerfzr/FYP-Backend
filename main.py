from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from keras.models import load_model
import pickle
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = FastAPI()

model = load_model('model/Conv-LSTM.h5', compile=False)
model.compile()

def to_onehot(yy):
    result = []
    for num in yy:
        result.append(round(num))
    return result

@app.get("/")
def classify():
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    file = open("singleton_final.dat",'rb')
    Xd = pickle.load(file)
    new_data = np.array(Xd)
    new_data = new_data[np.newaxis,...]
    new_data.shape
    result = model.predict(new_data)
    final_result = to_onehot(result[0])
    idx = final_result.index(1)
    return {"modulation:type":mods[idx]}

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    # Check if the uploaded file is a .dat file
    if not file.filename.endswith('.dat'):
        raise HTTPException(status_code=400, detail="File must be in .dat format")
    # Load the file into memory
    myfile = open(file,'rb')

    Xd = pickle.load(myfile)
    new_data = np.array(Xd)
    new_data = new_data[np.newaxis,...]
    new_data.shape
    result = model.predict(new_data)
    final_result = to_onehot(result[0])
    idx = final_result.index(1)
    return {"modulation:type":mods[idx]}