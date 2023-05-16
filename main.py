from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

app = FastAPI()

app.add_middleware(
   CORSMiddleware,
    allow_origins = ["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials =True,
    allow_methods = ["*"],
    allow_headers= ["*"],   
)

model = load_model('model/Conv-LSTM.h5', compile=False)
model.compile()


def process_array(arr):
    # Find the maximum value in the array
    max_value = max(map(max, arr))

    # Iterate over each element in the array
    processed_arr = []
    for row in arr:
        new_row = []
        for num in row:
            if num == max_value:
                new_row.append(1)
            else:
                new_row.append(0)
        processed_arr.append(new_row)

    return processed_arr

@app.get("/")
def classify():
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    file = open("singleton_final.dat",'rb')
    Xd = pickle.load(file)
    new_data = np.array(Xd)
    result = model.predict(new_data)
    final_result = process_array(result[0])
    idx = final_result.index(1)
    return {"modulation:type":mods[idx]}

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    # Check if the uploaded file is a .dat file
    if not file.filename.endswith('.dat'):
        raise HTTPException(status_code=400, detail="File must be in .dat format")
    # Load the file into memory
    # myfile = open(file.filename,'rb')

    Xd = pickle.load(file.file)
    new_data = np.array(Xd)
    result = model.predict(new_data)
    print(result)
    final_result = process_array(result)
    print(final_result)
    idx = final_result[0].index(1)
    return {"modulation:type":mods[idx]}