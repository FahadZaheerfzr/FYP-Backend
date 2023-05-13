from fastapi import FastAPI, File, UploadFile, HTTPException
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
async def predict_file(file: UploadFile):
    # Check if the uploaded file is a .dat file
    if not file.filename.endswith('.dat'):
        raise HTTPException(status_code=400, detail="File must be in .dat format")
    # Load the file into memory
    contents = await file.read()
    # Convert the file contents into a numpy array
    data = np.frombuffer(contents, dtype=np.float32)
    # Reshape the data to match the input shape of your model
    data = data.reshape((1, -1, 1))
    # Make the prediction using the loaded model
    result = model.predict(data)
    # Convert the prediction to a one-hot encoded format
    final_result = to_onehot(result[0])
    # Get the index of the predicted class
    idx = final_result.index(1)
    # Get the name of the predicted class from the mods list
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    prediction = mods[idx]
    # Return the predicted class as the response
    return json.loads(json.dumps({"prediction": prediction}, cls=NumpyEncoder))
