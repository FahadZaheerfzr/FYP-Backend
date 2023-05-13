from fastapi import FastAPI
from keras.models import load_model
import pickle
import numpy as np

app = FastAPI()

model = load_model('model/Conv-LSTM.h5', compile=False)

model.compile()

print(model.summary())



def to_onehot(yy):
    result = []
    for num in yy:
      result.append(round(num))
    return result

@app.get("/")
def classify():
    mods = [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
    file = open("/singleton_final.dat",'rb')
    Xd = pickle.load(file)
    new_data = np.array(Xd)
    new_data = new_data[np.newaxis,...]
    new_data.shape
    result = model.predict(new_data)
    final_result = to_onehot(result[0])
    idx = final_result.index(1)
    return {"modulation:type":mods[idx]}

