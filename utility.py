import numpy as np
from scipy.io import wavfile

def encode_32to16bits(data):
    # Todo check this function
    assert data.dtype == "float32", "input data is not 32bits"

    if (np.max(np.abs(data)) < 1.0):
        data = np.clip(data * 2**15, -2**15, 2**15 - 1).astype(np.int16)
    else:
        data = data.astype(np.int16)
    return data

def encode_16to32bits(data):
    # Todo check this function
    assert data.dtype == "int16", "input data is not 16bits"

    if (np.max(np.abs(data)) > 1.0):
        data = np.clip(data / 2**15, -1, 1).astype(np.float32)
    else:
        data = data.astype(np.float32)
    return data

def AudioRead(path):
    fs, data = wavfile.read(path)
    data = encode_16to32bits(data)
    # data = data.astype(np.float)
    return data, fs

def AudioWrite(data, fs, name):
    assert data.dtype == "float32", "data type error"
    data = encode_32to16bits(data)
    wavfile.write(name, fs, data)