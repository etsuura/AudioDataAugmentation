import numpy as np
from scipy.io import wavfile
# import sounddevice as sd
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

def encode_16bits(data) :
    if (np.max(np.abs(data)) < 1.0):
        data = np.clip(data * 2**15, -2**15, 2**15 - 1).astype(np.int16)
    else:
        data = data.astype(np.int16)
    return data


def AudioRead(path):
    fs, data = wavfile.read(path)
    # data = data.astype(np.float)
    return data, fs

def main():
    AudioPath = "./data/wav/31/f0002/20150211152959223_f0002_31.wav"
    data, fs = AudioRead(AudioPath)

    data = data[:].reshape(1, -1)

    # Run the TSM procedure
    reader = ArrayReader(data)
    writer = ArrayWriter(channels=1)
    tsm = wsola(channels=1, speed=0.5)
    tsm.run(reader, writer)

    output = np.ascontiguousarray(writer.data.T)
    output = encode_16bits(output)
    wavfile.write("output.wav", fs, output)


if __name__ == '__main__':
    main()