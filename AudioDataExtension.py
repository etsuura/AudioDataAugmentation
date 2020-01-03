import numpy as np
import scipy
# import sounddevice as sd
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

def AudioRead(path):
    fs, data = scipy.wavfile.read(path)
    return data, fs

def main():
    AudioPath = "./data/wav/31/f0002/20150211152959223_f0002_31.wav"
    data, fs = AudioRead(AudioPath)

    # Run the TSM procedure
    reader = ArrayReader(data)
    writer = ArrayWriter(channels=1)

    tsm = wsola(channels=1, speed=0.5)
    tsm.run(reader, writer)


if __name__ == '__main__':
    main()