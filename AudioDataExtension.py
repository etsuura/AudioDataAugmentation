import numpy as np
from scipy.io import wavfile
import librosa
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

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

def time_stretch(data, speed):
    data = data[:].reshape(1, -1)

    reader = ArrayReader(data)
    writer = ArrayWriter(channels=1)
    tsm = wsola(channels=1, speed=speed)
    tsm.run(reader, writer)

    output = np.ascontiguousarray(writer.data.T)
    output = encode_32to16bits(output)  #必要？

    output = output.flatten()

    return output

# Function to specify the size of ndarray by cutting or zero padding
def fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def pitch_shift(data, fs, pitch_factor):
    return librosa.effects.pitch_shift(data, fs, pitch_factor)

def pitch_shift_wsola(data, fs, n_steps, bins_per_octave=12):
    assert not(bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer)), \
        'bins_per_octave must be a positive integer.'

    # Todo data encode to 32bits float

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    # Todo Research resample
    # y_shift = librosa.core.resample(librosa.effects.time_stretch(data, rate), float(fs)/rate, fs, res_type='kaiser_best')
    y_shift = librosa.core.resample(time_stretch(data, rate), float(fs)/rate, fs, res_type='kaiser_best')


    # Crop to the same dimension as the input
    return fix_length(y_shift, len(data))



def AudioRead(path):
    fs, data = wavfile.read(path)
    # data = data.astype(np.float)
    return data, fs

def main():
    AudioPath = "./data/RedDot/35/m0022/20150325233545661_m0022_35.wav"
    data, fs = AudioRead(AudioPath)

    time_stretch_output = time_stretch(data, 0.8)
    wavfile.write("time_stretch_output.wav", fs, time_stretch_output)



if __name__ == '__main__':
    main()