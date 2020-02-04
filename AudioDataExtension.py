from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
import librosa
import numpy as np
import pysptk
import pyworld as pw

import utility

# 扱うデータの方針：扱うデータは常にfloat32, 入出力時に型変換を行う

def time_stretch(data, speed):
    assert data.dtype == "float32", "data type error"

    data = data[:].reshape(1, -1)

    reader = ArrayReader(data)
    writer = ArrayWriter(channels=1)
    tsm = wsola(channels=1, speed=speed)
    tsm.run(reader, writer)

    output = np.ascontiguousarray(writer.data.T)
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
    assert data.dtype == "float32", "data type error"

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    # Todo Research resample
    # y_shift = librosa.core.resample(librosa.effects.time_stretch(data, rate), float(fs)/rate, fs, res_type='kaiser_best')
    y_shift = librosa.core.resample(time_stretch(data, rate), float(fs)/rate, fs, res_type='kaiser_best')

    # Crop to the same dimension as the input
    return fix_length(y_shift, len(data))

# def frame_shift(data, fs):
#     default_frame_period = 5.0
#     # default_f0_floor = 71.0
#     # default_f0_ceil = 800.0
#
#     return shift_data

def main():
    AudioPath = "./data/RedDot/35/m0022/20150325233545661_m0022_35.wav"
    data, fs = utility.AudioRead(AudioPath)

    time_stretch_output = time_stretch(data, 0.5)
    utility.AudioWrite(time_stretch_output, fs, "time_stretch_output.wav")

    pitch_shift_output = pitch_shift_wsola(data, fs, 2)
    utility.AudioWrite(pitch_shift_output, fs, "pitch_shift_output.wav")

if __name__ == '__main__':
    main()