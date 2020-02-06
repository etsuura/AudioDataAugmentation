import numpy as np
from scipy.io import wavfile
import pysptk
import pyworld as pw

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

class audioProcessing():
    def __init__(self):
        self.data = None
        self.fs = None
        self.fo = None
        self.sp = None
        self.ap = None
        self.order = None
        self.alpha = None
        self.synthesized = None

    def setData(self, path):
        self.fs, self.data = wavfile.read(path)
        self.data = self.data.astype(np.float)  # If you use world, convert it to float.
        self.fftlen = pw.get_cheaptrick_fft_size(self.fs)
        self.alpha = pysptk.util.mcepalpha(self.fs)
        self.order = 24

    def getData(self):
        #Todo use try and except
        assert self.data.all, "data is None"
        return self.data

    def getFs(self):
        assert self.fs, "fs is None"
        return self.fs

    def getFo(self):
        assert self.fo.all, "fo is None"
        return self.fo

    def getSp(self):
        assert self.sp.all, "sp is None"
        return self.sp

    def getAp(self):
        assert self.ap.all, "data is None"
        return self.ap

    def setPara(self):
        assert self.data.all or self.fs, "data or fs is None"
        _fo, _time = pw.dio(self.data, self.fs)
        self.fo = pw.stonemask(self.data, _fo, _time, self.fs)
        self.sp = pw.cheaptrick(self.data, self.fo, _time, self.fs)
        self.ap = pw.d4c(self.data, self.fo, _time, self.fs)

    def synthesizeVoice(self):
        assert self.fs or self.fo.all or self.sp.all or self.ap.all, "fs or fo or sp or ap is None"
        self.synthesized = pw.synthesize(self.fo, self.sp, self.ap, self.fs)
        self.synthesized = self.synthesized.astype(np.int16)

    def getSynthesizeVoice(self):
        assert self.synthesized.all, "data is None"
        return self.synthesized

class calcPara(audioProcessing):
    def __init__(self):
        self.mcep = None
        self.bap = None
        self.spCalc = None

    def setBap(self):
        self.bap = pw.code_aperiodicity(self.ap, self.fs)

    def getBap(self):
        if self.bap != None:
            return self.bap

    def setSp2Mc(self):   # alpha is all-pass constant
        assert self.sp.all or self.order or self.alpha, "sp or order or alpha is None"
        self.mcep = pysptk.conversion.sp2mc(self.sp, self.order, self.alpha)
        return self.mcep

    def setMc2Sp(self):
        assert self.mcep.all or self.alpha or self.fftlen, "mcep or alpha or fftlen is None"
        self.spCalc = pysptk.conversion.mc2sp(self.mcep, self.alpha, self.fftlen)
        return self.spCalc
