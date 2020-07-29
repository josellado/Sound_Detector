
from scipy.fftpack import fft
import numpy as np

def audioFeaturesFourie(audio):
    array = audio.get_array_of_samples()
    abs_four = np.abs(fft(array,n=754))
    return abs_four