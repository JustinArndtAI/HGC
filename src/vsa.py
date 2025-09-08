import numpy as np

class VSA:
    def __init__(self, dim=2048):
        self.dim = dim

    def bind(self, vec1, vec2):
        fft1 = np.fft.fft(vec1)
        fft2 = np.fft.fft(vec2)
        return np.fft.ifft(fft1 * fft2).real

    def unbind(self, vec1, vec2):
        fft1 = np.fft.fft(vec1)
        fft2 = np.fft.fft(vec2).conj()
        return np.fft.ifft(fft1 * fft2).real

    def bundle(self, vec_list):
        return np.sum(vec_list, axis=0)
