import numpy as np

class VSA:
    """
    A class to handle Vector Symbolic Architecture operations,
    specifically Holographic Reduced Representations (HRR).
    """
    def __init__(self, dim: int):
        self.dim = dim

    def make_vector(self) -> np.ndarray:
        """Creates a normalized random vector."""
        return np.random.normal(0, 1/np.sqrt(self.dim), self.dim)

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Binds two vectors using circular convolution.
        """
        if vec1.shape != (self.dim,) or vec2.shape != (self.dim,):
            raise ValueError(f"All vectors must have dimension {self.dim}")
        
        fft1 = np.fft.fft(vec1)
        fft2 = np.fft.fft(vec2)
        return np.fft.ifft(fft1 * fft2).real

    def unbind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Unbinds two vectors using circular correlation. The complex conjugate is critical.
        """
        if vec1.shape != (self.dim,) or vec2.shape != (self.dim,):
            raise ValueError(f"All vectors must have dimension {self.dim}")

        fft1 = np.fft.fft(vec1)
        fft2 = np.fft.fft(vec2)
        return np.fft.ifft(fft1.conj() * fft2).real

    def bundle(self, vec_list: list) -> np.ndarray:
        """Bundles a list of vectors using element-wise addition."""
        return np.sum(vec_list, axis=0)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
