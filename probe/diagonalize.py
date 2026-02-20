# dense diagonalization
import numpy as np

def diagonalize_dense(H):
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals.real
