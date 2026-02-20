# spectral observables
import numpy as np

def compute_gap(eigs, k_min=10):
    eigs = np.sort(eigs)
    delta1 = eigs[1] - eigs[0]
    deltak = eigs[k_min] - eigs[0] if len(eigs) > k_min else None
    return delta1, deltak

def spacing_ratios(eigs, beta0=0.2, beta1=0.8):
    eigs = np.sort(eigs)
    N = len(eigs)
    i0 = int(beta0 * N)
    i1 = int(beta1 * N)

    spacings = np.diff(eigs)
    r = []

    for i in range(i0+1, i1-1):
        s1 = spacings[i-1]
        s2 = spacings[i]
        if s1 > 0 and s2 > 0:
            r.append(min(s1, s2) / max(s1, s2))

    return np.array(r)
