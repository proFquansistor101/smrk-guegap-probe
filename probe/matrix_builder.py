# matrix construction logic
import numpy as np
from math import log, pi

def first_primes(M):
    primes = []
    n = 2
    while len(primes) < M:
        for p in primes:
            if n % p == 0:
                break
        else:
            primes.append(n)
        n += 1
    return primes

def build_matrix(params):
    N = params["N"]
    M = params["M"]
    theta = params["theta"]
    boundary = params["boundary"]

    kappa = theta["kappa"]
    lam = theta["lambda"]
    alpha = theta["alpha"]
    eta = theta["eta"]
    s = theta["s"]
    t = theta["t"]
    phi_offsets = theta["phi_offsets"]

    H = np.zeros((N, N), dtype=np.complex128)

    # Diagonal potential
    for n in range(N):
        H[n, n] = lam * log(1 + (n + 1))

    primes = first_primes(M)

    for m, p in enumerate(primes):
        for n in range(N):
            target = n + p
            if boundary == "cyclic":
                target = target % N
            elif boundary == "dirichlet":
                if target >= N:
                    continue

            amp = kappa / ((log(1 + p) ** s) * (log(1 + (n + 1)) ** t))
            liouville = (-1) ** (n + 1)  # simple deterministic sign
            amp *= (1 + eta * liouville)

            phase = 2 * pi * alpha * (n + 1) + phi_offsets[m]
            w = amp * np.exp(1j * phase)

            H[n, target] += w
            H[target, n] += np.conjugate(w)

    return H
