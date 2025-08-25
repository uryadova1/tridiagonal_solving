import numpy as np


def generate_coefs(n=2 << 10):
    # n = 2 << 10
    num_systems = 1
    rng = np.random.default_rng(0)

    a = rng.normal(size=(num_systems, n)).astype(np.float32)
    a[:, 0] = 0.0
    c = rng.normal(size=(num_systems, n)).astype(np.float32)
    c[:, -1] = 0.0
    b = (np.abs(a) + np.abs(c) + 1.0).astype(np.float32)
    d = rng.normal(size=(num_systems, n)).astype(np.float32)

    np.save("coefs/A.npy", a)
    np.save("coefs/B.npy", b)
    np.save("coefs/C.npy", c)
    np.save("coefs/D.npy", d)
