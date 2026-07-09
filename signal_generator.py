import numpy as np

FS = 100000


def u(t):
    return np.heaviside(t, 0)


def generate_oscillatory():

    T = 20e-3
    duration = np.random.uniform(0.3e-3, 50e-3)
    tau = duration / 4.605
    max_start = max(0.002, T - 4.605 * tau)
    t1 = np.random.uniform(0.002, max_start)
    t = np.arange(t1 - 5e-3, t1 + 4.605 * tau + 5e-3, 1 / FS)

    A = 240 * np.sqrt(2)
    f = 50
    w = 2 * np.pi * f
    alpha = np.random.uniform(0.1, 0.8)
    f_n = np.random.uniform(1000, 5000)
    w_n = 2 * np.pi * f_n
    phi = np.random.uniform(0, 2 * np.pi)

    y = A * (
        np.sin(w * t + phi)
        + (alpha * np.exp(-(t - t1) / tau) * np.sin(w_n * (t - t1)) * (u(t - t1)))
    )
    y += np.random.normal(0, 0.005 * A, len(t))

    return y


def generate_impulsive():

    T = 20e-3
    duration = np.random.uniform(200e-6, 1e-3)
    tau = duration / 4.605

    t1 = np.random.uniform(0.002, T - 2e-3)

    t = np.arange(t1 - 5e-3, t1 + 5e-3, 1 / FS)

    A = 240 * np.sqrt(2)
    f = 50
    w = 2 * np.pi * f
    phi = np.random.uniform(0, 2 * np.pi)
    alpha = np.random.uniform(0.5, 5)
    polarity = np.random.choice([-1, 1])
    pulse = np.zeros_like(t)

    idx = np.where(t >= t1)[0]
    if len(idx) > 0:

        decay = t[idx[:]] - t[idx[0]]
        pulse[idx[:]] = np.exp(-decay / tau)

    y = A * (np.sin(w * t + phi) + polarity * alpha * pulse)
    y += np.random.normal(0, 0.005 * A, len(t))

    return y
