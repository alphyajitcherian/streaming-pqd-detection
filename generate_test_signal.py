import numpy as np


def impulsive_transient(t, t1, A):

    duration = np.random.uniform(200e-6, 1e-3)
    tau = duration / 4.605
    end = t1 + duration
    alpha = np.random.uniform(0.5, 5)
    polarity = np.random.choice([-1, 1])
    pulse = np.zeros_like(t)

    idx = np.where(t >= t1)[0]
    if len(idx) > 0:

        decay = t[idx[:]] - t[idx[0]]
        pulse[idx[:]] = np.exp(-decay / tau)

    specs = {
        "type": "Impulsive",
        "start_ms": t1 * 1000,
        "end_ms": (t1 + duration) * 1000,
        "duration_ms": duration * 1000,
    }

    return polarity * alpha * A * pulse, specs


def oscillatory_transient(t, t1, A):

    duration = np.random.uniform(0.3e-3, 50e-3)
    tau = duration / 4.605
    end = t1 + duration
    alpha = np.random.uniform(0.1, 0.8)
    f_n = np.random.uniform(1000, 5000)
    w_n = 2 * np.pi * f_n

    exponent = -(t - t1) / tau
    envelope = np.exp(np.clip(exponent, -500, 0))
    envelope[t < t1] = 0

    osc = np.sin(w_n * (t - t1))
    osc[t < t1] = 0

    specs = {
        "type": "Oscillatory",
        "start_ms": t1 * 1000,
        "end_ms": (t1 + duration) * 1000,
        "duration_ms": duration * 1000,
    }

    return A * alpha * envelope * osc, specs


def generate_long_test_signal(
    FS=100000, duration_ms=500, n_impulsive=2, n_oscillatory=2
):

    duration = duration_ms * 1e-3
    t = np.arange(0, duration, 1 / FS)

    A = 240 * np.sqrt(2)
    f = 50
    w = 2 * np.pi * f

    phi = np.random.uniform(0, 2 * np.pi)
    signal = A * np.sin(w * t + phi)
    signal += np.random.normal(0, 0.005 * A, len(t))

    if n_impulsive == 0 and n_oscillatory == 0:
        return (signal / A), []

    types = ["impulsive"] * n_impulsive + ["oscillatory"] * n_oscillatory
    np.random.shuffle(types)

    n_total = len(types)
    margin = 0.02
    available = duration - 2 * margin
    slot_size = available / n_total
    min_gap = 0.05

    if slot_size < min_gap:
        raise ValueError(
            f"Not enough room — {n_total} transients need "
            f"{(n_total * min_gap+2*margin)*1000:.0f}ms minimum needed"
        )

    jitter = (slot_size - min_gap) / 2
    times = []
    for i in range(n_total):

        times.append(
            margin + i * slot_size + slot_size / 2 + np.random.uniform(-jitter, jitter)
        )

    ground_truth = []

    for t1, ttype in zip(times, types):
        if ttype == "impulsive":
            trans, specs = impulsive_transient(t, t1, A)
        else:
            trans, specs = oscillatory_transient(t, t1, A)
        signal += trans
        ground_truth.append(specs)

    return t, signal, ground_truth
