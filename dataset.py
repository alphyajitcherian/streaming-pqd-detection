import numpy as np
from scipy.signal import resample
from signal_generator import generate_impulsive, generate_oscillatory


def preprocess(y):

    y = resample(y, 512)
    peak = np.max(np.abs(y))
    y = y / peak
    y = y[:, np.newaxis]

    return y


def create_training_data(n_samples):

    signals = []
    labels = []

    for i in range(n_samples):
        sig = generate_impulsive()
        signals.append(preprocess(sig))
        labels.append(0)

    for _ in range(n_samples):
        sig = generate_oscillatory()
        signals.append(preprocess(sig))
        labels.append(1)

    signals = np.array(signals)
    labels = np.array(labels)

    return signals, labels
