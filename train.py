from tensorflow.keras import layers, models
from dataset import create_training_data
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def create_model():

    model = models.Sequential(
        [
            layers.Input(shape=(512, 1)),
            layers.Conv1D(32, 7, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


x, y = create_training_data(10000)
idx = np.random.permutation(len(x))
x, y = x[idx], y[idx]

model = create_model()

model.fit(x, y, epochs=20, batch_size=32, validation_split=0.2, shuffle=True)

model.save("transient_classifier.keras")
