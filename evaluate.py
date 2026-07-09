import numpy as np
from generate_test_signal import generate_long_test_signal
from main import Transient_Detector, signal_stream, FS, BUFFER_SIZE
from dataset import preprocess
import tensorflow as tf
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = tf.keras.models.load_model("transient_classifier.keras")


def evaluate(trials=100):

    total = 0
    detected = 0
    correct_type = 0
    false_positives = 0
    imp_as_imp = 0
    imp_as_osc = 0
    osc_as_osc = 0
    osc_as_imp = 0
    start_errors = []
    imp_errors = []
    osc_errors = []

    for i in range(trials):

        _, signal, ground_truth = generate_long_test_signal(
            n_impulsive=2, n_oscillatory=2
        )
        detector = Transient_Detector(display=False, log=False)

        for chunk in signal_stream(signal):
            detector.feed_chunk(chunk)
        detector.flush()

        for j in detector.detections:
            matched = 0
            for g in ground_truth:
                if abs(j["start_ms"] - g["start_ms"]) <= 5:
                    matched = 1
                    break
            if not matched:
                false_positives += 1
            matched = 0

        for g in ground_truth:
            total += 1
            valid_cases = []
            for d in detector.detections:
                if abs(d["start_ms"] - g["start_ms"]) <= 5:
                    valid_cases.append(d)

            if not valid_cases:
                continue

            best = min(valid_cases, key=lambda d: abs(d["start_ms"] - g["start_ms"]))
            detected += 1
            error = abs(best["start_ms"] - g["start_ms"])
            start_errors.append(error)

            if best["type"] == g["type"]:
                correct_type += 1
                if g["type"] == "Impulsive":
                    imp_as_imp += 1
                    imp_errors.append(error)
                else:
                    osc_as_osc += 1
                    osc_errors.append(error)
            else:
                if g["type"] == "Impulsive":
                    imp_as_osc += 1
                else:
                    osc_as_imp += 1

    detection_rate = detected / total * 100
    type_accuracy = correct_type / detected * 100 if detected > 0 else 0
    mean_start_error = np.mean(start_errors) if start_errors else 0
    mean_imp_error = np.mean(imp_errors) if imp_errors else 0
    mean_osc_error = np.mean(osc_errors) if osc_errors else 0

    print(f"\nEvaluation — {trials} trials, {total} transients\n")
    print(f"  Detection rate:   {detection_rate:.2f}%  ({detected}/{total})")
    print(f"  Type accuracy:    {type_accuracy:.2f}%  ({correct_type}/{detected})")
    print(f"  Mean start error: {mean_start_error:.3f} ms")
    print(f"  Mean impulsive start error:   {mean_imp_error:.3f} ms")
    print(f"  Mean oscillatory start error: {mean_osc_error:.3f} ms")
    print(f"  False positives:  {false_positives} ({100*false_positives/total:.2f}%)")

    print(f"\nConfusion Matrix:")
    print(f"                   Pred Impulsive   Pred Oscillatory")
    print(f"Actual Impulsive   {imp_as_imp:10d}       {imp_as_osc:10d}")
    print(f"Actual Oscillatory {osc_as_imp:10d}       {osc_as_osc:10d}")

    accuracy = (
        (imp_as_imp + osc_as_osc) / (imp_as_imp + imp_as_osc + osc_as_imp + osc_as_osc)
        if detected > 0
        else 0
    )
    precision_imp = (
        imp_as_imp / (imp_as_imp + osc_as_imp) if (imp_as_imp + osc_as_imp) > 0 else 0
    )
    precision_osc = (
        osc_as_osc / (osc_as_osc + imp_as_osc) if (osc_as_osc + imp_as_osc) > 0 else 0
    )
    recall_imp = (
        imp_as_imp / (imp_as_imp + imp_as_osc) if (imp_as_imp + imp_as_osc) > 0 else 0
    )
    recall_osc = (
        osc_as_osc / (osc_as_osc + osc_as_imp) if (osc_as_osc + osc_as_imp) > 0 else 0
    )
    f1_imp = (
        2 * precision_imp * recall_imp / (precision_imp + recall_imp)
        if (precision_imp + recall_imp) > 0
        else 0
    )
    f1_osc = (
        2 * precision_osc * recall_osc / (precision_osc + recall_osc)
        if (precision_osc + recall_osc) > 0
        else 0
    )

    print(f"\nAccuracy              {accuracy:.3f}")
    print(
        f"Precision  Impulsive: {precision_imp:.3f}   Oscillatory: {precision_osc:.3f}"
    )
    print(f"Recall     Impulsive: {recall_imp:.3f}   Oscillatory: {recall_osc:.3f}")
    print(f"F1 score   Impulsive: {f1_imp:.3f}   Oscillatory: {f1_osc:.3f}")


evaluate(trials=10000)
