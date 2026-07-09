import numpy as np
import pywt
from generate_test_signal import generate_long_test_signal
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import preprocess
import datetime
import csv
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FS = 100000  # Sampling frequency
CHUNK_SIZE = int(FS * 0.02)  # 20ms chunks
BUFFER_SIZE = 10000  # 10k sample buffer
BASELINE_SAMPLES = int(FS / 50)  # First power cycle
MERGE_GAP = int(10e-3 * FS)  # 10ms merge gap
PADDING = int(5e-3 * FS)  # 5ms on either end
MAX_DURATION = int(80e-3 * FS)  # If transient longer than 80ms emit it

model = tf.keras.models.load_model("transient_classifier.keras")


def energy(signal):

    coeffs = pywt.wavedec(signal, "db4", level=6)
    coeffs[0] = np.zeros_like(coeffs[0])
    transient_signal = pywt.waverec(coeffs, "db4")[: len(signal)]
    return transient_signal**2


class Transient_Detector:

    def __init__(self, display=True, log=False):
        self.buffer = np.zeros(BUFFER_SIZE)
        self.threshold = None
        self.in_transient = False
        self.current_start = None
        self.current_end = None
        self.pointer = 0
        self.last_emitted = 0
        self.n_written = 0
        self.transient_count = 0
        self.detections = []
        self.display = display
        self.log = log

        if log:
            self.log_file = "transients.csv"
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Timestamp",
                        "Type",
                        "Start (ms)",
                        "End (ms)",
                        "Duration (ms)",
                        "Confidence (%)",
                    ]
                )

    def write(self, chunk):

        for sample in chunk:
            self.buffer[self.pointer] = sample
            self.pointer = (self.pointer + 1) % BUFFER_SIZE
        self.n_written += len(chunk)

    def feed_chunk(self, chunk):

        self.write(chunk)
        if self.threshold is None:
            self.set_threshold()
            return
        if self.n_written < BUFFER_SIZE:
            return
        self.detect()

    def order(self):

        return np.concatenate(
            [self.buffer[self.pointer :], self.buffer[: self.pointer]]
        )

    def set_threshold(self):

        if self.n_written < BASELINE_SAMPLES:
            return
        baseline = self.order()[-BASELINE_SAMPLES:]
        e = energy(baseline)
        self.threshold = 10 * np.percentile(e, 99)

    def detect(self):

        ordered = self.order()
        e = energy(ordered)
        is_over = e > self.threshold

        if not np.any(is_over):
            return

        edges = np.diff(is_over.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1

        if is_over[0]:
            starts = np.insert(starts, 0, 0)
        if is_over[-1]:
            ends = np.append(ends, None)

        merged = []
        for s, en in zip(starts, ends):
            if (
                merged
                and merged[-1][1] is not None
                and (s - merged[-1][1]) <= MERGE_GAP
            ):
                merged[-1][1] = en
            else:
                merged.append([s, en])

        origin = self.n_written - BUFFER_SIZE
        for s_local, e_local in merged:
            global_start = origin + s_local

            if e_local is None:
                global_end = None
            else:
                global_end = origin + e_local

            if global_end is not None and global_start <= self.last_emitted + MERGE_GAP:
                continue

            if global_end is None:
                if not self.in_transient:
                    self.current_start = global_start
                    self.current_end = origin + len(e) - 1
                    self.in_transient = True

                if self.current_end - self.current_start > MAX_DURATION:
                    self.emit(ordered, self.current_start, self.current_end)
                    self.in_transient = False
            else:
                if self.in_transient:
                    start = min(global_start, self.current_start)
                    self.in_transient = False
                else:
                    start = global_start

                self.emit(ordered, start, global_end)

    def emit(self, ordered, global_start, global_end):

        self.transient_count += 1
        origin = self.n_written - BUFFER_SIZE
        t_start = max(0, (global_start - origin) - PADDING)
        t_end = min(BUFFER_SIZE, (global_end - origin) + PADDING)
        transient = ordered[t_start:t_end]
        sample = preprocess(transient)
        pred = model(sample[np.newaxis], training=False).numpy()
        label = int(np.argmax(pred))
        confidence = float(np.max(pred))

        wavelet_end = global_end

        if label == 0:
            global_start, global_end = self.imp_precise_stamp(
                transient, t_start, t_end, origin
            )

        start_ms = global_start / FS * 1000
        end_ms = global_end / FS * 1000

        tag = "Impulsive" if label == 0 else "Oscillatory"

        if self.display:
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {tag} transient\n"
                f" start: {start_ms:.3f} ms\n"
                f" end: {end_ms:.3f} ms\n"
                f" duration: {end_ms - start_ms:.3f} ms\n"
                f" confidence: {confidence:.1%}\n"
            )

        if self.log:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        tag,
                        f"{start_ms:.3f}",
                        f"{end_ms:.3f}",
                        f"{end_ms - start_ms:.3f}",
                        f"{confidence*100:.1f}",
                    ]
                )

        self.detections.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "type": tag,
                "confidence": confidence,
            }
        )
        self.last_emitted = wavelet_end

    def imp_precise_stamp(self, transient, t_start, t_end, origin):

        derivative = np.diff(transient)
        peak = int(np.argmax(np.abs(derivative)))
        start = peak - 1
        pre_samples = transient[max(0, start - 20) : max(0, start - 10)]
        noise_std = np.std(pre_samples)
        noise_threshold = 2.5 * noise_std
        end = peak
        for i in range(peak, len(transient) - 10):
            if np.std(transient[i : i + 10]) <= noise_threshold:
                end = i
                break

        precise_start = origin + t_start + start
        precise_end = origin + t_start + end
        return int(precise_start), int(precise_end)

    def flush(self):

        MIN_DURATION = int(0.5e-3 * FS)
        END_MARGIN = int(5e-3 * FS)

        if self.in_transient:
            if self.current_start is None or self.current_end is None:
                self.in_transient = False
            else:
                if (
                    self.current_end - self.current_start < MIN_DURATION
                    or self.current_start > self.n_written - END_MARGIN
                ):
                    self.in_transient = False
                else:
                    ordered = self.order()
                    self.emit(ordered, self.current_start, self.current_end)
                    self.in_transient = False

        if self.display:
            duration = self.n_written / FS * 1000
            if self.transient_count == 0:
                print(f"Stream ended — no transients in {duration:.2f} ms of signal.")
            else:
                print(
                    f"Stream ended — {self.transient_count} transient(s) detected in {duration:.2f} ms of signal."
                )


def signal_stream(signal, chunk_size=CHUNK_SIZE):

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i : i + chunk_size]
        yield chunk


if __name__ == "__main__":

    while True:
        print("\n1. Generate test signal")
        print("2. Upload signal from file")
        choice = input("Enter choice (1 or 2):")
        if choice == "1":
            t, signal, ground_truth = generate_long_test_signal()
            break
        elif choice == "2":
            file_path = input(
                "Enter file path (.npy or .csv with 1 column of voltage readings):"
            )
            try:
                if file_path.endswith(".npy"):
                    signal = np.load(file_path)
                    t = np.arange(len(signal)) / FS
                else:
                    signal = np.loadtxt(file_path, delimiter=",")
                    t = np.arange(len(signal)) / FS
            except FileNotFoundError:
                print("File not found. Please try again.")
                continue
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue

    while True:
        flag = input("Log detections to transients.csv? (y/n):")
        if flag.lower() in ["y", "n"]:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue

    if choice == "1":

        print("\nGround truth:")
        print(
            f"{'#':<4} {'Type':<15} {'Start(ms)':<12} {'End(ms)':<12} {'Duration(ms)':<12}"
        )
        print("-" * 55)

        for i, g in enumerate(ground_truth):
            print(
                f"{i+1:<4} {g['type']:<15} {g['start_ms']:<12.3f} {g['end_ms']:<12.3f} {g['duration_ms']:<12.3f}"
            )
        print()

    print("Processing signal...\n")

    stream = signal_stream(signal)
    detector = Transient_Detector(display=True, log=(flag.lower() == "y"))

    for chunk in stream:
        detector.feed_chunk(chunk)

    detector.flush()

    normalized = signal / np.max(np.abs(signal))
    plt.figure()
    plt.plot(t * 1000, normalized)
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized Voltage (p.u.)")
    plt.title(
        "Test Signal (Synthetic)"
        if choice == "1"
        else f"Signal: {os.path.basename(file_path)}"
    )
    plt.show()
