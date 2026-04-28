from pathlib import Path
from datetime import datetime
import sys
import json
import random
import csv

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# Project settings
# --------------------------------------------------
SAMPLE_RATE = 16000

# Your custom recordings are 1.5 seconds, so we train on 1.5 second windows.
CLIP_SECONDS = 1.5
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("../final_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = Path("runs")
RUN_ID = datetime.now().strftime("run_%Y_%m_%d_%H%M%S")
RUN_DIR = RUNS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["background", "unknown", "flying", "happy"]
NUM_CLASSES = len(LABELS)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------------------------------------------
# Console logger
# Saves terminal output while still printing to screen
# --------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# --------------------------------------------------
# Audio helpers
# --------------------------------------------------
def load_wav(path):
    sr, audio = wavfile.read(path)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, sr // gcd)

    # Normalize to about -1 to 1
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32)


def crop_or_pad(audio, random_crop=False):
    if len(audio) > CLIP_SAMPLES:
        if random_crop:
            start = random.randint(0, len(audio) - CLIP_SAMPLES)
        else:
            start = (len(audio) - CLIP_SAMPLES) // 2

        audio = audio[start:start + CLIP_SAMPLES]

    elif len(audio) < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - len(audio)
        left = pad // 2
        right = pad - left
        audio = np.pad(audio, (left, right))

    return audio.astype(np.float32)


def make_spectrogram(audio):
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    # 30 ms window, 20 ms step at 16 kHz
    stft = tf.signal.stft(
        audio,
        frame_length=480,
        frame_step=320,
        fft_length=512
    )

    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log(spectrogram + 1e-6)

    # Add channel dimension for Conv2D
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram.numpy().astype(np.float32)


def load_class_files(label):
    folder = DATASET_DIR / label
    return sorted(folder.glob("*.wav"))


# --------------------------------------------------
# Dataset builder
# --------------------------------------------------
def build_dataset():
    X = []
    y = []

    dataset_file_counts = {}
    label_to_index = {label: i for i, label in enumerate(LABELS)}

    print("Loading dataset...\n")

    for label in LABELS:
        files = load_class_files(label)
        dataset_file_counts[label] = len(files)
        print(f"{label:12s}: {len(files)} wav files")

        if label == "background":
            # Background files are usually long.
            # We cut many random 1.5-second chunks from each background file.
            chunks_per_file = 50

            for path in files:
                audio = load_wav(path)

                if len(audio) < CLIP_SAMPLES:
                    continue

                for _ in range(chunks_per_file):
                    chunk = crop_or_pad(audio, random_crop=True)
                    spec = make_spectrogram(chunk)
                    X.append(spec)
                    y.append(label_to_index[label])

        else:
            for path in files:
                audio = load_wav(path)
                audio = crop_or_pad(audio, random_crop=False)
                spec = make_spectrogram(audio)
                X.append(spec)
                y.append(label_to_index[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    dataset_sample_counts = {}
    for label_index, label in enumerate(LABELS):
        dataset_sample_counts[label] = int(np.sum(y == label_index))

    print("\nFinal dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nFinal sample counts after background chunking:")
    for label in LABELS:
        print(f"{label:12s}: {dataset_sample_counts[label]} samples")

    return X, y, dataset_file_counts, dataset_sample_counts


# --------------------------------------------------
# CNN model
# --------------------------------------------------
def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# TFLite conversion
# --------------------------------------------------
def representative_dataset_gen(X_train):
    for i in range(min(100, len(X_train))):
        sample = X_train[i:i + 1].astype(np.float32)
        yield [sample]


def save_tflite_models(model, X_train):
    float_path = MODEL_DIR / "keyword_model_float.tflite"
    quant_path = MODEL_DIR / "keyword_model_quantized.tflite"

    print("\nConverting to float TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = converter.convert()
    float_path.write_bytes(float_model)
    print(f"Saved: {float_path}")

    print("\nConverting to fully quantized int8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quant_model = converter.convert()
    quant_path.write_bytes(quant_model)
    print(f"Saved: {quant_path}")

    return float_path, quant_path


# --------------------------------------------------
# TFLite evaluation
# --------------------------------------------------
def evaluate_tflite_model(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]

    correct = 0
    predictions = []

    for i in range(len(X_test)):
        x = X_test[i:i + 1]

        if input_details["dtype"] == np.int8:
            x = x / input_scale + input_zero_point
            x = np.clip(x, -128, 127).astype(np.int8)
        else:
            x = x.astype(np.float32)

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details["index"])
        pred = int(np.argmax(output, axis=1)[0])
        predictions.append(pred)

        if pred == y_test[i]:
            correct += 1

    accuracy = correct / len(y_test)

    tflite_details = {
        "model_path": str(tflite_path),
        "accuracy": accuracy,
        "input_dtype": str(input_details["dtype"]),
        "output_dtype": str(output_details["dtype"]),
        "input_shape": input_details["shape"].tolist(),
        "output_shape": output_details["shape"].tolist(),
        "input_quantization": input_details["quantization"],
        "output_quantization": output_details["quantization"],
        "file_size_bytes": tflite_path.stat().st_size,
    }

    return accuracy, predictions, tflite_details


# --------------------------------------------------
# Saving helpers
# --------------------------------------------------
def save_model_summary(model):
    summary_path = RUN_DIR / "model_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    print(f"Saved model summary: {summary_path}")


def save_training_history(history):
    history_path = RUN_DIR / "training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    # Accuracy plot
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_plot_path = RUN_DIR / "accuracy_plot.png"
    plt.savefig(acc_plot_path, dpi=200)
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_plot_path = RUN_DIR / "loss_plot.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()

    print(f"Saved accuracy plot: {acc_plot_path}")
    print(f"Saved loss plot:     {loss_plot_path}")


def save_confusion_matrix_outputs(cm, report_text, report_dict):
    cm_txt_path = RUN_DIR / "confusion_matrix.txt"
    cm_csv_path = RUN_DIR / "confusion_matrix.csv"
    cm_png_path = RUN_DIR / "confusion_matrix.png"
    report_txt_path = RUN_DIR / "classification_report.txt"
    report_json_path = RUN_DIR / "classification_report.json"

    # TXT
    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write(str(cm))

    # CSV
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + LABELS)
        for i, label in enumerate(LABELS):
            writer.writerow([label] + list(cm[i]))

    # PNG
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(LABELS)), LABELS)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=200)
    plt.close()

    # Classification report
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    print(f"Saved confusion matrix txt: {cm_txt_path}")
    print(f"Saved confusion matrix csv: {cm_csv_path}")
    print(f"Saved confusion matrix png: {cm_png_path}")
    print(f"Saved classification report txt: {report_txt_path}")
    print(f"Saved classification report json: {report_json_path}")


def save_predictions(y_test, y_pred):
    predictions_path = RUN_DIR / "test_predictions.csv"

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_index", "true_label", "pred_index", "pred_label", "correct"])

        for i, (true_idx, pred_idx) in enumerate(zip(y_test, y_pred)):
            true_idx = int(true_idx)
            pred_idx = int(pred_idx)
            writer.writerow([
                i,
                true_idx,
                LABELS[true_idx],
                pred_idx,
                LABELS[pred_idx],
                true_idx == pred_idx
            ])

    print(f"Saved test predictions: {predictions_path}")


def calculate_frr(report_dict, label):
    # FRR = false rejection rate = 1 - recall
    # For keyword spotting, this means how often the true keyword was missed.
    recall = report_dict[label]["recall"]
    return 1.0 - recall


def save_run_summary(
    input_shape,
    dataset_file_counts,
    dataset_sample_counts,
    split_shapes,
    train_acc,
    val_acc,
    test_acc,
    float_tflite_acc,
    quant_tflite_acc,
    model,
    h5_path,
    keras_path,
    float_path,
    quant_path,
    cm,
    report_text,
    report_dict,
    float_tflite_details,
    quant_tflite_details
):
    flying_frr = calculate_frr(report_dict, "flying")
    happy_frr = calculate_frr(report_dict, "happy")

    summary_path = RUN_DIR / "run_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ECGR 4127 Project 2 Keyword Spotting Training Run\n")
        f.write("=" * 70 + "\n\n")

        f.write("Run information\n")
        f.write("-" * 70 + "\n")
        f.write(f"Run ID: {RUN_ID}\n")
        f.write(f"Run folder: {RUN_DIR}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("Project setup\n")
        f.write("-" * 70 + "\n")
        f.write("Board: ESP32-EYE / ESP32-S3-EYE\n")
        f.write("Custom keyword: flying\n")
        f.write("Speech Commands keyword: happy\n")
        f.write("Model type: Small 2D CNN\n")
        f.write(f"Labels/order: {LABELS}\n\n")

        f.write("Audio and feature settings\n")
        f.write("-" * 70 + "\n")
        f.write(f"Sample rate: {SAMPLE_RATE} Hz\n")
        f.write(f"Clip length: {CLIP_SECONDS} seconds\n")
        f.write(f"Clip samples: {CLIP_SAMPLES}\n")
        f.write("Feature type: log magnitude STFT spectrogram\n")
        f.write("STFT frame length: 480 samples, 30 ms at 16 kHz\n")
        f.write("STFT frame step: 320 samples, 20 ms at 16 kHz\n")
        f.write("STFT FFT length: 512\n")
        f.write(f"Input tensor shape: {input_shape}\n\n")

        f.write("Dataset file counts\n")
        f.write("-" * 70 + "\n")
        for label in LABELS:
            f.write(f"{label}: {dataset_file_counts[label]} wav files\n")

        f.write("\nDataset sample counts after processing\n")
        f.write("-" * 70 + "\n")
        for label in LABELS:
            f.write(f"{label}: {dataset_sample_counts[label]} samples\n")

        f.write("\nSplit sizes\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train X shape: {split_shapes['X_train']}\n")
        f.write(f"Train y shape: {split_shapes['y_train']}\n")
        f.write(f"Validation X shape: {split_shapes['X_val']}\n")
        f.write(f"Validation y shape: {split_shapes['y_val']}\n")
        f.write(f"Test X shape: {split_shapes['X_test']}\n")
        f.write(f"Test y shape: {split_shapes['y_test']}\n\n")

        f.write("Keras / float model accuracy\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")

        f.write("TFLite accuracy\n")
        f.write("-" * 70 + "\n")
        f.write(f"Float TFLite test accuracy: {float_tflite_acc:.4f}\n")
        f.write(f"Quantized TFLite test accuracy: {quant_tflite_acc:.4f}\n\n")

        f.write("False rejection rate\n")
        f.write("-" * 70 + "\n")
        f.write(f"Custom word FRR, flying: {flying_frr:.4f}\n")
        f.write(f"Speech Commands word FRR, happy: {happy_frr:.4f}\n\n")

        f.write("Model size\n")
        f.write("-" * 70 + "\n")
        f.write(f"Parameter count: {model.count_params()}\n")
        f.write(f"H5 file size bytes: {h5_path.stat().st_size}\n")
        f.write(f"Keras file size bytes: {keras_path.stat().st_size}\n")
        f.write(f"Float TFLite file size bytes: {float_path.stat().st_size}\n")
        f.write(f"Quantized TFLite file size bytes: {quant_path.stat().st_size}\n\n")

        f.write("TFLite input/output details\n")
        f.write("-" * 70 + "\n")
        f.write("Float TFLite:\n")
        f.write(json.dumps(float_tflite_details, indent=4, default=str))
        f.write("\n\nQuantized TFLite:\n")
        f.write(json.dumps(quant_tflite_details, indent=4, default=str))
        f.write("\n\n")

        f.write("Confusion matrix\n")
        f.write("-" * 70 + "\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification report\n")
        f.write("-" * 70 + "\n")
        f.write(report_text)
        f.write("\n\n")

        f.write("Saved files\n")
        f.write("-" * 70 + "\n")
        f.write(f"H5 model: {h5_path}\n")
        f.write(f"Keras model: {keras_path}\n")
        f.write(f"Float TFLite model: {float_path}\n")
        f.write(f"Quantized TFLite model: {quant_path}\n")
        f.write(f"Run folder: {RUN_DIR}\n")

    print(f"Saved run summary: {summary_path}")


def save_config():
    config = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "clip_samples": CLIP_SAMPLES,
        "labels": LABELS,
        "random_seed": RANDOM_SEED,
        "dataset_dir": str(DATASET_DIR),
        "model_dir": str(MODEL_DIR),
        "run_dir": str(RUN_DIR),
        "model_type": "small_2d_cnn",
        "feature_type": "log_magnitude_stft_spectrogram",
        "stft_frame_length": 480,
        "stft_frame_step": 320,
        "stft_fft_length": 512,
    }

    config_path = RUN_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"Saved config: {config_path}")


# --------------------------------------------------
# Main training flow
# --------------------------------------------------
def main():
    print(f"Run folder: {RUN_DIR}\n")

    save_config()

    X, y, dataset_file_counts, dataset_sample_counts = build_dataset()

    # Split into train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    split_shapes = {
        "X_train": X_train.shape,
        "y_train": y_train.shape,
        "X_val": X_val.shape,
        "y_val": y_val.shape,
        "X_test": X_test.shape,
        "y_test": y_test.shape,
    }

    print("\nSplit sizes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    input_shape = X_train.shape[1:]
    print("\nInput shape:", input_shape)

    model = build_cnn_model(input_shape)
    model.summary()
    save_model_summary(model)

    print("\nTraining model...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks
    )

    save_training_history(history)

    print("\nEvaluating Keras model...")

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=LABELS)
    report_dict = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report_text)

    save_confusion_matrix_outputs(cm, report_text, report_dict)
    save_predictions(y_test, y_pred)

    print("\nSaving model files...")

    h5_path = MODEL_DIR / "keyword_model.h5"
    keras_path = MODEL_DIR / "keyword_model.keras"
    labels_path = MODEL_DIR / "labels.txt"

    model.save(h5_path)
    model.save(keras_path)

    with open(labels_path, "w", encoding="utf-8") as f:
        for label in LABELS:
            f.write(label + "\n")

    # Also save labels inside the run folder.
    with open(RUN_DIR / "labels.txt", "w", encoding="utf-8") as f:
        for label in LABELS:
            f.write(label + "\n")

    print(f"Saved H5 model:    {h5_path}")
    print(f"Saved Keras model: {keras_path}")
    print(f"Saved labels:      {labels_path}")
    print(f"Saved run labels:  {RUN_DIR / 'labels.txt'}")

    float_path, quant_path = save_tflite_models(model, X_train)

    print("\nEvaluating TFLite models...")
    float_tflite_acc, float_tflite_pred, float_tflite_details = evaluate_tflite_model(
        float_path, X_test, y_test
    )
    quant_tflite_acc, quant_tflite_pred, quant_tflite_details = evaluate_tflite_model(
        quant_path, X_test, y_test
    )

    print(f"Float TFLite test accuracy:     {float_tflite_acc:.4f}")
    print(f"Quantized TFLite test accuracy: {quant_tflite_acc:.4f}")

    print("\nModel parameter count:", model.count_params())

    save_run_summary(
        input_shape=input_shape,
        dataset_file_counts=dataset_file_counts,
        dataset_sample_counts=dataset_sample_counts,
        split_shapes=split_shapes,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        float_tflite_acc=float_tflite_acc,
        quant_tflite_acc=quant_tflite_acc,
        model=model,
        h5_path=h5_path,
        keras_path=keras_path,
        float_path=float_path,
        quant_path=quant_path,
        cm=cm,
        report_text=report_text,
        report_dict=report_dict,
        float_tflite_details=float_tflite_details,
        quant_tflite_details=quant_tflite_details
    )

    print("\nDone.")
    print(f"All run outputs saved in: {RUN_DIR}")


if __name__ == "__main__":
    log_path = RUN_DIR / "full_console_output.txt"

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)

        try:
            main()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print(f"\nFull console output saved to: {log_path}")