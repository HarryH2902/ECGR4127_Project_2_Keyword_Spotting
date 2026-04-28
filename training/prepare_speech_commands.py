from pathlib import Path
import urllib.request
import tarfile
import random
import shutil

# -----------------------------
# Settings
# -----------------------------
DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

EXTERNAL_DIR = BASE_DIR / "external_data"
ARCHIVE_PATH = EXTERNAL_DIR / "speech_commands_v0.02.tar.gz"
SPEECH_DIR = EXTERNAL_DIR / "speech_commands_v0.02"

TARGET_HAPPY_COUNT = 305
TARGET_UNKNOWN_COUNT = 305

UNKNOWN_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "cat", "dog", "bird", "tree", "zero", "one",
    "two", "three", "four", "five", "six", "seven", "eight", "nine"
]

random.seed(42)


def download_dataset():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH}")
    else:
        print("Downloading Google Speech Commands dataset...")
        print("This is a large download, so it may take a while.")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
        print("Download complete.")

    if SPEECH_DIR.exists() and any(SPEECH_DIR.iterdir()):
        print(f"Dataset already extracted: {SPEECH_DIR}")
    else:
        print("Extracting dataset...")
        SPEECH_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(SPEECH_DIR)
        print("Extraction complete.")


def clear_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for file in folder.glob("*.wav"):
        file.unlink()


def copy_happy():
    src_dir = SPEECH_DIR / "happy"
    dst_dir = DATASET_DIR / "happy"
    clear_folder(dst_dir)

    files = sorted(src_dir.glob("*.wav"))
    random.shuffle(files)
    selected = files[:TARGET_HAPPY_COUNT]

    print(f"Copying {len(selected)} happy files...")

    for i, src in enumerate(selected):
        dst = dst_dir / f"happy_{i:04d}.wav"
        shutil.copy2(src, dst)

    print(f"Done: {dst_dir}")


def copy_unknown():
    dst_dir = DATASET_DIR / "unknown"
    clear_folder(dst_dir)

    all_files = []

    for word in UNKNOWN_WORDS:
        word_dir = SPEECH_DIR / word
        if word_dir.exists():
            for file in word_dir.glob("*.wav"):
                all_files.append((word, file))

    random.shuffle(all_files)
    selected = all_files[:TARGET_UNKNOWN_COUNT]

    print(f"Copying {len(selected)} unknown files...")

    for i, (word, src) in enumerate(selected):
        dst = dst_dir / f"unknown_{word}_{i:04d}.wav"
        shutil.copy2(src, dst)

    print(f"Done: {dst_dir}")


def copy_background():
    src_dir = SPEECH_DIR / "_background_noise_"
    dst_dir = DATASET_DIR / "background"
    clear_folder(dst_dir)

    files = sorted(src_dir.glob("*.wav"))

    print(f"Copying {len(files)} background noise files...")

    for i, src in enumerate(files):
        dst = dst_dir / f"background_{i:04d}.wav"
        shutil.copy2(src, dst)

    print(f"Done: {dst_dir}")


def print_summary():
    print("\nDataset summary:")
    for folder_name in ["flying", "happy", "unknown", "background"]:
        folder = DATASET_DIR / folder_name
        count = len(list(folder.glob("*.wav")))
        print(f"{folder_name:12s}: {count} wav files")


if __name__ == "__main__":
    download_dataset()
    copy_happy()
    copy_unknown()
    copy_background()
    print_summary()