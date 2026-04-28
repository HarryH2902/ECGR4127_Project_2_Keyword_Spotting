import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path
import time

SAMPLE_RATE = 16000
DURATION = 1.5  # seconds
COUNTDOWN = 1.0


def get_next_index(out_dir, word):
    existing = list(out_dir.glob(f"{word}_*.wav"))

    max_index = -1
    for file in existing:
        try:
            number_part = file.stem.split("_")[-1]
            max_index = max(max_index, int(number_part))
        except ValueError:
            pass

    return max_index + 1


def record_word(word, num_samples):
    out_dir = Path("dataset") / word
    out_dir.mkdir(parents=True, exist_ok=True)

    start_index = get_next_index(out_dir, word)

    print(f"\nRecording {num_samples} samples of '{word}'")
    print(f"Each recording is {DURATION} seconds.")
    print(f"Starting at file number: {start_index:04d}")
    print("Say the word when you see RECORDING NOW.\n")

    for i in range(num_samples):
        filename = out_dir / f"{word}_{start_index + i:04d}.wav"

        input(f"Press Enter. Say '{word}' when you see RECORDING NOW... ")

        print("Get ready...")
        time.sleep(COUNTDOWN)

        print("RECORDING NOW")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        write(filename, SAMPLE_RATE, audio)
        print(f"Saved {filename}\n")


if __name__ == "__main__":
    word = input("Word to record: ").strip().lower()
    num_samples = int(input("Number of samples: "))
    record_word(word, num_samples)