import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
from scipy.signal import resample
from scipy.io import wavfile
from datetime import datetime
import os
import winsound
import time

# Define dangerous animal sounds
DANGEROUS_SOUNDS = ["Roar", "Grunt", "Growling", "Wolf", "Howl", "Roaring cats (lions, tigers)"]

# Audio parameters
CHUNK_DURATION = 0.96  # seconds - YAMNet's optimal window size
RATE = 16000  # Sample rate expected by YAMNet

model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def process_audio_chunk(waveform, timestamp):
    """Process a chunk of audio data."""
    # Run inference
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    
    # Get top 3 predictions for each frame
    top_predictions = []
    for frame_scores in scores_np:
        top_indices = np.argsort(frame_scores)[-3:][::-1]  # Get top 3 indices
        for idx in top_indices:
            if frame_scores[idx] > 0.1:  # Only include if confidence > 0.1
                top_predictions.append((class_names[idx], frame_scores[idx]))
    
    # Process unique predictions
    processed_predictions = {}
    for prediction, confidence in top_predictions:
        if prediction not in processed_predictions or confidence > processed_predictions[prediction]:
            processed_predictions[prediction] = confidence
    
    # Check predictions and log them
    for prediction, confidence in processed_predictions.items():
        # Check for dangerous sounds
        if any(sound.lower() in prediction.lower() for sound in DANGEROUS_SOUNDS):
            print(f"\n⚠️ ALERT! Detected potentially dangerous animal sound: {prediction}")
            print(f"Time: {timestamp:.1f}s, Confidence: {confidence:.3f}\n")
            # Play alert sound (two beeps)
            winsound.Beep(1000, 100)  # 1000 Hz for 300ms
            winsound.Beep(1000, 200)  # Second beep
        
        # Only print predictions with good confidence
        if confidence > 0.25:
            print(f"Time {timestamp:.1f}s: {prediction} (confidence: {confidence:.3f})")
            
            # Save to CSV
            with open(f'realtime_predictions_{start_time}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, prediction, confidence])

def main():
    global start_time
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load the full audio file
    wav_file_name = 'inj.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name)
    
    #convert to 16kHz if not already
    if sample_rate != RATE:
        print(f"Resampling audio from {sample_rate} Hz to {RATE} Hz...")
        num_samples = int(len(wav_data) * RATE / sample_rate)
        wav_data = resample(wav_data, num_samples)
        sample_rate = RATE

    # Convert to mono if stereo
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    
    # Convert to float32 and normalize
    wav_data = wav_data.astype(np.float32)
    if wav_data.max() > 1.0:
        wav_data = wav_data / 32768.0
    
    # Calculate chunk size in samples
    chunk_size = int(RATE * CHUNK_DURATION)
    total_chunks = len(wav_data) // chunk_size
    
    # Create CSV file with headers
    with open(f'realtime_predictions_{start_time}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'sound_type', 'confidence'])
    
    print(f"\n🎵 Processing {wav_file_name}")
    print(f"Total duration: {len(wav_data)/RATE:.1f} seconds")
    print("Analyzing audio...\n")
    
    try:
        for chunk_idx in range(total_chunks):
            start = time.time()
            # Get the chunk of audio
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            chunk = wav_data[start_idx:end_idx]
            
            # Process the chunk
            process_audio_chunk(chunk, chunk_idx * CHUNK_DURATION)
            end = time.time()
            print(f"Processed chunk {chunk_idx + 1}/{total_chunks} in {end - start:.2f} seconds")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Processing stopped by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
    finally:
        print("\n✅ Processing complete and results saved to CSV")

if __name__ == "__main__":
    main()