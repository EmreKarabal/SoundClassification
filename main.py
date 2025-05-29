import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# CSV dosyası için
from datetime import datetime
import os

model = hub.load('https://tfhub.dev/google/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
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

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

# Load audio file
wav_file_name ='sound.wav'
sample_rate, wav_data = wavfile.read(wav_file_name)
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Convert to float32 and normalize if needed
wav_data = wav_data.astype(np.float32)
if wav_data.max() > 1.0:
    wav_data = wav_data / 32768.0  # Normalize if it's int16 format

# Convert stereo to mono if necessary
if len(wav_data.shape) > 1:
    print(f"Converting from {wav_data.shape[1]} channels to mono")
    wav_data = np.mean(wav_data, axis=1)  # Average the channels

# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')

# Run inference on the audio file
print("Running YAMNet inference...")
scores, embeddings, spectrogram = model(wav_data)
scores_np = scores.numpy()

# Calculate mean scores across all frames
mean_scores = np.mean(scores_np, axis=0)

# Calculate frame duration and create time array
frame_duration = duration / len(scores_np)
time_frames = np.arange(len(scores_np)) * frame_duration

# For each time frame, get the top prediction with confidence threshold
print("\nDetected sounds in chronological order:")
last_prediction = ""
min_confidence = 0.2  # Minimum confidence threshold düşürüldü
min_duration = 0.3    # Minimum duration düşürüldü
current_prediction_start = 0
current_prediction_confidence = 0

# CSV dosyası için hazırlık
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'sound_predictions_{timestamp}.csv'
predictions_list = []

for i, frame_scores in enumerate(scores_np):
    top_class_index = np.argmax(frame_scores)
    prediction = class_names[top_class_index]
    confidence = frame_scores[top_class_index]
    
    # Only process predictions above confidence threshold
    if confidence > min_confidence:
        if prediction != last_prediction:
            if last_prediction:
                # Print the previous sound segment
                duration_sec = (i * frame_duration) - (current_prediction_start * frame_duration)
                if duration_sec >= min_duration:
                    start_time = current_prediction_start * frame_duration
                    end_time = i * frame_duration
                    print(f"Time {start_time:.1f}s - {end_time:.1f}s: "
                          f"{last_prediction} (confidence: {current_prediction_confidence:.3f})")
                    
                    # CSV için veri kaydetme
                    predictions_list.append({
                        'start_time': f"{start_time:.1f}",
                        'end_time': f"{end_time:.1f}",
                        'sound_type': last_prediction,
                        'confidence': f"{current_prediction_confidence:.3f}"
                    })
            
            # Start new prediction tracking
            current_prediction_start = i
            current_prediction_confidence = confidence
            last_prediction = prediction

# Son tahmini işle
if last_prediction and (len(scores_np) - current_prediction_start) * frame_duration >= min_duration:
    start_time = current_prediction_start * frame_duration
    end_time = duration
    print(f"Time {start_time:.1f}s - {end_time:.1f}s: "
          f"{last_prediction} (confidence: {current_prediction_confidence:.3f})")
    
    # Son tahmini CSV için kaydet
    predictions_list.append({
        'start_time': f"{start_time:.1f}",
        'end_time': f"{end_time:.1f}",
        'sound_type': last_prediction,
        'confidence': f"{current_prediction_confidence:.3f}"
    })

# CSV dosyasına yazma
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['start_time', 'end_time', 'sound_type', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for prediction in predictions_list:
        writer.writerow(prediction)

print(f"\nPredictions saved to {csv_filename}")

# Plot the waveform
plt.figure(figsize=(12, 4))
time_axis = np.linspace(0, duration, len(wav_data))
plt.plot(time_axis, wav_data)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.show()

# Plot the predictions over time
plt.figure(figsize=(12, 8))
plt.imshow(scores_np.T, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.ylabel('Audio Classes (521 total)')
plt.xlabel('Time (seconds)')
plt.title('YAMNet Predictions Over Time')
plt.colorbar(label='Prediction Score')

# Add labels for top classes
if len(scores_np) > 0:
    # Show labels for top 10 classes
    top_10_indices = np.argsort(mean_scores)[::-1][:10]
    yticks = []
    ylabels = []
    for idx in top_10_indices:
        yticks.append(idx)
        ylabels.append(f'{class_names[idx]} ({mean_scores[idx]:.2f})')
    
    plt.yticks(yticks[:5], ylabels[:5])  # Show only top 5 to avoid crowding

plt.tight_layout()
plt.show()

print(f"\nAnalysis complete! Processed {duration:.1f} seconds of audio.")