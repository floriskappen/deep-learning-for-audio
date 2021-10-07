import os
import librosa
import math
import json

DATASET_PATH = "Data/genres_original"
JSON_PATH = "../2_model/data.json"

SAMPLE_RATE = 22050
DURATION = 30 # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

  # Dictionary to store data
  data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
  }

  num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
  expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2

  # Loop through all the genres
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    
    # Ensure that we're not at the root level
    if dirpath is not dataset_path:
      
      # Save the semantic label
      dirpath_components = dirpath.split("\\") # genre/blues -> ["genre", "blues"]
      semantic_label = dirpath_components[-1]
      data["mapping"].append(semantic_label)
      print(f"\nProcessing {semantic_label}")

      # Process files for a specific genre
      for f in filenames:

        # Load audio file
        file_path = os.path.join(dirpath, f)
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Process segments extracting mfcc and sorting data
        for s in range(num_segments):
          start_sample = num_samples_per_segment * s # s=0 -> 0
          finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment
          
          mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                      sr=sr,
                                      n_fft=n_fft,
                                      n_mfcc=n_mfcc,
                                      hop_length=hop_length)

          mfcc = mfcc.T

          # Store mfcc for segment if it has the expected length
          if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(i-1)
            print(f"{file_path}, segment:{s+1}")
  with open(json_path, "w") as fp:
    json.dump(data, fp, indent=4)

if __name__ == "__main__":
  save_mfcc(dataset_path=DATASET_PATH, json_path=JSON_PATH, num_segments=10)