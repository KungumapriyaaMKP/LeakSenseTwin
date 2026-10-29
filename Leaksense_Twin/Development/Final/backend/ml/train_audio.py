import os
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def extract_audio_features(file_path):
    sample_rate, data = wavfile.read(file_path)
    
    # If stereo, convert to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Convert int16 to float32 normalized between -1.0 and 1.0
    data = data.astype(np.float32) / 32768.0
    
    # Time-domain features
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    crest_factor = peak / (rms + 1e-8)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0) / len(data)
    
    # Frequency-domain features using FFT
    fft_vals = np.abs(np.fft.rfft(data))
    fft_freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
    
    # Avoid log of zero
    fft_vals_epsilon = fft_vals + 1e-8
    
    # Spectral Centroid
    spectral_centroid = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
    
    # Spectral Flatness
    log_mean = np.mean(np.log(fft_vals_epsilon))
    arith_mean = np.mean(fft_vals)
    spectral_flatness = np.exp(log_mean) / (arith_mean + 1e-8)
    
    # Energy in 4 sub-bands
    band_1 = np.mean(fft_vals[(fft_freqs >= 0) & (fft_freqs < 250)])
    band_2 = np.mean(fft_vals[(fft_freqs >= 250) & (fft_freqs < 1000)])
    band_3 = np.mean(fft_vals[(fft_freqs >= 1000) & (fft_freqs < 4000)])
    band_4 = np.mean(fft_vals[(fft_freqs >= 4000)])
    
    # Combine features into list
    # Features must match the exact order during inference!
    features = [
        float(rms),
        float(peak),
        float(crest_factor),
        float(zero_crossings),
        float(spectral_centroid),
        float(spectral_flatness),
        float(band_1),
        float(band_2),
        float(band_3),
        float(band_4)
    ]
    return features

def train_acoustic_model():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(backend_dir, "Data")
    
    good_dir = os.path.join(data_dir, "good")
    leak_dir = os.path.join(data_dir, "leak")
    
    X = []
    y = []
    
    # Process healthy files
    print("Extracting features from healthy files...")
    for filename in os.listdir(good_dir):
        if filename.endswith(".wav"):
            path = os.path.join(good_dir, filename)
            try:
                feats = extract_audio_features(path)
                X.append(feats)
                y.append(0) # 0 for healthy
                print(f"  [OK] {filename} (rms={feats[0]:.4f}, centroid={feats[4]:.1f})")
            except Exception as e:
                print(f"  [ERROR] Failed to process {filename}: {e}")
                
    # Process leak files
    print("\nExtracting features from leaking files...")
    for filename in os.listdir(leak_dir):
        if filename.endswith(".wav"):
            path = os.path.join(leak_dir, filename)
            try:
                feats = extract_audio_features(path)
                X.append(feats)
                y.append(1) # 1 for leak
                print(f"  [OK] {filename} (rms={feats[0]:.4f}, centroid={feats[4]:.1f})")
            except Exception as e:
                print(f"  [ERROR] Failed to process {filename}: {e}")
                
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nDataset size: {len(X)} samples ({np.sum(y == 0)} healthy, {np.sum(y == 1)} leak)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    models_dir = os.path.join(backend_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "acoustic_model.joblib")
    scaler_path = os.path.join(models_dir, "acoustic_scaler.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nSuccess! Saved model to {model_path} and scaler to {scaler_path}")

if __name__ == "__main__":
    train_acoustic_model()
