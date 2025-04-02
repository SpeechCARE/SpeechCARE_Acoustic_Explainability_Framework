"""
Acoustic Feature Analysis Module

This module provides functionality for analyzing acoustic features from audio files,
including energy calculations, shimmer analysis, and comparative feature categorization.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import opensmile
import scipy.signal as signal
from typing import Tuple, Dict, Union


# Constants
DEFAULT_CUTOFF_FREQ = 8000  # Hz
DEFAULT_FILTER_ORDER = 5
FEATURE_COLUMNS = [
    'time_domain_energy',
    'frequency_domain_energy',
    'mean_shimmer',
    'std_shimmer'
]


def remove_wav_extension(filename: str) -> str:
    """Remove .wav extension from filename if present.
    
    Args:
        filename: Input filename (e.g., 'audio.wav')
        
    Returns:
        Filename without .wav extension (e.g., 'audio')
    """
    return filename.replace('.wav', '')


def get_audio_path(uid: str, audio_dir: str) -> str:
    """Construct full path to audio file from unique ID.
    
    Args:
        uid: Unique identifier for audio file
        audio_dir: Directory containing audio files
        
    Returns:
        Full path to audio file (e.g., '/path/to/audio_dir/uid.mp3')
    """
    audio_file = f"{uid}.mp3"
    return os.path.join(audio_dir, audio_file)


def apply_lowpass_filter(
    waveform: np.ndarray,
    sampling_rate: int,
    cutoff_freq: float = DEFAULT_CUTOFF_FREQ,
    order: int = DEFAULT_FILTER_ORDER
) -> np.ndarray:
    """Apply low-pass Butterworth filter to audio signal.
    
    Args:
        waveform: Input audio signal
        sampling_rate: Sampling rate of audio signal (Hz)
        cutoff_freq: Cutoff frequency for low-pass filter (Hz)
        order: Filter order
        
    Returns:
        Filtered audio signal
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, waveform)


def compute_features(audio_path: str) -> Tuple[float, float, float, float]:
    """Compute acoustic features from audio file.
    
    Features include:
    - Time and frequency domain energy (in dB)
    - Shimmer statistics (mean and std)
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple containing:
        - time_energy_db: Time-domain energy in dB
        - freq_energy_db: Frequency-domain energy in dB
        - shimmer_mean: Mean shimmer value
        - shimmer_std: Standard deviation of shimmer
    """
    # Load and preprocess audio
    audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
    filtered_audio = apply_lowpass_filter(audio_data, sr)
    
    # Energy calculations
    energy_time = np.sum(filtered_audio ** 2)
    energy_freq = np.sum(np.abs(np.fft.rfft(filtered_audio)) ** 2)
    
    # Convert to dB scale with small epsilon to avoid log(0)
    time_energy_db = 10 * np.log10(energy_time + 1e-12)
    freq_energy_db = 10 * np.log10(energy_freq + 1e-12)
    
    # OpenSMILE feature extraction
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_signal(filtered_audio, sampling_rate=sr)
    
    # Extract shimmer features
    shimmer_mean = features['shimmerLocaldB_sma3nz_amean'].values[0]
    shimmer_std = features['shimmerLocaldB_sma3nz_stddevNorm'].values[0]
    
    return time_energy_db, freq_energy_db, shimmer_mean, shimmer_std


def _ensure_features_exist(
    data_df: pd.DataFrame,
    feature_columns: list = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Ensure feature columns exist in DataFrame, computing them if necessary.
    
    Args:
        data_df: Input DataFrame containing audio paths
        feature_columns: List of expected feature columns
        
    Returns:
        DataFrame with guaranteed feature columns
    """
    if not all(col in data_df.columns for col in feature_columns):
        tqdm.pandas(desc="Extracting audio features")
        data_df['features'] = data_df['path'].progress_apply(compute_features)
        data_df[feature_columns] = pd.DataFrame(
            data_df['features'].tolist(),
            index=data_df.index
        )
        data_df.drop(columns=['features'], inplace=True)
    return data_df


def _calculate_value_ranges(values: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Calculate quartile ranges for given values.
    
    Args:
        values: Array of numerical values
        
    Returns:
        Dictionary mapping category names to (start, end) ranges
    """
    min_value = values.min()
    max_value = values.max()
    category_width = (max_value - min_value) / 4
    
    return {
        "(0) Very Low": (min_value, min_value + category_width),
        "(1) Low": (min_value + category_width, min_value + 2 * category_width),
        "(2) Moderate": (min_value + 2 * category_width, min_value + 3 * category_width),
        "(3) High": (min_value + 3 * category_width, max_value)
    }


def analyze_column_single(
    data_df: pd.DataFrame,
    test_path: str,
    column_name: str,
    feature_columns: list = FEATURE_COLUMNS
) -> Dict[str, Union[str, float]]:
    """Analyze a test audio file relative to training data distribution.
    
    Args:
        data_df: DataFrame containing training audio features/paths
        test_path: Path to test audio file
        column_name: Feature column to analyze
        feature_columns: List of feature columns
        
    Returns:
        Dictionary containing:
        - 'value': The test value
        - 'category': Quartile category
        - 'ranges': Dictionary of all category ranges
    """
    # Input validation
    if column_name not in feature_columns:
        raise ValueError(f"Column {column_name} not in available features: {feature_columns}")
    
    # Ensure features exist in training data
    data_df = _ensure_features_exist(data_df, feature_columns)
    
    # Compute test features
    test_features = compute_features(test_path)
    test_sample_df = pd.DataFrame([test_features], columns=feature_columns)
    test_value = test_sample_df[column_name].values[0]
    
    # Calculate value ranges
    train_values = data_df[column_name]
    ranges = _calculate_value_ranges(train_values)
    
    # Determine category
    category = "Unknown"
    for cat_name, (start, end) in ranges.items():
        if start <= test_value < end:
            category = cat_name
            break
    
    # Print results
    print("\nValue ranges for each category:")
    for cat_name, (start, end) in ranges.items():
        print(f"{cat_name}: {start:.2f} - {end:.2f}")
    
    print(f"\nThe value of {column_name} for:")
    print(f"{test_value:.2f} => Quartile: '{category}'")
    
    return {
        'value': test_value,
        'category': category,
        'ranges': ranges
    }
