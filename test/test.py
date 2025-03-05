import sys
import argparse
import torch
import librosa
import yaml
from typing import Optional, Tuple, Dict, Any
# Add custom paths to sys.path (if needed)
sys.path.append("SpeechCARE_Acoustic_Explainability_Framework")

# Import custom modules
from models.ModelWrapper import ModelWrapper
from utils.Config import Config
from SHAP.Shap import AcousticShap
from pauseExtraction.Pause_extraction import PauseExtraction


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test Acoustic Explainability Framework")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the PyTorch model to use as the pretrained model.")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to the audio file of the selected sample.")
    parser.add_argument("--demography_info", type=float, required=True,
                        help="Demographic information of shape [batch_size, 1].")

    # Optional arguments
    parser.add_argument("--fig_save_path", type=str, default=None,
                        help="Path to save the figures.")
    parser.add_argument("--word_segments", type=str, default="",
                        help="Path to saved JSON file of the word segments.")
    parser.add_argument("--energy_threshold", type=float, default=0.001,
                        help="Energy threshold for pause refinement.")
    parser.add_argument("--min_pause_duration", type=float, default=0.2,
                        help="Minimum pause duration for pause refinement.")
    parser.add_argument("--expansion_threshold", type=float, default=0.03,
                        help="Expansion threshold for pause refinement.")
    parser.add_argument("--plot", action="store_true",
                        help="Whether to display the plot.")

    return parser.parse_args()


def initialize_pause_extractor(config_pause, audio_path, word_segments):
    """
    Initialize and configure the PauseExtraction module.

    Args:
        config_pause (Config): Configuration object for pause extraction.
        audio_path (str): Path to the audio file.
        word_segments (str): Path to the word segments JSON file.

    Returns:
        PauseExtraction: Initialized pause extractor.
    """
    pause_extractor = PauseExtraction(config_pause, audio_path, word_segments)
    return pause_extractor

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def refine_pauses(pause_extractor, sr, energy_threshold, min_pause_duration, expansion_threshold):
    """
    Extract and refine pauses from the audio.

    Args:
        pause_extractor (PauseExtraction): Pause extraction module.
        sr (int): Sampling rate of the audio.
        energy_threshold (float): Energy threshold for pause refinement.
        min_pause_duration (float): Minimum pause duration.
        expansion_threshold (float): Expansion threshold for pause refinement.

    Returns:
        list: Refined pauses.
    """
    pauses = pause_extractor.extract_pauses()
    marked_pauses = pause_extractor.mark_pauses(pauses)
    refine_pause = pause_extractor.refine_pauses(
        marked_pauses, sr=sr, energy_threshold=energy_threshold,
        min_pause_duration=min_pause_duration, expansion_threshold=expansion_threshold
    )
    return refine_pause


def initialize_model(config, model_checkpoint):
    """
    Initialize the model wrapper and load the pretrained model.

    Args:
        config (Config): Configuration object for the model.
        model_checkpoint (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded model.
    """
    wrapper = ModelWrapper(config)
    model = wrapper.get_model(model_checkpoint)
    return model

 
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load pause configuration from YAML file
    config_path = "SpeechCARE_Acoustic_Explainability_Framework/data/pause_config.yaml"  # Path to your YAML configuration file
    config_pause = Config(load_config_from_yaml(config_path))
  

    # Initialize pause extractor
    pause_extractor = initialize_pause_extractor(config_pause, args.audio_path, args.word_segments)

    # Load audio and refine pauses
    y, sr = librosa.load(args.audio_path, sr=48000)
    refine_pause = refine_pauses(
        pause_extractor, sr, args.energy_threshold,
        args.min_pause_duration, args.expansion_threshold
    )

 
    # Load model configuration from YAML file
    config_path = "SpeechCARE_Acoustic_Explainability_Framework/data/model_config.yaml"  # Path to your YAML configuration file
    config = Config(load_config_from_yaml(config_path))

    # Initialize and load the model
    model = initialize_model(config, args.model_checkpoint)

    # Initialize SHAP explainer
    shap = AcousticShap(model)

    # Convert demographic information to a tensor
    demography_tensor = torch.tensor(args.demography_info, dtype=torch.float16).reshape(1, 1)

    # Generate and visualize the SHAP spectrogram
    modified_spectrogram, freq_shann_ent = shap.get_speech_spectrogram(
        args.audio_path, demography_tensor, config,
        pauses=refine_pause, fig_save_path=args.fig_save_path, plot=args.plot
    )


if __name__ == "__main__":
    main()