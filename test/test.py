import sys
import argparse
import torch
import librosa
from typing import Optional, Tuple, Dict, Any
# Add custom paths to sys.path (if needed)
sys.path.append("SpeechCARE_Acoustic_Explainability_Framework")

# Import custom modules
from model.ModelWrapper import ModelWrapper
from utils.Config import Config
from utils.Utils import load_yaml_file
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

 
def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load pause configuration from YAML file
    config_path = "SpeechCARE_Acoustic_Explainability_Framework/data/pause_config.yaml"  # Path to your YAML configuration file
    config_pause = Config(load_yaml_file(config_path))

    pause_extractor = PauseExtraction(config_pause, args.audio_path, args.word_segments)
    pauses = pause_extractor.extract_pauses(sr = 48000, energy_threshold = args.energy_threshold,
                                            min_pause_duration = args.min_pause_duration,
                                            expansion_threshold = args.expansion_threshold,
                                            marked = True, refined = True)

 
    # Load model configuration from YAML file
    config_path = "SpeechCARE_Acoustic_Explainability_Framework/data/model_config.yaml"  # Path to your YAML configuration file
    config = Config(load_yaml_file(config_path))

    # Initialize and load the model
    wrapper = ModelWrapper(config)
    model = wrapper.get_model( args.model_checkpoint)

    # Initialize SHAP explainer
    shap = AcousticShap(model)

    # Convert demographic information to a tensor
    demography_tensor = torch.tensor(args.demography_info, dtype=torch.float16).reshape(1, 1)

    # Generate and visualize the SHAP spectrogram
    modified_spectrogram, freq_shann_ent = shap.get_speech_spectrogram(
        args.audio_path, demography_tensor, config,
        pauses=pauses, fig_save_path=args.fig_save_path, plot=args.plot
    )


if __name__ == "__main__":
    main()