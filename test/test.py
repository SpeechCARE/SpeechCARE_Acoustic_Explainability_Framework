import sys
import argparse
import torch
import librosa

# Add custom paths to sys.path (if needed)
sys.path.append("")

# Import custom modules
from SpeechCARE_Acoustic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Acoustic_Explainability_Framework.Config import Config
from SpeechCARE_Acoustic_Explainability_Framework.SHAP.Shap import AcousticShap
from SpeechCARE_Acoustic_Explainability_Framework.pauseExtraction.Pause_extraction import PauseExtraction


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

    # Initialize pause extraction configuration
    config_pause = Config()
    config_pause.device = "cpu"
    config_pause.batch_size = 16
    config_pause.compute_type = "int8"
    config_pause.model_id = "large-v3"

    # Initialize pause extractor
    pause_extractor = initialize_pause_extractor(config_pause, args.audio_path, args.word_segments)

    # Load audio and refine pauses
    y, sr = librosa.load(args.audio_path, sr=48000)
    refine_pause = refine_pauses(
        pause_extractor, sr, args.energy_threshold,
        args.min_pause_duration, args.expansion_threshold
    )

    # Initialize model configuration
    SIMPLE_ATTENTION = 16
    config = Config()
    config.seed = 133
    config.bs = 4
    config.epochs = 14
    config.lr = 1e-6
    config.hidden_size = 128
    config.wd = 1e-3
    config.integration = SIMPLE_ATTENTION
    config.num_labels = 3
    config.txt_transformer_chp = config.MGTEBASE
    config.speech_transformer_chp = config.mHuBERT
    config.segment_size = 5
    config.active_layers = 12
    config.demography = 'age_bin'
    config.demography_hidden_size = 128
    config.max_num_segments = 7

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