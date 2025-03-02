import sys
sys.path.append("")

import argparse
import torch
import librosa

from SpeechCARE_Acoustic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Acoustic_Explainability_Framework.Config import Config
from SpeechCARE_Acoustic_Explainability_Framework.SHAP.Shap import AcousticShap
from SpeechCARE_Acoustic_Explainability_Framework.pauseExtraction.Pause_extraction import PauseExtraction

def main():
    parser = argparse.ArgumentParser(description="Test Acoustic Explainability Framework")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the PyTorch model to use as the pretrained model.")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to the audio file of the selected sample.")
    parser.add_argument("--demography_info", type=float, required=True,
                        help="Demographic information of shape [batch_size, 1].")
    parser.add_argument("--fig_save_path", type=str, required=False,default =None,
                        help="Path to save the figures.")
                        

    args = parser.parse_args()

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

    wrapper = ModelWrapper(config)
    model = wrapper.get_model(args.model_checkpoint)
    shap = AcousticShap(model)

    # Convert the scalar to a torch.Tensor
    demography_tensor = torch.tensor(args.demography_info,dtype=torch.float16).reshape(1, 1) 

    modified_spectrogram, freq_shann_ent = shap.get_speech_spectrogram(args.audio_path,demography_tensor,config,fig_save_path = None)

    config_pause = Config()
    config_pause.device  = "cpu"
    config_pause.batch_size = 16
    config_pause.compute_type = "int8"
    config_pause.model_id = "large-v3"

    pause_extractor = PauseExtraction(config_pause,args.audio_path)
    pauses = pause_extractor.extract_pauses()
    marked_pauses = pause_extractor.mark_pauses(pauses)

    y, sr = librosa.load(args.audio_path,sr=48000)
    # pause_extractor.plot_spec_pause(modified_spectrogram,sr,y,marked_pauses, save_path = args.fig_save_path)

    refine_pause = pause_extractor.refine_pauses(marked_pauses, sr=sr, energy_threshold=0.001, min_pause_duration=0.2,expansion_threshold = 0.03)
    pause_extractor.plot_spec_pause(modified_spectrogram,sr,y,refine_pause, save_path = args.fig_save_path)



if __name__ == "__main__":
    main()