import sys
sys.path.append("")

import argparse

from SpeechCARE_Acoustic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Acoustic_Explainability_Framework.Config import Config

def main():
    # parser = argparse.ArgumentParser(description="Test Acoustic Explainability Framework")
    # parser.add_argument("--feature_extractor", type=str, required=True,
    #                     help="Path to the PyTorch model to use as the feature extractor")

    # args = parser.parse_args()
    SIMPLE_ATTENTION = 16
    config = Config()
    config.max_num_segments = 100
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

    wrapper = ModelWrapper(config)
    model = wrapper.get_model("model_checkpoint.pth")

if __name__ == "__main__":
    main()