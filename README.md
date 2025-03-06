# Model Explainability for Acoustic Inputs

This repository implements an **acoustic transformer-based classification pipeline** for distinguishing between individuals with **Alzheimer’s Disease and Related Dementias (ADRD)**, **Mild Cognitive Impairment (MCI)**, and **healthy controls** based on audio recordings. To enhance interpretability, we provide tools to explain the model's decision-making process by calculating **SHAP (SHapley Additive exPlanations)** values for acoustic inputs. This allows us to identify and visualize the specific parts of the audio signal that the model attends to most when making predictions.

## 🚀 Installation

First, install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuring `*.yml`

Before starting the training process, update the **`data/model_config.yml`** and **`data/pause_config.yml`** files with the appropriate paths and settings.

### ✅ Set Pretrained Checkpoints

Choose a pretrained acoustic transformer model by specifying its checkpoint in the configuration file. The pipeline supports various self-supervised speech models:

```yaml
# mHuBERT: Multilingual HuBERT model for robust speech representation learning
speech_transformer_chp: "utter-project/mHuBERT-147"
```

```yaml
# wav2vec 2.0: Self-supervised model trained on 960 hours of English speech
speech_transformer_chp: "facebook/wav2vec2-base-960h"
```

```yaml
# HuBERT: Hidden-unit BERT model trained on the LibriSpeech 960h dataset
speech_transformer_chp: "facebook/hubert-base-ls960"
```

### ✅ Set Training Hyperparameters

Change other training parameters or model configs like epoch, learning rate and etc.

---

## 🛠️ Usage

To use the provided explainability methods (SHAP) on an acoustic input, you can run the `test.py` file using the following bash script. This script generates explanations for a given audio sample and saves the results.

### Running the Script

Use the following command to run the `test.py` file:

```bash
!python SpeechCARE_Acoustic_Explainability_Framework/test/test.py --model_checkpoint $CHECKPOINTS_FILE \
                                                                 --audio_path $AUDIO_PATH \
                                                                 --demography_info $DEMOGRAPHIC_INFO \
                                                                 --fig_save_path $FIG_SAVE_PATH \
                                                                 --word_segments $WORD_SEGMENTS_PATH \
                                                                 --min_pause_duration 0.15
```

### Arguments

- **`--model_checkpoint`**:  
  Path to the pretrained TBNet model weights. This file contains the trained model parameters required for inference.

- **`--audio_path`**:  
  Path to the audio sample for which you want to generate explanations. The audio file should be in a supported format (e.g., WAV).

- **`--demography_info`**:  
  A scalar value (e.g., age) associated with the audio sample. This information can be used as additional input for the model, if required.

- **`--fig_save_path`**:  
  Path to save the generated spectrogram image with SHAP values visualized. This image highlights the parts of the audio signal that the model attended to most.

- **`--word_segments` (optional)**:  
  A JSON file containing the words in the audio file along with their respective start and end times. This file is used to detect pauses in the audio, which are important indicators for classification. The JSON file should have the following format:
  ```json
  [
    { "word": "example", "start": 0.0, "end": 0.5 },
    { "word": "audio", "start": 0.6, "end": 1.0 }
  ]
  ```

## 📁 Repository Structure

```
├── data/                       # Contains necessary data
├── dataset/                    # Dataset architecture
├── models/                      # Model architecture
├── pauseExtraction/             # Contains code to extract pauses from audio input
├── utils/                      # Utility scripts for preprocessing and evaluation
├── test/                 # A sample script for using the explanation on acoustic data
├── result.ipynb                      # A notebook sample to show the output of the explanation method used
├── requirements.txt              # Dependencies for the project
```

---
