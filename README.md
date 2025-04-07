# Acoustic Explainability Framework

This repository introduces a model-agnostic explainability framework designed for acoustic deep learning pipelines. The framework integrates advanced visualization techniques, such as spectrogram analysis and SHAP (SHapley Additive exPlanations), to identify and visualize acoustic cues associated with desired model outcomes (e.g., cognitive impairment). It comprises three complementary explanation layers:

- **Audio Spectrogram:** Visualizes the energy of frequency components over time.
- **SHAP-based Spectrogram:** Highlights time spans that are most informative for the predicted outcome.
- **Explainable Acoustic Features:** Provides deeper insights into how key acoustic features (Fundamental Frequency [F0] , Third Formant Frequency [F3], Shimmer, Energy of Frequency Domain, Rhythmic Structure, Pause count) relate to the desired outcome (e.g., cognitive status).

The output of this framework is a spectrogram visualization that emphasizes informative acoustic and temporal features, accompanied by human-interpretable explanations.

The framework also includes linguistic explainability. For more details, refer to the [Linguistic Explainability Framework repository](https://github.com/SpeechCARE/SpeechCARE_Linguistic_Explainability_Framework.git).

Below is a sample output of our acoustic explainability framework applied to a classification task from the [SpeechCARE challenge](https://github.com/SpeechCARE), where the subject's class was Mild Cognitive Impairment (MCI).

![Example Output](figs/qnvo_spectrogram.png)

## ⚙️ Setup

To run this project successfully, you need to install **Python 3.10.0**.

### ✅ Step 1: Install Python 3.10.0

If you're using **Linux** or **macOS**, run the following commands:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

Then make sure to use Python 3.10:

```bash
python3.10 --version
```

If you're using **Windows**, download Python 3.10.0 from the official site:  
📎 [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

## 📁 Repository Structure

```
├── data/                       # Contains necessary data
├── dataset/                    # Dataset architecture
├── generalMethods/             # General-purpose methods used
├── interpretation/             # Scripts for reporting interpretation results
├── model/                      # Model architecture
├── pauseExtraction/             # Contains code to extract pauses from audio input
├── SHAP/             # SHAP-based explainability methods for model interpretation
├── test/                 # A sample script for using the explanation on acoustic data
├── utils/                      # Utility scripts for preprocessing and evaluation
├── requirements.txt              # Dependencies for the project
```

## 📌 Notes

- Ensure that the dataset paths and model checkpoints are correctly set.

## 🔗 References
