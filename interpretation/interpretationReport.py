import pandas as pd

# Helper functions
def _get_pause_category(count):
    if count == 0: return "None"
    elif count == 1: return "Single"
    elif 2 <= count <= 3: return "Few"
    else: return "Several"

def _get_pause_interpretation(count):
    interpretations = {
        0: "typical fluent speech",
        1: "slightly disrupted speech",
        2: "moderate pause frequency",
        3: "high pause frequency suggestive of cognitive load or speech disorder"
    }
    return interpretations.get(min(count, 3), "abnormal pause pattern")

def _get_shimmer_interpretation(cat):
    return {
        'Stable (Q1)': "normal vocal fold vibration",
        'Nearly Stable (Q2)': "mild vocal instability",
        'Marginally Unstable (Q3)': "moderate voice abnormality",
        'Unstable (Q4)': "severe vocal pathology likely"
    }.get(cat, "undefined stability pattern")

def _get_energy_interpretation(cat):
    return {
        'Balanced (Q1)': "normal spectral distribution",
        'Modulated (Q2)': "slight energy variations",
        'Irregular (Q3)': "noticeable spectral irregularities",
        'Erratic (Q4)': "abnormal energy distribution"
    }.get(cat, "atypical spectral characteristics")

def generate_audio_report(audio_path: str, 
                         pause_count: int, 
                         shimmer_std: float, 
                         shimmer_category: str,
                         freq_energy: float,
                         freq_category: str,
                         quartile_ranges: dict):
    """
    Generates a professional audio analysis report with visual formatting.
    
    Args:
        audio_path: Path/name of audio file
        pause_count: Number of detected pauses
        shimmer_std: Calculated shimmer standard deviation (dB)
        shimmer_category: Quartile category for shimmer
        freq_energy: Frequency domain energy value
        freq_category: Quartile category for frequency energy
        quartile_ranges: Dictionary containing quartile ranges for both features
    """
    # Create category explanation tables
    shimmer_table = pd.DataFrame({
        'Category': ['Stable (Q1)', 'Nearly Stable (Q2)', 
                    'Marginally Unstable (Q3)', 'Unstable (Q4)'],
        'Shimmer Range (dB)': quartile_ranges['shimmer']
    })
    
    energy_table = pd.DataFrame({
        'Category': ['Balanced (Q1)', 'Modulated (Q2)', 
                    'Irregular (Q3)', 'Erratic (Q4)'],
        'Energy Profile': quartile_ranges['freq_energy']
    })
    
    # Generate the report
    report = f"""
    ## 🎵 Audio Analysis Report: `{audio_path.split('/')[-1]}`

    ### 🔊 Vocal Feature Summary
    | Feature               | Value           | Category          |
    |-----------------------|-----------------|-------------------|
    | **Pause Count**       | {pause_count}   | {_get_pause_category(pause_count)} |
    | **Shimmer Stability** | {shimmer_std:.2f} dB | {shimmer_category} |
    | **Spectral Energy**   | {freq_energy:.2f}   | {freq_category} |

    ### 📊 Quartile Reference Ranges
    **Shimmer Stability Categories:**
    {shimmer_table.to_markdown(index=False)}

    **Spectral Energy Categories:**
    {energy_table.to_markdown(index=False)}

    ### 📝 Interpretation Notes
    - Pause patterns suggest: {_get_pause_interpretation(pause_count)}
    - Shimmer stability indicates: {_get_shimmer_interpretation(shimmer_category)}
    - Energy distribution shows: {_get_energy_interpretation(freq_category)}
    """
    
    return report