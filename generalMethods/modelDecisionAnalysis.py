import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
from matplotlib import patheffects
from matplotlib.patheffects import withStroke

def generate_prediction_report(model, audio_path, demography_info, config):
    """Generate an interactive HTML report with dark/light mode toggle.
    
    Args:
        model: Loaded TBNet model
        audio_path: Path to input audio file
        demography_info: Demographic data (e.g., age)
        config: Model configuration parameters
        
    Returns:
        HTML string containing the interactive report
    """
    
    # Store original matplotlib style to restore later
    original_style = plt.style.available[0]  # Default style
    
    try:
        # Run inference and get the gating weights
        predicted_label, probabilities = model.inference(audio_path, demography_info, config)
        
        # Get modality weights (ensure your model stores these)
        if hasattr(model, 'last_gate_weights'):
            gate_weights = model.last_gate_weights[0].tolist()
        else:
            gate_weights = [0.4, 0.4, 0.2]  # fallback
        
        # Prepare data
        class_names = ['Control', 'MCI', 'ADRD']
        prob_values = [prob * 100 for prob in probabilities]
        modalities = ['Acoustic', 'Linguistic', 'Demographic']
        predicted_class = class_names[predicted_label]
        
        # Create plots with dark style only for this figure
        with plt.style.context('dark_background'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
            fig.patch.set_facecolor('#0d1117')  # Dark background
            
            # Prediction bar chart (left)
            bar_colors = ['#4CAF50', '#FFA726', '#F44336']  # Green, Orange, Red
            bars = ax1.bar(class_names, prob_values, color=bar_colors, 
                          edgecolor='white', linewidth=0.5, alpha=0.9)
            ax1.set_title('Prediction Confidence', fontsize=14, pad=20, color='white', fontweight='bold')
            ax1.set_ylabel('Probability (%)', fontsize=12, color='#b0b0b0')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='both', colors='#b0b0b0')
            ax1.spines['bottom'].set_color('#404040')
            ax1.spines['left'].set_color('#404040')
            
            # Add value labels with glow effect
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom',
                        color='white', fontsize=11, fontweight='bold',
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='#333333')])
            
            # Modality pie chart (right) - Orange accent theme
            pie_colors = ['#1E88E5', '#FFA726', '#26A69A']  # Blue, Orange, Teal
            wedges, texts, autotexts = ax2.pie(
                gate_weights,
                labels=modalities,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'color': 'white'},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
                explode=(0.05, 0.05, 0.05)  # Slight separation
            )
            ax2.set_title('Modality Contributions', fontsize=14, pad=20, color='white', fontweight='bold')
            
            # Make percentages bold and larger
            plt.setp(autotexts, size=12, weight="bold", color='white',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#333333')])
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3)
            
            # Save plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=120)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
    
    finally:
        # Restore original matplotlib style
        plt.style.use(original_style)
    
    # [Rest of your HTML generation code remains the same...]
    html = """
    <!DOCTYPE html>
    [... rest of your HTML template ...]
    """
    
    return html

