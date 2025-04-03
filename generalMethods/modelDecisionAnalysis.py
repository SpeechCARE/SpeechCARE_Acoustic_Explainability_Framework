import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_prediction_report(model, audio_path, demography_info, config):
    # Run inference and get the gating weights
    predicted_label, probabilities = model.inference(audio_path, demography_info, config)
    
    if hasattr(model, 'last_gate_weights'):
        gate_weights = model.last_gate_weights[0].tolist()  # Get weights for first sample
    else:
        # Fallback - you'll need to modify your model to store/return these
        gate_weights = [0.4, 0.4, 0.2]  # Example weights
    
    # Get class names and probabilities
    class_names = ['Control', 'MCI', 'ADRD']
    prob_values = [prob * 100 for prob in probabilities]  # Convert to percentages
    
    # Create figures
    plt.figure(figsize=(12, 5))
    
    # Bar chart for predictions
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_names, prob_values, color=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Prediction Probabilities', fontsize=12)
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    # Pie chart for modality contributions
    plt.subplot(1, 2, 2)
    modalities = ['Acoustic', 'Linguistic', 'Demographic']
    colors = ['#2196F3', '#9C27B0', '#009688']
    wedges, texts, autotexts = plt.pie(
        gate_weights,
        labels=modalities,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    plt.title('Modality Contributions', fontsize=12)
    plt.setp(autotexts, size=10, weight="bold")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save combined plot to bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Determine predicted class with max probability
    predicted_class = class_names[predicted_label]
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Prediction Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .result {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
                font-size: 18px;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .details {{
                margin-top: 30px;
                padding: 15px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }}
            .highlight {{
                font-weight: bold;
                color: #2c3e50;
            }}
            .two-columns {{
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
            }}
            .column {{
                width: 48%;
            }}
            .modality-details {{
                margin-top: 15px;
            }}
            .modality {{
                display: flex;
                align-items: center;
                margin: 8px 0;
            }}
            .color-box {{
                width: 20px;
                height: 20px;
                margin-right: 10px;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Model Decision Analysis</h1>
                <p>Audio analysis results for: {audio_path}</p>
            </div>
            
            <div class="result">
                <h2>Predicted Class: <span class="highlight">{predicted_class}</span></h2>
                <p>with {prob_values[predicted_label]:.1f}% confidence</p>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{plot_data}" alt="Analysis Results" style="max-width: 100%;">
            </div>
            
            <div class="two-columns">
                <div class="column">
                    <h3>Class Probabilities:</h3>
                    <ul>
                        <li>Control: {prob_values[0]:.1f}%</li>
                        <li>Mild Cognitive Impairment (MCI): {prob_values[1]:.1f}%</li>
                        <li>Alzheimer's Disease (ADRD): {prob_values[2]:.1f}%</li>
                    </ul>
                </div>
                
                <div class="column">
                    <h3>Modality Contributions:</h3>
                    <div class="modality-details">
                        <div class="modality">
                            <span class="color-box" style="background-color: #2196F3;"></span>
                            Acoustic: {gate_weights[0]*100:.1f}%
                        </div>
                        <div class="modality">
                            <span class="color-box" style="background-color: #9C27B0;"></span>
                            Linguistic: {gate_weights[1]*100:.1f}%
                        </div>
                        <div class="modality">
                            <span class="color-box" style="background-color: #009688;"></span>
                            Demographic: {gate_weights[2]*100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="details">
                <h3>Additional Information:</h3>
                <p><strong>Demographic Factor:</strong> {demography_info}</p>
                <p><strong>Transcription:</strong> {model.transcription}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html