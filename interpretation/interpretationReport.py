import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Optional, Tuple, Union, Any,List,Dict

def categorize_pauses(num_pauses):
    if num_pauses == 0:
        return "None"
    elif num_pauses == 1:
        return "Single"
    elif num_pauses in [2, 3]:
        return "Few"
    else:
        return "Several"

def categorize_rhythmic_structure(flat_segments: List[Tuple[float, float]]) -> str:
    """
    Categorizes speech rhythm based on flat/monotonous segments in audio.
    
    Args:
        flat_segments: List of (start, end) tuples representing flat segments in seconds.
                       Empty list indicates no flat segments.
    
    Returns:
        One of four rhythm categories:
        - 'Rhythmic': No flat segments
        - 'Relatively Rhythmic': Minimal flat segments (1 segment of 5-6s)
        - 'Less Rhythmic': Some flat segments (2 segments of 5-6s or 1 segment of 6-10s)
        - 'Non-Rhythmic': Significant flat segments (>2 segments or any segment >10s)
    """
    if not flat_segments:
        return "Rhythmic"
    
    durations = [end - start for start, end in flat_segments]
    segment_count = len(durations)
    has_long_segment = any(d > 10 for d in durations)
    has_medium_segment = any(6 < d <= 10 for d in durations)
    all_medium = all(5 <= d <= 6 for d in durations)
    
    if segment_count == 1 and 5 <= durations[0] <= 6:
        return "Relatively Rhythmic"
    elif (segment_count == 2 and all_medium) or has_medium_segment:
        return "Less Rhythmic"
    elif segment_count > 2 or has_long_segment:
        return "Non-Rhythmic"
    return "Rhythmic"


def generate_vocal_analysis_report(
    sample_name: str,
    f0_values: List[float],  
    f3_values: List[float],  
    pause_count: int,
    flat_segments: List[Tuple[float, float]],
    shimmer_value: float,
    shimmer_category: str,
    energy_value: float,
    energy_category: str,
    spectrogram_plot: Optional[str] = None
) -> str:
    """
    Generate an interactive HTML report for vocal feature analysis with clinical interpretation.
    
    Args:
        sample_name: Name/ID of the audio sample
        f0_values: Array of fundamental frequency measurements (Hz) over time
        f3_values: Array of formant frequency F3 measurements (Hz) over time
        pause_count: Number of noun pauses detected
        flat_segments: List of (start,end) timestamps for flat/monotonous segments
        shimmer_value: Shimmer measurement value
        shimmer_category: Shimmer classification category
        energy_value: Vocal energy measurement (dB)
        energy_category: Energy classification category
        spectrogram_plot: Optional base64 encoded spectrogram image
        
    Returns:
        HTML string with formatted analysis report including F0/F3 plots
    """
    # Generate plots for F0 and F3
    f0_plot = generate_plot(f0_values, "Fundamental Frequency (F0)", "Hz", color='#1E88E5')
    f3_plot = generate_plot(f3_values, "Formant Frequency (F3)", "Hz", color='#FFA726')
    
    # Calculate rhythm metrics
    total_flat_duration = sum(end - start for start, end in flat_segments)
    longest_flat = max([end - start for start, end in flat_segments] or [0])
    rhythm_category = categorize_rhythmic_structure(flat_segments)
    pause_category = categorize_pauses(pause_count)
    
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vocal Feature Analysis Report</title>
            <style>
                :root {{
                    --bg-color: #0d1117;
                    --text-color: #e6edf3;
                    --card-bg: #161b22;
                    --border-color: #30363d;
                    --highlight: #FFA726;
                    --accent-blue: #1E88E5;
                    --accent-green: #4CAF50;
                    --accent-teal: #26A69A;
                    --accent-red: #F44336;
                }}
                
                [data-theme="light"] {{
                    --bg-color: #f8f9fa;
                    --text-color: #212529;
                    --card-bg: #ffffff;
                    --border-color: #dee2e6;
                    --highlight: #FF7043;
                }}
                
                body {{
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    background-color: var(--bg-color);
                    color: var(--text-color);
                    margin: 0;
                    padding: 0;
                    transition: all 0.3s ease;
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .feature-section {{
                    background-color: var(--card-bg);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    border: 1px solid var(--border-color);
                }}
                
                .feature-title {{
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 15px;
                    color: var(--highlight);
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 8px;
                }}
                
                .feature-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 15px;
                    margin-bottom: 15px;
                }}
                
                .feature-value {{
                    display: flex;
                    justify-content: space-between;
                }}

                
                .reference-table {{
                    width: 100%;
                    margin-top: 15px;
                    border-collapse: collapse;
                }}
                
                .reference-table th, .reference-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                }}
                
                .chart-container {{
                    background-color: var(--card-bg);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    border: 1px solid var(--border-color);
                }}
                
                .plot-vertical {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .plot {{
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Multimodal Audio Classification</h1>
                    <h2>Clinical Interpretation Report</h2>
                    <p>Analysis for sample: <strong>{sample_name}</strong></p>
                </div>
                
                <div class="feature-section">
                    <div class="feature-title">Noun Pause Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Pause Count:</span>
                            <span><strong>{pause_count}</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{pause_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Pause Count</th>
                            <th>Category</th>
                        </tr>
                        <tr>
                            <td>0 pauses</td>
                            <td>None (0)</td>
                        </tr>
                        <tr>
                            <td>1 pause</td>
                            <td>Single (1)</td>
                        </tr>
                        <tr>
                            <td>2-3 pauses</td>
                            <td>Few (2)</td>
                        </tr>
                        <tr>
                            <td>>3 pauses</td>
                            <td>Several (3)</td>
                        </tr>
                    </table>
                </div>
                
                <div class="feature-section">
                    <div class="feature-title">Rhythmic Structure</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Flat Segments:</span>
                            <span><strong>{len(flat_segments)}</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Total Duration:</span>
                            <span><strong>{total_flat_duration:.2f} sec</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Longest Segment:</span>
                            <span><strong>{longest_flat:.2f} sec</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{rhythm_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Criteria</th>
                            <th>Category</th>
                        </tr>
                        <tr>
                            <td>No flat segments</td>
                            <td>Rhythmic</td>
                        </tr>
                        <tr>
                            <td>1 segment (5-6s)</td>
                            <td>Relatively Rhythmic</td>
                        </tr>
                        <tr>
                            <td>2 segments or 1 >6s</td>
                            <td>Less Rhythmic</td>
                        </tr>
                        <tr>
                            <td>>2 segments or >10s</td>
                            <td>Non-Rhythmic</td>
                        </tr>
                    </table>
                </div>
                
                <div class="feature-section">
                    <div class="feature-title">Shimmer Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{shimmer_value:.2f}%</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{shimmer_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Shimmer Range</th>
                            <th>Category</th>
                        </tr>
                        <tr>
                            <td>0-2.5% (Q1)</td>
                            <td>Stable</td>
                        </tr>
                        <tr>
                            <td>2.5-3.5% (Q2)</td>
                            <td>Almost Stable</td>
                        </tr>
                        <tr>
                            <td>3.5-5% (Q3)</td>
                            <td>Almost Unstable</td>
                        </tr>
                        <tr>
                            <td>>5% (Q4)</td>
                            <td>Unstable</td>
                        </tr>
                    </table>
                </div>
                
                <div class="feature-section">
                    <div class="feature-title">Energy (Frequency Domain)</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{energy_value:.2f} dB</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{energy_category}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Energy Range</th>
                            <th>Category</th>
                        </tr>
                        <tr>
                            <td><55dB (Q1)</td>
                            <td>Low Energy</td>
                        </tr>
                        <tr>
                            <td>55-65dB (Q2)</td>
                            <td>Mild Energy</td>
                        </tr>
                        <tr>
                            <td>65-75dB (Q3)</td>
                            <td>Moderate Energy</td>
                        </tr>
                        <tr>
                            <td>>75dB (Q4)</td>
                            <td>Energetic</td>
                        </tr>
                    </table>
                </div>
            </div>
            
            <script>
                function toggleTheme() {{
                    const html = document.documentElement;
                    const currentTheme = html.getAttribute('data-theme');
                    const toggleBtn = document.querySelector('.theme-toggle');
                    
                    if (currentTheme === 'light') {{
                        html.removeAttribute('data-theme');
                        toggleBtn.innerHTML = `
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                            </svg>
                            <span>Dark Mode</span>
                        `;
                    }} else {{
                        html.setAttribute('data-theme', 'light');
                        toggleBtn.innerHTML = `
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="5"></circle>
                                <line x1="12" y1="1" x2="12" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="23"></line>
                                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                                <line x1="1" y1="12" x2="3" y2="12"></line>
                                <line x1="21" y1="12" x2="23" y2="12"></line>
                                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                            </svg>
                            <span>Light Mode</span>
                        `;
                    }}
                }}
            </script>
        </body>
        </html>
    """

    return html

def generate_plot(values: List[float], title: str, ylabel: str, color: str) -> str:
    """Generate a base64 encoded plot from an array of values"""
    plt.figure(figsize=(8, 4), facecolor='#161b22')
    plt.plot(values, color=color)
    plt.title(title, color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel(ylabel, color='white')
    plt.gca().set_facecolor('#0d1117')
    plt.gca().tick_params(colors='white')
    plt.grid(True, alpha=0.2)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#161b22')
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')



