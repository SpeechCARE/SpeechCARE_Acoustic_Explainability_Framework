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
    f0_analysis: Dict[str, Union[str, float, Dict]],  # {'value': float, 'category': str, 'ranges': Dict}
    f3_analysis: Dict[str, Union[str, float, Dict]], 
    pause_count: int,
    flat_segments: List[Tuple[float, float]],
    shimmer_analysis: Dict[str, Union[str, float, Dict]],
    energy_analysis: Dict[str, Union[str, float, Dict]]
) -> str:
    """
    Generate an interactive HTML report for vocal feature analysis with dynamic ranges.
    
    Args:
        sample_name: Name/ID of the audio sample
        f0_analysis: Fundamental frequency analysis results
        f3_analysis: Formant frequency analysis results
        pause_count: Number of noun pauses detected
        flat_segments: List of (start,end) timestamps for flat/monotonous segments
        shimmer_analysis: Shimmer analysis results
        energy_analysis: Energy analysis results
        
    Returns:
        HTML string with dynamic ranges
    """
    # Calculate rhythm metrics
    total_flat_duration = sum(end - start for start, end in flat_segments)
    longest_flat = max([end - start for start, end in flat_segments] or [0])
    rhythm_category = categorize_rhythmic_structure(flat_segments)
    pause_category = categorize_pauses(pause_count)
        
    def create_ranges_table(ranges_dict: Dict, feature_type: str) -> str:
        """Generate HTML table rows for value ranges with interpretations
        
        Args:
            ranges_dict: Analysis dictionary containing 'ranges'
            feature_type: Type of feature ('shimmer', 'energy', 'f0', 'f3')
            
        Returns:
            HTML string with table rows
        """
        # Interpretation guides for each feature type
        interpretations = {
            'shimmer': {
                'Stable': 'Normal vocal fold vibration with minimal instability',
                'Almost Stable': 'Mild vocal instability, typically not noticeable',
                'Almost Unstable': 'Moderate instability that may affect voice quality',
                'Unstable': 'Severe vocal instability, often clinically significant'
            },
            'energy': {
                'Very Low': 'Reduced vocal intensity, may indicate weak phonation',
                'Low': 'Slightly below average vocal power',
                'Moderate': 'Normal speaking intensity range',
                'High': 'Strong vocal projection'
            },
            'f0': {
                'Very Flat': 'Extremely limited pitch variation (monotone)',
                'Slightly Flat': 'Reduced but perceptible pitch changes',
                'Natural': 'Normal pitch variation for speech',
                'Dynamic': 'Exaggerated pitch changes'
            },
            'f3': {
                'Very Limited Coordination': 'Very limited tongue–lip movement',
                'Limited Coordination': 'Below-average coordination',
                'Normal Coordination': 'Healthy tongue–lip timing and placement',
                'High Coordination': 'Highly dynamic and well-controlled articulation'
            }
        }
        
        rows = []
        for category_name, (start, end) in ranges_dict['ranges'].items():
            # Extract just the descriptive part after "Q1: ", "Q2: " etc.
            display_name = category_name.split(': ')[1] if ': ' in category_name else category_name
            interpretation = interpretations[feature_type].get(display_name, 'No interpretation available')
            
            rows.append(f"""
                <tr>
                    <td>{start:.2f} to {end:.2f}</td>
                    <td>{display_name}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
    def create_pause_table() -> str:
        """Generate HTML table for pause analysis with interpretations"""
        pause_interpretations = {
            'None': 'Normal speech flow without interruptions',
            'Single': 'Minimal pausing, likely natural hesitation',
            'Few': 'Moderate pausing that may affect fluency',
            'Several': 'Excessive pausing, potentially clinically significant'
        }
        
        rows = []
        for count, category in [('0 pauses', 'None'),
                              ('1 pause', 'Single'),
                              ('2-3 pauses', 'Few'),
                              ('>3 pauses', 'Several')]:
            interpretation = pause_interpretations.get(category, '')
            rows.append(f"""
                <tr>
                    <td>{count}</td>
                    <td>{category}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
    def create_rhythm_table() -> str:
        """Generate HTML table for rhythm analysis with interpretations"""
        rhythm_interpretations = {
            'Rhythmic': 'Normal speech rhythm with good prosody',
            'Relatively Rhythmic': 'Mild rhythm deviations, mostly natural',
            'Less Rhythmic': 'Noticeable rhythm disturbances',
            'Non-Rhythmic': 'Severe rhythm impairment, potentially pathological'
        }
        
        rows = []
        for criteria, category in [('No flat segments', 'Rhythmic'),
                                 ('1 segment (5-6s)', 'Relatively Rhythmic'),
                                 ('2 segments or 1 >6s', 'Less Rhythmic'),
                                 ('>2 segments or >10s', 'Non-Rhythmic')]:
            interpretation = rhythm_interpretations.get(category, '')
            rows.append(f"""
                <tr>
                    <td>{criteria}</td>
                    <td>{category}</td>
                    <td>{interpretation}</td>
                </tr>
            """)
        return '\n'.join(rows)
    
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
                    text-align: center;
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
                <!-- Pause Analysis Section -->
                <div class="feature-section">
                    <div class="feature-title">Pause Analysis</div>
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
                            <th>Interpretation</th>
                        </tr>
                        {create_pause_table()}
                    </table>
                </div>
                
                <!-- Energy Analysis Section-->
                <div class="feature-section">
                    <div class="feature-title">Energy of Frequency Domain Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{energy_analysis['value']:.2f} dB</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{energy_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Energy of Frequency Domain Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(energy_analysis,'energy')}
                    </table>
                </div>
                <!-- Rhythmic Structure Section-->
                <div class="feature-section">
                    <div class="feature-title">Rhythmic Structure Analysis</div>
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
                            <th>Interpretation</th>
                        </tr>
                        {create_rhythm_table()}
                    </table>
                </div>
                <!-- Shimmer Analysis Section-->
                <div class="feature-section">
                    <div class="feature-title">Shimmer Standard Deviation Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{shimmer_analysis['value']:.2f}%</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{shimmer_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Shimmer Standard Deviation Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(shimmer_analysis,'shimmer')}
                    </table>
                </div>
                
                <!-- Fundamental Frequency Analysis-->
                 <div class="feature-section">
                    <div class="feature-title">Fundamental Frequency Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{f0_analysis['value']:.2f} Hz</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{f0_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Frequency Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(f0_analysis,'f0')}
                    </table>
                </div>

                <!-- Formant Frequency Analysis -->

                <div class="feature-section">
                    <div class="feature-title">Formant Frequency Analysis</div>
                    <div class="feature-grid">
                        <div class="feature-value">
                            <span>Measured Value:</span>
                            <span><strong>{f3_analysis['value']:.2f} Hz</strong></span>
                        </div>
                        <div class="feature-value">
                            <span>Category:</span>
                            <span><strong>{f3_analysis['category'].split(': ')[1]}</strong></span>
                        </div>
                    </div>
                    <table class="reference-table">
                        <tr>
                            <th>Frequency Range</th>
                            <th>Category</th>
                            <th>Interpretation</th>
                        </tr>
                        {create_ranges_table(f3_analysis,'f3')}
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




