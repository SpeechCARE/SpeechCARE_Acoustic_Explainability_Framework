def generate_audio_report_html(audio_path: str, 
                             pause_count: int, 
                             shimmer_std: float, 
                             shimmer_category: str,
                             freq_energy: float,
                             freq_category: str,
                             quartile_ranges: dict) -> str:
    """
    Generates a professional audio analysis report as HTML.
    
    Args:
        audio_path: Path/name of audio file
        pause_count: Number of detected pauses
        shimmer_std: Calculated shimmer standard deviation (dB)
        shimmer_category: Quartile category for shimmer
        freq_energy: Frequency domain energy value
        freq_category: Quartile category for frequency energy
        quartile_ranges: Dictionary containing quartile ranges for both features
    
    """
    mapping_shimmer = {'(0) Very Low': 'Stable','(1) Low':'Almost Stable', '(2) Moderate':'Almost Unstable', '(3) High':'Unstable' }
    mapping_energy = {'(0) Very Low': 'Balanced','(1) Low':'Modulated', '(2) Moderate':'Irregular', '(3) High':'Erratic' }    

    shimmer_category = mapping_shimmer[shimmer_category]
    freq_category = mapping_energy[freq_category]

    # Helper functions (unchanged from original)
    def get_pause_category(count):
        if count == 0: return "None"
        elif count == 1: return "Single"
        elif 2 <= count <= 3: return "Few"
        else: return "Several"

    def get_pause_interpretation(count):
        interpretations = {
            0: "typical fluent speech",
            1: "slightly disrupted speech",
            2: "moderate pause frequency",
            3: "high pause frequency suggestive of cognitive load or speech disorder"
        }
        return interpretations.get(min(count, 3), "abnormal pause pattern")

    def get_shimmer_interpretation(cat):
        return {
            'Stable': "normal vocal fold vibration",
            'Almost Stable': "mild vocal instability",
            'Almost Unstable': "moderate voice abnormality",
            'Unstable': "severe vocal pathology likely"
        }.get(cat, "undefined stability pattern")

    def get_energy_interpretation(cat):
        return {
            'Balanced': "normal spectral distribution",
            'Modulated': "slight energy variations",
            'Irregular': "noticeable spectral irregularities",
            'Erratic': "abnormal energy distribution"
        }.get(cat, "atypical spectral characteristics")
    
    html_content = f"""
    <html>
    <head>
        <style>
            :root {{
                color-scheme: light dark;
            }}
            
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                
                /* Light mode defaults */
                --bg-color: #ffffff;
                --text-color: #333333;
                --primary-color: #2980b9;
                --secondary-color: #2c3e50;
                --table-header-bg: #3498db;
                --table-header-text: #ffffff;
                --table-row-even: #f2f2f2;
                --table-row-odd: #ffffff;
                --table-border: #dddddd;
                --interpretation-bg: #f9f9f9;
                --interpretation-border: #3498db;
            }}
            
            @media (prefers-color-scheme: dark) {{
                body {{
                    --bg-color: #1e1e1e;
                    --text-color: #e0e0e0;
                    --primary-color: #5dade2;
                    --secondary-color: #7f8c8d;
                    --table-header-bg: #2874a6;
                    --table-header-text: #ffffff;
                    --table-row-even: #2d2d2d;
                    --table-row-odd: #252525;
                    --table-border: #444444;
                    --interpretation-bg: #2a2a2a;
                    --interpretation-border: #5dade2;
                }}
            }}
            
            body {{
                background-color: var(--bg-color);
                color: var(--text-color);
            }}
            
            h1 {{
                color: var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 10px;
            }}
            
            h2 {{
                color: var(--primary-color);
                margin-top: 25px;
            }}
            
            h3 {{
                color: var(--secondary-color);
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.2);
            }}
            
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid var(--table-border);
            }}
            
            th {{
                background-color: var(--table-header-bg);
                color: var(--table-header-text);
            }}
            
            tr:nth-child(even) {{
                background-color: var(--table-row-even);
            }}
            
            tr:nth-child(odd) {{
                background-color: var(--table-row-odd);
            }}
            
            .interpretation {{
                background-color: var(--interpretation-bg);
                padding: 15px;
                border-left: 4px solid var(--interpretation-border);
                margin: 20px 0;
                border-radius: 4px;
            }}
            
            .emoji {{
                font-size: 1.2em;
                margin-right: 5px;
            }}
            
            strong {{
                color: var(--primary-color);
            }}
        </style>
    </head>
    <body>
        <h1><span class="emoji">🎵</span> Audio Analysis Report: {audio_path.split('/')[-1]}</h1>
        
        <h2><span class="emoji">🔊</span> Vocal Feature Summary</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Category</th>
            </tr>
            <tr>
                <td><strong>Pause Count</strong></td>
                <td>{pause_count}</td>
                <td>{get_pause_category(pause_count)}</td>
            </tr>
            <tr>
                <td><strong>Shimmer Stability</strong></td>
                <td>{shimmer_std:.2f} dB</td>
                <td>{shimmer_category}</td>
            </tr>
            <tr>
                <td><strong>Spectral Energy</strong></td>
                <td>{freq_energy:.2f}</td>
                <td>{freq_category}</td>
            </tr>
        </table>
        
        <h2><span class="emoji">📊</span> Quartile Reference Ranges</h2>
        
        <h3>Shimmer Stability Categories:</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Shimmer Range (dB)</th>
            </tr>
            <tr>
                <td>Stable (Q1)</td>
                <td>{quartile_ranges['shimmer'][0]}</td>
            </tr>
            <tr>
                <td>Almost Stable (Q2)</td>
                <td>{quartile_ranges['shimmer'][1]}</td>
            </tr>
            <tr>
                <td>Almost Unstable (Q3)</td>
                <td>{quartile_ranges['shimmer'][2]}</td>
            </tr>
            <tr>
                <td>Unstable (Q4)</td>
                <td>{quartile_ranges['shimmer'][3]}</td>
            </tr>
        </table>
        
        <h3>Spectral Energy Categories:</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Energy Profile</th>
            </tr>
            <tr>
                <td>Balanced (Q1)</td>
                <td>{quartile_ranges['freq_energy'][0]}</td>
            </tr>
            <tr>
                <td>Modulated (Q2)</td>
                <td>{quartile_ranges['freq_energy'][1]}</td>
            </tr>
            <tr>
                <td>Irregular (Q3)</td>
                <td>{quartile_ranges['freq_energy'][2]}</td>
            </tr>
            <tr>
                <td>Erratic (Q4)</td>
                <td>{quartile_ranges['freq_energy'][3]}</td>
            </tr>
        </table>
        
        <h2><span class="emoji">📝</span> Interpretation Notes</h2>
        <div class="interpretation">
            <p><strong>Pause patterns suggest:</strong> {get_pause_interpretation(pause_count)}</p>
            <p><strong>Shimmer stability indicates:</strong> {get_shimmer_interpretation(shimmer_category)}</p>
            <p><strong>Energy distribution shows:</strong> {get_energy_interpretation(freq_category)}</p>
        </div>
    </body>
    </html>
    """
    
    return html_content