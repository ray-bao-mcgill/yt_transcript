#!/usr/bin/env python3
"""
Flask Web Application for Political Bias Analysis
Provides a user-friendly interface for analyzing YouTube transcripts
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import sys
from datetime import datetime
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from political_bias_analyzer import PoliticalBiasAnalyzer, analyze_youtube_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize analyzer
analyzer = None
try:
    analyzer = PoliticalBiasAnalyzer()
    print("Political Bias Analyzer initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize analyzer: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze YouTube video for political bias"""
    try:
        data = request.get_json()
        video_url = data.get('video_url', '').strip()
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        # Analyze the video
        results = analyze_youtube_video(video_url)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        # Generate report
        report = analyzer.generate_report(results) if analyzer else "Analysis completed"
        
        # Get transcript text if available
        transcript_text = None
        if 'transcript_text' in results:
            transcript_text = results['transcript_text']
        elif 'transcript' in results:
            transcript_text = results['transcript']
        
        # Print transcript to console
        if transcript_text:
            print("\n" + "="*80)
            print("YOUTUBE TRANSCRIPT")
            print("="*80)
            print(transcript_text)
            print("="*80)
            print(f"Transcript length: {len(transcript_text)} characters")
            print("="*80 + "\n")
        else:
            print("No transcript available for this video.")
        
        return jsonify({
            'success': True,
            'results': results,
            'report': report,
            'transcript': transcript_text
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze custom text for political bias"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 50:
            return jsonify({'error': 'Text too short for meaningful analysis'}), 400
        
        # Analyze the text
        results = analyzer.analyze_transcript(text) if analyzer else {'error': 'Analyzer not available'}
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        # Generate report
        report = analyzer.generate_report(results) if analyzer else "Analysis completed"
        
        return jsonify({
            'success': True,
            'results': results,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download analysis results as JSON file"""
    try:
        data = request.get_json()
        results = data.get('results', {})
        
        if not results:
            return jsonify({'error': 'No results to download'}), 400
        
        # Create temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bias_analysis_{timestamp}.json'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f, indent=2, default=str)
            temp_path = f.name
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'analyzer_available': analyzer is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Political Bias Analysis Web App...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
