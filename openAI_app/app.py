#!/usr/bin/env python3
"""
Flask Web Application for OpenAI Political Bias Analysis
Simple web interface for analyzing political bias in YouTube transcripts using OpenAI
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import sys
from datetime import datetime
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai_bias_analyzer import OpenAIBiasAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'openai-bias-analyzer-secret-key'

def get_openai_key():
    """Prompt user for OpenAI API key if not set"""
    if 'OPENAI_API_KEY' in os.environ:
        return os.environ['OPENAI_API_KEY']
    
    print("OpenAI Political Bias Analyzer")
    print("=" * 40)
    print("❌ OPENAI_API_KEY not found in environment")
    print("\nTo use this analyzer, you need an OpenAI API key.")
    print("Get one at: https://platform.openai.com/api-keys")
    print("\nEnter your OpenAI API key (starts with 'sk-'):")
    
    while True:
        api_key = input("API Key: ").strip()
        if api_key.startswith('sk-'):
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ API key set successfully!")
            return api_key
        else:
            print("❌ Invalid API key format. Should start with 'sk-'")
            print("Please try again:")

# Initialize analyzer
analyzer = None
try:
    # Try to get API key if not set
    if 'OPENAI_API_KEY' not in os.environ:
        get_openai_key()
    
    analyzer = OpenAIBiasAnalyzer()
    print("✅ OpenAI Bias Analyzer initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI analyzer: {e}")
    print("Please check your API key and try again.")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube_video():
    """Analyze political bias in a YouTube video"""
    try:
        data = request.get_json()
        video_url = data.get('video_url', '')
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        if not analyzer:
            return jsonify({'error': 'OpenAI Analyzer not available. Please restart the app and provide a valid API key.'}), 400
        
        # Analyze the YouTube video
        results = analyzer.analyze_youtube_video(video_url)
        
        if not results.get('success', False):
            return jsonify({'error': results.get('error', 'Analysis failed')}), 400
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze political bias in custom text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        title = data.get('title', 'Custom Text')
        
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Text must be at least 50 characters long'}), 400
        
        if not analyzer:
            return jsonify({'error': 'OpenAI Analyzer not available. Please restart the app and provide a valid API key.'}), 400
        
        # Analyze the text
        results = analyzer.analyze_transcript(text, title)
        
        if not results.get('success', False):
            return jsonify({'error': results.get('error', 'Analysis failed')}), 400
        
        return jsonify({
            'success': True,
            'results': results
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
        filename = f'openai_bias_analysis_{timestamp}.json'
        
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
        'openai_analyzer_available': analyzer is not None,
        'api_key_set': analyzer is not None and analyzer.api_key is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test():
    """Test endpoint with sample text"""
    try:
        if not analyzer:
            return jsonify({'error': 'Analyzer not available. Please restart the app and provide a valid API key.'}), 400
        
        # Sample text for testing
        sample_text = """
        The government should reduce regulations and allow free market principles to drive economic growth. 
        Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
        We need to strengthen our borders and enforce immigration laws. 
        The rule of law is fundamental to our democracy and national security. 
        Traditional family values are the foundation of a strong society. 
        We must protect religious freedom and support pro-life policies.
        """
        
        # Perform analysis
        results = analyzer.analyze_transcript(sample_text, "Test Political Video")
        
        if not results.get('success', False):
            return jsonify({'error': results.get('error', 'Test failed')}), 400
        
        return jsonify({
            'success': True,
            'test_results': {
                'transcript_length': results['transcript_length'],
                'analysis': results['analysis']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting OpenAI Political Bias Analysis Web App...")
    print("Access the application at: http://localhost:5002")
    print("Health check: http://localhost:5002/health")
    print("Test endpoint: http://localhost:5002/test")
    app.run(debug=True, host='0.0.0.0', port=5002)
