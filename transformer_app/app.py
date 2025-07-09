#!/usr/bin/env python3
"""
Flask Web Application for BERT Word Embedding Clustering Analysis
Provides a user-friendly interface for clustering word embeddings within YouTube transcripts
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import sys
from datetime import datetime
import tempfile
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bert_word_clustering_analyzer import BERTWordClusteringAnalyzer
from youtube_transcript import get_transcript

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bert-word-clustering-analyzer-secret-key'

# Initialize analyzer
analyzer = None
try:
    analyzer = BERTWordClusteringAnalyzer()
    print("BERT Word Clustering Analyzer initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize BERT analyzer: {e}")

def convert_ndarrays_to_lists(obj):
    """Recursively convert numpy arrays and numpy scalar types to lists/native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(v) for v in obj]
    else:
        return obj

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube_transcript():
    """Analyze word embeddings in a single YouTube transcript"""
    try:
        data = request.get_json()
        video_url = data.get('video_url', '')
        method = data.get('method', 'kmeans')
        n_clusters = data.get('n_clusters', 5)
        min_word_length = data.get('min_word_length', 3)
        
        # Additional clustering parameters
        clustering_params = {}
        if method == 'dbscan':
            clustering_params['eps'] = data.get('eps', 0.5)
            clustering_params['min_samples'] = data.get('min_samples', 2)
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
        
        if not analyzer:
            return jsonify({'error': 'BERT Analyzer not available'}), 400
        
        # Extract transcript
        try:
            transcript_result = get_transcript(video_url)
            if not transcript_result.get('success', False):
                error_msg = transcript_result.get('error', 'Could not extract transcript from video')
                return jsonify({'error': error_msg}), 400
            
            transcript = transcript_result['transcript_text']
            video_title = transcript_result.get('title', 'Unknown Video')
            
        except Exception as e:
            return jsonify({'error': f'Error extracting transcript: {str(e)}'}), 400
        
        # Perform word clustering analysis
        results = analyzer.analyze_single_transcript(
            transcript, 
            method=method, 
            n_clusters=n_clusters,
            **clustering_params
        )
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        # Add video information to results
        results['video_info'] = {
            'url': video_url,
            'title': video_title,
            'transcript_length': len(transcript)
        }
        # Convert numpy arrays to lists for JSON serialization
        results = convert_ndarrays_to_lists(results)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text_transcript():
    """Analyze word embeddings in custom text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        method = data.get('method', 'kmeans')
        n_clusters = data.get('n_clusters', 5)
        min_word_length = data.get('min_word_length', 3)
        
        # Additional clustering parameters
        clustering_params = {}
        if method == 'dbscan':
            clustering_params['eps'] = data.get('eps', 0.5)
            clustering_params['min_samples'] = data.get('min_samples', 2)
        
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Text must be at least 50 characters long'}), 400
        
        if not analyzer:
            return jsonify({'error': 'BERT Analyzer not available'}), 400
        
        # Perform word clustering analysis
        results = analyzer.analyze_single_transcript(
            text, 
            method=method, 
            n_clusters=n_clusters,
            **clustering_params
        )
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        # Convert numpy arrays to lists for JSON serialization
        results = convert_ndarrays_to_lists(results)
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download word clustering results as JSON file"""
    try:
        data = request.get_json()
        results = data.get('results', {})
        
        if not results:
            return jsonify({'error': 'No results to download'}), 400
        
        # Create temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bert_word_clustering_analysis_{timestamp}.json'
        
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
        'bert_analyzer_available': analyzer is not None,
        'bert_model_loaded': analyzer and analyzer.model is not None,
        'visualization_available': analyzer and hasattr(analyzer, 'create_word_visualizations'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test():
    """Test endpoint with sample text"""
    try:
        if not analyzer:
            return jsonify({'error': 'Analyzer not available'}), 400
        
        # Sample text for testing
        sample_text = """
        The government should reduce regulations and allow free market principles to drive economic growth. 
        Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
        We need to strengthen our borders and enforce immigration laws. 
        The rule of law is fundamental to our democracy and national security. 
        Traditional family values are the foundation of a strong society. 
        We must protect religious freedom and support pro-life policies.
        """
        
        # Perform word clustering analysis
        results = analyzer.analyze_single_transcript(sample_text, 'kmeans', 5)
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        return jsonify({
            'success': True,
            'test_results': {
                'transcript_length': len(sample_text),
                'embedded_words': results['embedding_results']['embedded_words'],
                'n_clusters': results['cluster_results']['n_clusters'],
                'silhouette_score': results['silhouette_score'],
                'cluster_analysis': results['cluster_analysis']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting BERT Word Embedding Clustering Analysis Web App...")
    print("Access the application at: http://localhost:5001")
    print("Health check: http://localhost:5001/health")
    print("Test endpoint: http://localhost:5001/test")
    app.run(debug=True, host='0.0.0.0', port=5001)
