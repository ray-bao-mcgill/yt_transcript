#!/usr/bin/env python3
"""
BERT Embedding Clustering Analyzer for YouTube Transcripts
Uses BERT embeddings to cluster transcripts and visualize political bias patterns
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# BERT and ML libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT libraries not available. Install with: pip install transformers torch sentence-transformers")

# Clustering and visualization
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available. Install with: pip install scikit-learn matplotlib seaborn plotly")

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_transcript import get_transcript


class BERTClusteringAnalyzer:
    """
    BERT-based clustering analyzer using embeddings for YouTube transcript analysis
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the BERT clustering analyzer"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clustering results storage
        self.embeddings_cache = {}
        self.clustering_results = {}
        
        # Initialize BERT model
        self._initialize_bert_model()
        
    def _initialize_bert_model(self):
        """Initialize BERT model for embedding generation"""
        if not BERT_AVAILABLE:
            print("BERT libraries not available. Cannot initialize model.")
            return
        
        try:
            print(f"Loading BERT model: {self.model_name}")
            
            # Use SentenceTransformer for better embedding quality
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            
            print("✓ BERT model loaded successfully")
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.model = None
    
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT embeddings from a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        if not self.model:
            raise ValueError("BERT model not available")
        
        try:
            # Clean and filter texts
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                cleaned_text = self._clean_text(text)
                if len(cleaned_text.strip()) > 10:  # Minimum length
                    valid_texts.append(cleaned_text)
                    valid_indices.append(i)
            
            if not valid_texts:
                raise ValueError("No valid texts found")
            
            print(f"Extracting embeddings for {len(valid_texts)} texts...")
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=self.device
            )
            
            # Convert to numpy and normalize
            embeddings = embeddings.cpu().numpy()
            
            # Store in cache
            for i, idx in enumerate(valid_indices):
                self.embeddings_cache[idx] = embeddings[i]
            
            return embeddings, valid_indices
            
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        return text
    
    def cluster_embeddings(self, embeddings: np.ndarray, method: str = 'kmeans', 
                          n_clusters: int = 3, **kwargs) -> Dict:
        """
        Cluster embeddings using various algorithms
        
        Args:
            embeddings: numpy array of embeddings
            method: clustering method ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: number of clusters (for kmeans/agglomerative)
            **kwargs: additional parameters for clustering
            
        Returns:
            Dictionary with clustering results
        """
        if not VISUALIZATION_AVAILABLE:
            raise ValueError("Clustering libraries not available")
        
        try:
            print(f"Clustering embeddings using {method}...")
            
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Perform clustering
            if method == 'kmeans':
                clusterer = KMeans(
                    n_clusters=min(n_clusters, len(embeddings)),
                    random_state=42,
                    **kwargs
                )
            elif method == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 2)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            elif method == 'agglomerative':
                clusterer = AgglomerativeClustering(
                    n_clusters=min(n_clusters, len(embeddings)),
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Fit and predict
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Calculate silhouette score if possible
            silhouette = None
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                try:
                    silhouette = silhouette_score(embeddings_scaled, cluster_labels)
                except:
                    pass
            
            # Reduce dimensionality for visualization
            embeddings_2d = self._reduce_dimensionality(embeddings_scaled)
            
            return {
                'method': method,
                'n_clusters': len(set(cluster_labels)),
                'cluster_labels': cluster_labels,
                'embeddings_2d': embeddings_2d,
                'silhouette_score': silhouette,
                'clusterer': clusterer
            }
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            raise
    
    def _reduce_dimensionality(self, embeddings: np.ndarray, method: str = 'pca') -> np.ndarray:
        """Reduce dimensionality for visualization"""
        if embeddings.shape[1] <= 2:
            return embeddings
        
        try:
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
            return reducer.fit_transform(embeddings)
            
        except Exception as e:
            print(f"Error in dimensionality reduction: {e}")
            # Fallback to PCA
            return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    
    def analyze_clusters(self, texts: List[str], cluster_results: Dict) -> Dict:
        """
        Analyze the characteristics of each cluster
        
        Args:
            texts: Original texts
            cluster_results: Results from cluster_embeddings
            
        Returns:
            Dictionary with cluster analysis
        """
        cluster_labels = cluster_results['cluster_labels']
        unique_labels = set(cluster_labels)
        
        cluster_analysis = {}
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get texts in this cluster
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == label]
            
            # Analyze cluster characteristics
            cluster_info = self._analyze_cluster_texts(cluster_texts, label)
            cluster_analysis[f'cluster_{label}'] = cluster_info
        
        return cluster_analysis
    
    def _analyze_cluster_texts(self, texts: List[str], cluster_id: int) -> Dict:
        """Analyze characteristics of texts in a cluster"""
        if not texts:
            return {}
        
        # Basic statistics
        total_length = sum(len(text) for text in texts)
        avg_length = total_length / len(texts)
        
        # Extract key terms and topics
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(10)
        
        # Political content analysis
        political_terms = self._extract_political_terms(texts)
        bias_indicators = self._analyze_bias_indicators(texts)
        
        return {
            'size': len(texts),
            'avg_length': avg_length,
            'total_length': total_length,
            'top_words': top_words,
            'political_terms': political_terms,
            'bias_indicators': bias_indicators,
            'sample_texts': texts[:3]  # First 3 texts as examples
        }
    
    def _extract_political_terms(self, texts: List[str]) -> List[str]:
        """Extract political terms from texts"""
        political_keywords = [
            'government', 'politics', 'political', 'election', 'campaign',
            'president', 'congress', 'senate', 'policy', 'legislation',
            'democrat', 'republican', 'liberal', 'conservative',
            'vote', 'voting', 'democracy', 'republic', 'administration'
        ]
        
        found_terms = []
        for text in texts:
            text_lower = text.lower()
            for term in political_keywords:
                if term in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))
    
    def _analyze_bias_indicators(self, texts: List[str]) -> Dict:
        """Analyze bias indicators in texts"""
        bias_indicators = {
            'loaded_language': 0,
            'subjective_markers': 0,
            'emotional_intensity': 0
        }
        
        loaded_words = ['radical', 'extreme', 'dangerous', 'corrupt', 'evil', 'amazing', 'terrible']
        subjective_markers = ['obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely']
        
        for text in texts:
            text_lower = text.lower()
            
            # Count loaded language
            for word in loaded_words:
                bias_indicators['loaded_language'] += text_lower.count(word)
            
            # Count subjective markers
            for marker in subjective_markers:
                bias_indicators['subjective_markers'] += text_lower.count(marker)
            
            # Simple emotional intensity (exclamation marks, caps)
            bias_indicators['emotional_intensity'] += text.count('!') + len(re.findall(r'\b[A-Z]{3,}\b', text))
        
        return bias_indicators
    
    def create_visualizations(self, cluster_results: Dict, cluster_analysis: Dict, 
                            texts: List[str]) -> Dict:
        """
        Create comprehensive visualizations of clustering results
        
        Args:
            cluster_results: Results from cluster_embeddings
            cluster_analysis: Results from analyze_clusters
            texts: Original texts
            
        Returns:
            Dictionary with visualization HTML strings
        """
        if not VISUALIZATION_AVAILABLE:
            return {"error": "Visualization libraries not available"}
        
        try:
            visualizations = {}
            
            # 1. Scatter plot of clusters
            visualizations['scatter_plot'] = self._create_scatter_plot(
                cluster_results, cluster_analysis
            )
            
            # 2. Cluster summary dashboard
            visualizations['cluster_summary'] = self._create_cluster_summary(
                cluster_analysis
            )
            
            # 3. Word frequency analysis
            visualizations['word_analysis'] = self._create_word_analysis(
                cluster_analysis
            )
            
            # 4. Bias indicators comparison
            visualizations['bias_comparison'] = self._create_bias_comparison(
                cluster_analysis
            )
            
            return visualizations
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return {"error": f"Visualization creation failed: {str(e)}"}
    
    def _create_scatter_plot(self, cluster_results: Dict, cluster_analysis: Dict) -> str:
        """Create scatter plot of clusters"""
        embeddings_2d = cluster_results['embeddings_2d']
        labels = cluster_results['cluster_labels']
        
        fig = go.Figure()
        
        unique_labels = set(labels)
        colors = px.colors.qualitative.Set3
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(color='gray', size=8, opacity=0.6),
                    name='Noise Points',
                    hovertemplate='<b>Noise Point</b><extra></extra>'
                ))
            else:
                mask = labels == label
                cluster_info = cluster_analysis.get(f'cluster_{label}', {})
                
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=10,
                        opacity=0.8
                    ),
                    name=f'Cluster {label} (Size: {cluster_info.get("size", 0)})',
                    hovertemplate='<b>Cluster %{customdata}</b><br>Size: %{text}<extra></extra>',
                    customdata=[label] * sum(mask),
                    text=[cluster_info.get("size", 0)] * sum(mask)
                ))
        
        fig.update_layout(
            title="BERT Embeddings Clustering Visualization",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template="plotly_white",
            height=600,
            showlegend=True
        )
        
        return fig.to_html(full_html=False)
    
    def _create_cluster_summary(self, cluster_analysis: Dict) -> str:
        """Create cluster summary dashboard"""
        clusters = list(cluster_analysis.keys())
        sizes = [cluster_analysis[c]['size'] for c in clusters]
        avg_lengths = [cluster_analysis[c]['avg_length'] for c in clusters]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cluster Sizes", "Average Text Length", "Political Terms", "Bias Indicators"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Cluster sizes
        fig.add_trace(
            go.Bar(x=clusters, y=sizes, name="Size", marker_color="lightblue"),
            row=1, col=1
        )
        
        # Average lengths
        fig.add_trace(
            go.Bar(x=clusters, y=avg_lengths, name="Avg Length", marker_color="lightcoral"),
            row=1, col=2
        )
        
        # Political terms count
        political_counts = [len(cluster_analysis[c]['political_terms']) for c in clusters]
        fig.add_trace(
            go.Bar(x=clusters, y=political_counts, name="Political Terms", marker_color="lightgreen"),
            row=2, col=1
        )
        
        # Bias indicators
        bias_scores = [sum(cluster_analysis[c]['bias_indicators'].values()) for c in clusters]
        fig.add_trace(
            go.Bar(x=clusters, y=bias_scores, name="Bias Score", marker_color="lightyellow"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Cluster Analysis Summary",
            height=800,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def _create_word_analysis(self, cluster_analysis: Dict) -> str:
        """Create word frequency analysis"""
        # Collect all words from all clusters
        all_words = {}
        for cluster_name, cluster_data in cluster_analysis.items():
            for word, count in cluster_data['top_words']:
                if word not in all_words:
                    all_words[word] = {}
                all_words[word][cluster_name] = count
        
        # Get top words across all clusters
        top_words = sorted(all_words.keys(), 
                          key=lambda w: sum(all_words[w].values()), 
                          reverse=True)[:15]
        
        # Create heatmap data
        clusters = list(cluster_analysis.keys())
        heatmap_data = []
        for word in top_words:
            row = [all_words[word].get(cluster, 0) for cluster in clusters]
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=clusters,
            y=top_words,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Word Frequency Across Clusters",
            xaxis_title="Clusters",
            yaxis_title="Words",
            height=600,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def _create_bias_comparison(self, cluster_analysis: Dict) -> str:
        """Create bias indicators comparison"""
        clusters = list(cluster_analysis.keys())
        
        # Extract bias indicators
        loaded_language = [cluster_analysis[c]['bias_indicators']['loaded_language'] for c in clusters]
        subjective_markers = [cluster_analysis[c]['bias_indicators']['subjective_markers'] for c in clusters]
        emotional_intensity = [cluster_analysis[c]['bias_indicators']['emotional_intensity'] for c in clusters]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Loaded Language',
            x=clusters,
            y=loaded_language,
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            name='Subjective Markers',
            x=clusters,
            y=subjective_markers,
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            name='Emotional Intensity',
            x=clusters,
            y=emotional_intensity,
            marker_color='purple'
        ))
        
        fig.update_layout(
            title="Bias Indicators Comparison Across Clusters",
            xaxis_title="Clusters",
            yaxis_title="Count",
            barmode='group',
            height=500,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def analyze_youtube_videos(self, video_urls: List[str], method: str = 'kmeans', 
                              n_clusters: int = 3, **kwargs) -> Dict:
        """
        Complete analysis pipeline for YouTube videos
        
        Args:
            video_urls: List of YouTube URLs
            method: Clustering method
            n_clusters: Number of clusters
            **kwargs: Additional clustering parameters
            
        Returns:
            Complete analysis results
        """
        try:
            print(f"Analyzing {len(video_urls)} YouTube videos...")
            
            # Extract transcripts
            transcripts = []
            video_info = []
            
            for i, url in enumerate(video_urls):
                try:
                    result = get_transcript(url)
                    if 'transcript' in result:
                        transcripts.append(result['transcript'])
                        video_info.append({
                            'url': url,
                            'title': result.get('title', f'Video {i+1}'),
                            'length': len(result['transcript'])
                        })
                        print(f"✓ Extracted transcript {i+1}/{len(video_urls)}")
                    else:
                        print(f"✗ Could not extract transcript for video {i+1}")
                except Exception as e:
                    print(f"✗ Error processing video {i+1}: {e}")
                    continue
            
            if len(transcripts) < 2:
                return {"error": "Need at least 2 valid transcripts for clustering"}
            
            # Extract embeddings
            embeddings, valid_indices = self.extract_embeddings(transcripts)
            
            # Filter video info to match valid transcripts
            valid_video_info = [video_info[i] for i in valid_indices]
            valid_transcripts = [transcripts[i] for i in valid_indices]
            
            # Perform clustering
            cluster_results = self.cluster_embeddings(
                embeddings, method, n_clusters, **kwargs
            )
            
            # Analyze clusters
            cluster_analysis = self.analyze_clusters(valid_transcripts, cluster_results)
            
            # Create visualizations
            visualizations = self.create_visualizations(
                cluster_results, cluster_analysis, valid_transcripts
            )
            
            return {
                'success': True,
                'video_info': valid_video_info,
                'cluster_results': cluster_results,
                'cluster_analysis': cluster_analysis,
                'visualizations': visualizations,
                'embeddings_shape': embeddings.shape,
                'silhouette_score': cluster_results.get('silhouette_score')
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


def main():
    """Test the BERT clustering analyzer"""
    # Sample YouTube URLs for testing
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with actual URLs
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ]
    
    # Initialize analyzer
    analyzer = BERTClusteringAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_youtube_videos(test_urls, method='kmeans', n_clusters=3)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("Analysis completed successfully!")
        print(f"Found {results['cluster_results']['n_clusters']} clusters")
        print(f"Silhouette score: {results['silhouette_score']}")


if __name__ == "__main__":
    main() 