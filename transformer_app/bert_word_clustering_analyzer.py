#!/usr/bin/env python3
"""
BERT Word Embedding Clustering Analyzer for YouTube Transcripts
Clusters word embeddings within a single transcript to identify semantic groups and bias patterns
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
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("Warning: Clustering libraries not available. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available. Install with: pip install matplotlib seaborn plotly")

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_transcript import get_transcript


class BERTWordClusteringAnalyzer:
    """
    BERT-based word embedding clustering analyzer for single transcript analysis
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the BERT word clustering analyzer"""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Analysis results storage
        self.word_embeddings = {}
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
    
    def extract_word_embeddings(self, transcript: str, min_word_length: int = 3) -> Dict:
        """
        Extract BERT embeddings for individual words in a transcript
        
        Args:
            transcript: The transcript text
            min_word_length: Minimum word length to include
            
        Returns:
            Dictionary with word embeddings and analysis
        """
        if not self.model:
            raise ValueError("BERT model not available")
        
        try:
            print("Extracting word embeddings from transcript...")
            
            # Clean and tokenize transcript
            cleaned_text = self._clean_text(transcript)
            words = self._extract_significant_words(cleaned_text, min_word_length)
            
            if len(words) < 5:
                raise ValueError("Not enough significant words found in transcript")
            
            print(f"Found {len(words)} significant words for embedding extraction")
            
            # Extract embeddings for each word
            word_embeddings = {}
            word_contexts = {}
            
            for word in words:
                try:
                    # Create context for the word (surrounding words)
                    context = self._get_word_context(cleaned_text, word)
                    
                    # Generate embedding for the word in context
                    embedding = self.model.encode(
                        context,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.device
                    )
                    
                    word_embeddings[word] = embedding.cpu().numpy()
                    word_contexts[word] = context
                    
                except Exception as e:
                    print(f"Error extracting embedding for word '{word}': {e}")
                    continue
            
            if len(word_embeddings) < 3:
                raise ValueError("Not enough word embeddings extracted")
            
            print(f"Successfully extracted embeddings for {len(word_embeddings)} words")
            
            return {
                'word_embeddings': word_embeddings,
                'word_contexts': word_contexts,
                'total_words': len(words),
                'embedded_words': len(word_embeddings)
            }
            
        except Exception as e:
            print(f"Error extracting word embeddings: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        return text.lower()
    
    def _extract_significant_words(self, text: str, min_length: int = 3) -> List[str]:
        """Extract significant words from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter by length and stop words
        significant_words = []
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if (len(word) >= min_length and 
                word not in stop_words and 
                count >= 1):  # Word appears at least once
                significant_words.append(word)
        
        return significant_words
    
    def _get_word_context(self, text: str, word: str, context_size: int = 5) -> str:
        """Get context around a word for better embedding"""
        words = text.split()
        
        try:
            # Find the word in the text
            word_indices = [i for i, w in enumerate(words) if w == word]
            
            if not word_indices:
                return word
            
            # Use the first occurrence
            idx = word_indices[0]
            
            # Get surrounding context
            start = max(0, idx - context_size)
            end = min(len(words), idx + context_size + 1)
            
            context_words = words[start:end]
            return ' '.join(context_words)
            
        except Exception:
            return word
    
    def cluster_word_embeddings(self, word_embeddings: Dict, method: str = 'kmeans', 
                               n_clusters: int = 5, **kwargs) -> Dict:
        """
        Cluster word embeddings using various algorithms
        
        Args:
            word_embeddings: Dictionary of word -> embedding arrays
            method: clustering method ('kmeans', 'dbscan', 'agglomerative')
            n_clusters: number of clusters (for kmeans/agglomerative)
            **kwargs: additional parameters for clustering
            
        Returns:
            Dictionary with clustering results
        """
        if not CLUSTERING_AVAILABLE:
            raise ValueError("Clustering libraries not available")
        
        try:
            print(f"Clustering {len(word_embeddings)} word embeddings using {method}...")
            
            # Convert embeddings to array
            words = list(word_embeddings.keys())
            embeddings = np.array([word_embeddings[word] for word in words])
            
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
            
            # Create word-to-cluster mapping
            word_clusters = {}
            for i, word in enumerate(words):
                word_clusters[word] = {
                    'cluster': int(cluster_labels[i]),
                    'embedding_2d': embeddings_2d[i].tolist()
                }
            
            return {
                'method': method,
                'n_clusters': len(set(cluster_labels)),
                'cluster_labels': cluster_labels,
                'embeddings_2d': embeddings_2d,
                'silhouette_score': silhouette,
                'word_clusters': word_clusters,
                'words': words
                # 'clusterer': clusterer  # Removed for JSON serialization
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
    
    def analyze_word_clusters(self, word_clusters: Dict, word_contexts: Dict) -> Dict:
        """
        Analyze the characteristics of each word cluster
        
        Args:
            word_clusters: Results from cluster_word_embeddings
            word_contexts: Dictionary of word -> context
            
        Returns:
            Dictionary with cluster analysis
        """
        cluster_analysis = {}
        word_to_cluster = word_clusters['word_clusters']
        unique_labels = set(word_clusters['cluster_labels'])
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
            
            # Get words in this cluster
            cluster_words = [word for word, data in word_to_cluster.items() if data['cluster'] == label]
            
            # Analyze cluster characteristics
            cluster_info = self._analyze_word_cluster(cluster_words, word_contexts, label)
            cluster_analysis[f'cluster_{label}'] = cluster_info
        
        return cluster_analysis
    
    def _analyze_word_cluster(self, words: List[str], word_contexts: Dict, cluster_id: int) -> Dict:
        """Analyze characteristics of words in a cluster"""
        if not words:
            return {}
        
        # Basic statistics
        avg_length = sum(len(word) for word in words) / len(words)
        
        # Political content analysis
        political_terms = self._extract_political_terms(words)
        bias_indicators = self._analyze_bias_indicators(words)
        
        # Semantic analysis
        semantic_groups = self._group_semantically_similar_words(words)
        
        # Context analysis
        contexts = [word_contexts.get(word, word) for word in words]
        
        return {
            'size': len(words),
            'words': words,
            'avg_length': avg_length,
            'political_terms': political_terms,
            'bias_indicators': bias_indicators,
            'semantic_groups': semantic_groups,
            'sample_contexts': contexts[:5]  # First 5 contexts as examples
        }
    
    def _extract_political_terms(self, words: List[str]) -> List[str]:
        """Extract political terms from word list"""
        political_keywords = [
            'government', 'politics', 'political', 'election', 'campaign',
            'president', 'congress', 'senate', 'policy', 'legislation',
            'democrat', 'republican', 'liberal', 'conservative',
            'vote', 'voting', 'democracy', 'republic', 'administration',
            'law', 'rights', 'freedom', 'justice', 'equality', 'reform'
        ]
        
        return [word for word in words if word in political_keywords]
    
    def _analyze_bias_indicators(self, words: List[str]) -> Dict:
        """Analyze bias indicators in words"""
        bias_indicators = {
            'loaded_language': 0,
            'emotional_words': 0,
            'subjective_terms': 0
        }
        
        loaded_words = ['radical', 'extreme', 'dangerous', 'corrupt', 'evil', 'amazing', 'terrible', 'horrible', 'wonderful']
        emotional_words = ['love', 'hate', 'fear', 'anger', 'joy', 'sadness', 'hope', 'despair']
        subjective_terms = ['obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely', 'absolutely']
        
        for word in words:
            if word in loaded_words:
                bias_indicators['loaded_language'] += 1
            if word in emotional_words:
                bias_indicators['emotional_words'] += 1
            if word in subjective_terms:
                bias_indicators['subjective_terms'] += 1
        
        return bias_indicators
    
    def _group_semantically_similar_words(self, words: List[str]) -> Dict:
        """Group words by semantic similarity"""
        # Simple semantic grouping based on word patterns
        groups = {
            'political': [],
            'economic': [],
            'social': [],
            'technical': [],
            'emotional': [],
            'other': []
        }
        
        political_terms = ['government', 'politics', 'election', 'vote', 'policy', 'law']
        economic_terms = ['economy', 'money', 'business', 'market', 'trade', 'tax']
        social_terms = ['people', 'society', 'community', 'family', 'education', 'health']
        technical_terms = ['technology', 'science', 'research', 'data', 'system', 'process']
        emotional_terms = ['love', 'hate', 'fear', 'hope', 'anger', 'joy']
        
        for word in words:
            if word in political_terms:
                groups['political'].append(word)
            elif word in economic_terms:
                groups['economic'].append(word)
            elif word in social_terms:
                groups['social'].append(word)
            elif word in technical_terms:
                groups['technical'].append(word)
            elif word in emotional_terms:
                groups['emotional'].append(word)
            else:
                groups['other'].append(word)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def create_word_visualizations(self, cluster_results: Dict, cluster_analysis: Dict) -> Dict:
        """
        Create visualizations of word clustering results
        
        Args:
            cluster_results: Results from cluster_word_embeddings
            cluster_analysis: Results from analyze_word_clusters
            
        Returns:
            Dictionary with visualization HTML strings
        """
        if not VISUALIZATION_AVAILABLE:
            return {"error": "Visualization libraries not available"}
        
        try:
            visualizations = {}
            
            # 1. Scatter plot of word clusters
            visualizations['word_scatter'] = self._create_word_scatter_plot(cluster_results)
            
            # 2. Cluster summary
            visualizations['cluster_summary'] = self._create_word_cluster_summary(cluster_analysis)
            
            # 3. Word frequency analysis
            visualizations['word_frequency'] = self._create_word_frequency_analysis(cluster_analysis)
            
            # 4. Semantic group analysis
            visualizations['semantic_analysis'] = self._create_semantic_analysis(cluster_analysis)
            
            return visualizations
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return {"error": f"Visualization creation failed: {str(e)}"}
    
    def _create_word_scatter_plot(self, cluster_results: Dict) -> str:
        """Create scatter plot of word clusters"""
        embeddings_2d = cluster_results['embeddings_2d']
        labels = cluster_results['cluster_labels']
        words = cluster_results['words']
        
        fig = go.Figure()
        
        unique_labels = set(labels)
        colors = px.colors.qualitative.Set3
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers+text',
                    text=[words[j] for j in range(len(words)) if labels[j] == label],
                    textposition="top center",
                    marker=dict(color='gray', size=8, opacity=0.6),
                    name='Noise Words',
                    hovertemplate='<b>%{text}</b><br>Cluster: Noise<extra></extra>'
                ))
            else:
                mask = labels == label
                cluster_words = [words[j] for j in range(len(words)) if labels[j] == label]
                
                fig.add_trace(go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode='markers+text',
                    text=cluster_words,
                    textposition="top center",
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=10,
                        opacity=0.8
                    ),
                    name=f'Cluster {label} ({len(cluster_words)} words)',
                    hovertemplate='<b>%{text}</b><br>Cluster: %{customdata}<extra></extra>',
                    customdata=[label] * len(cluster_words)
                ))
        
        fig.update_layout(
            title="Word Embedding Clusters",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template="plotly_white",
            height=700,
            showlegend=True
        )
        
        return fig.to_html(full_html=False)
    
    def _create_word_cluster_summary(self, cluster_analysis: Dict) -> str:
        """Create word cluster summary dashboard"""
        clusters = list(cluster_analysis.keys())
        sizes = [cluster_analysis[c]['size'] for c in clusters]
        avg_lengths = [cluster_analysis[c]['avg_length'] for c in clusters]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cluster Sizes", "Average Word Length", "Political Terms", "Bias Indicators"),
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
            title="Word Cluster Analysis Summary",
            height=800,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def _create_word_frequency_analysis(self, cluster_analysis: Dict) -> str:
        """Create word frequency analysis"""
        # Collect all words from all clusters
        all_words = []
        for cluster_name, cluster_data in cluster_analysis.items():
            all_words.extend(cluster_data['words'])
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(20)
        
        words, counts = zip(*top_words)
        
        fig = go.Figure(data=go.Bar(
            x=words,
            y=counts,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Most Frequent Words Across Clusters",
            xaxis_title="Words",
            yaxis_title="Frequency",
            height=500,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def _create_semantic_analysis(self, cluster_analysis: Dict) -> str:
        """Create semantic group analysis"""
        # Collect semantic groups from all clusters
        all_semantic_groups = {}
        for cluster_name, cluster_data in cluster_analysis.items():
            for group_name, words in cluster_data['semantic_groups'].items():
                if group_name not in all_semantic_groups:
                    all_semantic_groups[group_name] = []
                all_semantic_groups[group_name].extend(words)
        
        # Count words in each semantic group
        group_counts = {name: len(words) for name, words in all_semantic_groups.items()}
        
        fig = go.Figure(data=go.Pie(
            labels=list(group_counts.keys()),
            values=list(group_counts.values()),
            hole=0.3
        ))
        
        fig.update_layout(
            title="Semantic Group Distribution",
            height=500,
            template="plotly_white"
        )
        
        return fig.to_html(full_html=False)
    
    def analyze_single_transcript(self, transcript: str, method: str = 'kmeans', 
                                 n_clusters: int = 5, **kwargs) -> Dict:
        """
        Complete analysis pipeline for a single transcript
        
        Args:
            transcript: The transcript text
            method: Clustering method
            n_clusters: Number of clusters
            **kwargs: Additional clustering parameters
            
        Returns:
            Complete analysis results
        """
        try:
            print(f"Analyzing single transcript with {len(transcript)} characters...")
            
            # Extract word embeddings
            embedding_results = self.extract_word_embeddings(transcript)
            word_embeddings = embedding_results['word_embeddings']
            word_contexts = embedding_results['word_contexts']
            
            # Perform clustering
            cluster_results = self.cluster_word_embeddings(
                word_embeddings, method, n_clusters, **kwargs
            )
            
            # Analyze clusters
            cluster_analysis = self.analyze_word_clusters(cluster_results, word_contexts)
            
            # Create visualizations
            visualizations = self.create_word_visualizations(cluster_results, cluster_analysis)
            
            return {
                'success': True,
                'transcript_length': len(transcript),
                'embedding_results': embedding_results,
                'cluster_results': cluster_results,
                'cluster_analysis': cluster_analysis,
                'visualizations': visualizations,
                'silhouette_score': cluster_results.get('silhouette_score')
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


def main():
    """Test the BERT word clustering analyzer"""
    # Sample transcript for testing
    sample_transcript = """
    The government should reduce regulations and allow free market principles to drive economic growth. 
    Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
    We need to strengthen our borders and enforce immigration laws. 
    The rule of law is fundamental to our democracy and national security. 
    Traditional family values are the foundation of a strong society. 
    We must protect religious freedom and support pro-life policies.
    """
    
    # Initialize analyzer
    analyzer = BERTWordClusteringAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_single_transcript(sample_transcript, method='kmeans', n_clusters=5)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("Analysis completed successfully!")
        print(f"Found {results['cluster_results']['n_clusters']} word clusters")
        print(f"Silhouette score: {results['silhouette_score']}")


if __name__ == "__main__":
    main() 