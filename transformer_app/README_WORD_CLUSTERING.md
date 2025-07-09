# BERT Word Embedding Clustering Analyzer

A sophisticated tool for clustering word embeddings within YouTube transcripts to identify semantic groups and bias patterns using BERT (Bidirectional Encoder Representations from Transformers).

## Overview

This analyzer extracts BERT embeddings for individual words within a single transcript and clusters them to identify:
- **Semantic word groups** - Words that are semantically similar
- **Political bias patterns** - Political terms and bias indicators
- **Contextual relationships** - How words relate to each other in context
- **Topic clusters** - Groups of words representing similar topics or themes

## Features

### 🔍 Word-Level Analysis
- Extracts BERT embeddings for individual words in context
- Filters out stop words and insignificant terms
- Analyzes word frequency and significance

### 🎯 Multiple Clustering Methods
- **K-Means**: Traditional clustering with specified number of clusters
- **DBSCAN**: Density-based clustering for variable cluster sizes
- **Agglomerative**: Hierarchical clustering approach

### 📊 Comprehensive Analysis
- **Political content detection**: Identifies political terms and concepts
- **Bias indicator analysis**: Loaded language, emotional words, subjective terms
- **Semantic grouping**: Categorizes words by meaning and context
- **Cluster quality metrics**: Silhouette scores and cluster statistics

### 📈 Interactive Visualizations
- **Word embedding scatter plots**: Visual representation of word clusters
- **Cluster summary dashboards**: Statistical overview of clusters
- **Word frequency analysis**: Most frequent words across clusters
- **Semantic group distribution**: Pie charts of semantic categories

### 🌐 Web Interface
- **YouTube video analysis**: Extract and analyze transcripts from YouTube URLs
- **Custom text analysis**: Analyze any text input
- **Interactive controls**: Adjust clustering parameters
- **Results download**: Export analysis results as JSON

## Installation

1. **Clone the repository** (if not already done):
```bash
cd yt_transcript/transformer_app
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Web Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Access the web interface**:
   - Open your browser to `http://localhost:5001`
   - Use the tabs to switch between YouTube video and custom text analysis

3. **Analyze a YouTube video**:
   - Enter a YouTube URL
   - Choose clustering method and parameters
   - Click "Analyze YouTube Transcript"

4. **Analyze custom text**:
   - Enter or paste your text
   - Configure clustering settings
   - Click "Analyze Custom Text"

### Command Line

```python
from bert_word_clustering_analyzer import BERTWordClusteringAnalyzer

# Initialize analyzer
analyzer = BERTWordClusteringAnalyzer()

# Analyze text
results = analyzer.analyze_single_transcript(
    text="Your text here...",
    method='kmeans',
    n_clusters=5
)

# Access results
print(f"Found {results['cluster_results']['n_clusters']} word clusters")
print(f"Silhouette score: {results['silhouette_score']}")
```

### Testing

Run the test suite:
```bash
python test_word_clustering.py
```

## API Endpoints

### `/analyze_youtube` (POST)
Analyze word embeddings in a YouTube transcript.

**Parameters:**
- `video_url`: YouTube video URL
- `method`: Clustering method ('kmeans', 'dbscan', 'agglomerative')
- `n_clusters`: Number of clusters (for kmeans/agglomerative)
- `min_word_length`: Minimum word length to include
- `eps`: Epsilon parameter for DBSCAN
- `min_samples`: Minimum samples for DBSCAN

### `/analyze_text` (POST)
Analyze word embeddings in custom text.

**Parameters:**
- `text`: Text to analyze
- `method`: Clustering method
- `n_clusters`: Number of clusters
- `min_word_length`: Minimum word length
- `eps`: Epsilon parameter for DBSCAN
- `min_samples`: Minimum samples for DBSCAN

### `/download_results` (POST)
Download analysis results as JSON file.

### `/health` (GET)
Health check endpoint.

### `/test` (GET)
Test endpoint with sample data.

## Output Format

The analyzer returns comprehensive results including:

```json
{
  "success": true,
  "results": {
    "transcript_length": 1234,
    "embedding_results": {
      "embedded_words": 45,
      "total_words": 67,
      "word_embeddings": {...},
      "word_contexts": {...}
    },
    "cluster_results": {
      "method": "kmeans",
      "n_clusters": 5,
      "silhouette_score": 0.234,
      "word_clusters": {...}
    },
    "cluster_analysis": {
      "cluster_0": {
        "size": 12,
        "words": ["government", "policy", "law", ...],
        "political_terms": ["government", "policy"],
        "bias_indicators": {
          "loaded_language": 2,
          "emotional_words": 1,
          "subjective_terms": 0
        },
        "semantic_groups": {
          "political": ["government", "policy"],
          "economic": ["economy", "business"]
        }
      }
    },
    "visualizations": {
      "word_scatter": "<plotly_html>",
      "cluster_summary": "<plotly_html>",
      "word_frequency": "<plotly_html>",
      "semantic_analysis": "<plotly_html>"
    }
  }
}
```

## Clustering Methods

### K-Means Clustering
- **Best for**: When you know the expected number of clusters
- **Parameters**: `n_clusters` (number of clusters)
- **Pros**: Fast, simple, works well with spherical clusters
- **Cons**: Requires specifying number of clusters, sensitive to initialization

### DBSCAN Clustering
- **Best for**: Discovering clusters of varying sizes and shapes
- **Parameters**: `eps` (neighborhood radius), `min_samples` (minimum points)
- **Pros**: No need to specify cluster count, handles noise
- **Cons**: Sensitive to parameter tuning, may not work well with varying densities

### Agglomerative Clustering
- **Best for**: Hierarchical clustering with specified number of clusters
- **Parameters**: `n_clusters` (number of clusters)
- **Pros**: Creates hierarchical structure, deterministic
- **Cons**: Computationally expensive for large datasets

## Bias Analysis

The analyzer identifies several types of bias indicators:

### Political Terms
- Government-related: government, politics, election, policy
- Economic terms: economy, business, market, trade
- Social issues: rights, freedom, justice, equality

### Bias Indicators
- **Loaded Language**: Words with strong emotional connotations
- **Emotional Words**: Words expressing emotions or feelings
- **Subjective Terms**: Words indicating opinion or judgment

### Semantic Groups
- **Political**: Government, policy, law-related terms
- **Economic**: Business, finance, trade-related terms
- **Social**: Society, community, family-related terms
- **Technical**: Technology, science, data-related terms
- **Emotional**: Feelings, emotions, attitudes

## Troubleshooting

### Common Issues

1. **BERT model loading fails**:
   - Check internet connection for model download
   - Ensure sufficient disk space
   - Try different model: `BERTWordClusteringAnalyzer("all-MiniLM-L6-v2")`

2. **Clustering fails**:
   - Increase minimum word length
   - Reduce number of clusters
   - Try different clustering method

3. **Visualization errors**:
   - Install missing libraries: `pip install plotly matplotlib seaborn`
   - Check if running in headless environment

4. **Memory issues**:
   - Reduce transcript length
   - Use smaller BERT model
   - Process in smaller chunks

### Performance Tips

- Use GPU if available (automatically detected)
- Process shorter texts for faster analysis
- Adjust clustering parameters based on text length
- Use K-Means for faster clustering with known cluster count

## Dependencies

- **BERT/Transformers**: `torch`, `transformers`, `sentence-transformers`
- **Machine Learning**: `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Web Framework**: `flask`
- **Data Processing**: `numpy`, `pandas`
- **Text Processing**: `nltk`, `textblob`
- **YouTube**: `youtube-transcript-api`

## License

This project is part of the YouTube Transcript Analysis suite.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test script for examples
3. Check the health endpoint: `http://localhost:5001/health`
4. Run the test endpoint: `http://localhost:5001/test` 