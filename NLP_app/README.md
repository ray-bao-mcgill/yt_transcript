# Political Bias Analyzer

A comprehensive NLP-based tool for analyzing political bias in YouTube transcripts and text content. This tool uses multiple advanced techniques to provide a holistic assessment of political bias, going beyond simple keyword matching to understand context and meaning.

## Features

### 🧠 Advanced NLP Analysis
- **Sentiment Analysis**: Analyzes emotional tone and subjectivity
- **Linguistic Markers**: Detects loaded language, subjective markers, and authority claims
- **Topic Modeling**: Identifies main themes and controversial topics
- **Framing Analysis**: Analyzes how topics are presented and framed
- **Credibility Assessment**: Evaluates source credibility and fact-checking indicators
- **Contextual Analysis**: Examines argument structure and vocabulary diversity

### 🎯 Holistic Approach
- **Multi-dimensional Analysis**: Combines multiple NLP techniques for comprehensive assessment
- **Context-Aware**: Considers the context in which words and phrases are used
- **Bias Scoring**: Provides quantitative bias scores with confidence levels
- **Detailed Reports**: Generates human-readable reports with actionable insights

### 🌐 Multiple Interfaces
- **Web Application**: Modern, responsive web interface
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Python API**: Direct integration with existing Python code

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)

### 1. Install Dependencies
```bash
cd yt_transcript/NLP_app
pip install -r requirements.txt
```

### 2. Download Required Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('subjectivity')"
```

### 3. Install FFmpeg
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

## Usage

### Web Interface
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

### Command Line Interface

#### Analyze YouTube Video
```bash
python cli_analyzer.py --youtube "https://www.youtube.com/watch?v=example"
```

#### Analyze Custom Text
```bash
python cli_analyzer.py --text "Your text here to analyze for political bias"
```

#### Analyze Text File
```bash
python cli_analyzer.py --text-file transcript.txt
```

#### Save Results to File
```bash
python cli_analyzer.py --youtube "https://youtu.be/example" --output results.json
```

#### Verbose Output
```bash
python cli_analyzer.py --youtube "https://youtu.be/example" --verbose
```

### Python API
```python
from political_bias_analyzer import PoliticalBiasAnalyzer, analyze_youtube_video

# Analyze YouTube video
results = analyze_youtube_video("https://www.youtube.com/watch?v=example")

# Or analyze custom text
analyzer = PoliticalBiasAnalyzer()
results = analyzer.analyze_transcript("Your text here")

# Generate report
report = analyzer.generate_report(results)
print(report)
```

## Analysis Components

### 1. Sentiment Analysis
- **Overall Sentiment**: Positive, negative, or neutral
- **Subjectivity Score**: How subjective vs. objective the content is
- **Emotional Intensity**: Frequency of emotionally charged language

### 2. Linguistic Markers
- **Loaded Language**: Words with strong emotional connotations
- **Subjective Markers**: Language indicating personal opinion
- **Authority Claims**: References to experts, studies, or research
- **One-sided Arguments**: Absolute statements and generalizations

### 3. Topic Analysis
- **Main Topics**: Primary themes discussed
- **Topic Diversity**: How varied the topics are
- **Political Topics**: Topics related to politics and government
- **Controversial Topics**: Potentially divisive subjects

### 4. Framing Analysis
- **Economic Framing**: Focus on economic aspects
- **Moral Framing**: Emphasis on values and ethics
- **Security Framing**: Focus on safety and protection
- **Social Framing**: Emphasis on community and society

### 5. Credibility Assessment
- **Factual Claims**: References to data and evidence
- **Opinion Markers**: Clear indicators of personal views
- **Uncertainty Markers**: Language indicating doubt or uncertainty
- **Qualifiers**: Words that limit or modify statements

## Output Format

The analyzer provides comprehensive results including:

```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "transcript_length": 5000,
  "analysis_timestamp": "2024-01-01T12:00:00",
  
  "sentiment_analysis": {
    "overall_sentiment": "negative",
    "subjectivity_score": 0.75,
    "emotional_intensity": 12.5
  },
  
  "linguistic_markers": {
    "loaded_language_count": 8.2,
    "subjective_markers": 15.3,
    "authority_claims": 3.1
  },
  
  "bias_assessment": {
    "overall_bias_score": 0.65,
    "bias_level": "High",
    "bias_factors": [
      "High use of loaded language",
      "Negative sentiment bias",
      "Low credibility indicators"
    ]
  },
  
  "recommendations": [
    "Consider seeking multiple perspectives on this topic",
    "Verify factual claims with independent sources"
  ]
}
```

## Bias Detection Methodology

The analyzer uses a sophisticated approach that goes beyond simple keyword matching:

### 1. Contextual Analysis
- Considers the context in which words and phrases appear
- Analyzes sentence structure and relationships
- Evaluates argument coherence and logic

### 2. Multi-dimensional Scoring
- Combines multiple analysis techniques
- Weights different factors based on their significance
- Provides confidence levels for assessments

### 3. Holistic Assessment
- Looks at the overall presentation and framing
- Considers source credibility and evidence
- Evaluates balance and fairness in presentation

## Limitations

- **Language Support**: Currently optimized for English content
- **Model Dependencies**: Requires significant computational resources for full analysis
- **Context Sensitivity**: May not capture all nuances of political discourse
- **Bias in Training Data**: Models may reflect biases in their training data

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the YouTube transcript extraction system
- Uses state-of-the-art NLP libraries and models
- Inspired by research in media bias detection and political communication analysis 