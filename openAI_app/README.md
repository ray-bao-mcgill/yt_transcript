# OpenAI Political Bias Analyzer

A simple web application that uses OpenAI's GPT models to analyze political bias in YouTube transcripts and custom text.

## Features

- **YouTube Video Analysis**: Extract transcripts from YouTube videos and analyze political bias
- **Custom Text Analysis**: Analyze political bias in any custom text input
- **Comprehensive Analysis**: Get detailed bias scores, indicators, and recommendations
- **Modern Web Interface**: Clean, responsive web UI for easy interaction
- **JSON Export**: Download analysis results for further processing

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

You need an OpenAI API key to use this application. Set it as an environment variable:

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

**Or create a .env file:**
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at: http://localhost:5002

## Usage

### Web Interface

1. **YouTube Video Analysis**:
   - Go to the "YouTube Video" tab
   - Enter a YouTube URL
   - Click "Analyze YouTube Video"
   - View the results in the "Results" tab

2. **Custom Text Analysis**:
   - Go to the "Custom Text" tab
   - Enter a title (optional)
   - Paste your text (minimum 50 characters)
   - Click "Analyze Custom Text"
   - View the results in the "Results" tab

### API Endpoints

- `GET /` - Main web interface
- `POST /analyze_youtube` - Analyze YouTube video
- `POST /analyze_text` - Analyze custom text
- `POST /download_results` - Download results as JSON
- `GET /health` - Health check
- `GET /test` - Test endpoint

### Command Line Usage

You can also use the analyzer directly from Python:

```python
from openai_bias_analyzer import OpenAIBiasAnalyzer

# Initialize analyzer
analyzer = OpenAIBiasAnalyzer()

# Analyze YouTube video
results = analyzer.analyze_youtube_video("https://www.youtube.com/watch?v=...")

# Analyze custom text
results = analyzer.analyze_transcript("Your text here", "Title")

# Check results
if results['success']:
    print(f"Bias: {results['analysis']['overall_bias']}")
    print(f"Score: {results['analysis']['bias_score']}")
else:
    print(f"Error: {results['error']}")
```

## Analysis Output

The analyzer provides comprehensive political bias analysis including:

- **Overall Bias**: left/right/center/neutral
- **Bias Score**: -100 to 100 (negative = left, positive = right)
- **Confidence Level**: high/medium/low
- **Key Indicators**: Specific phrases or topics indicating bias
- **Political Topics**: Main political topics discussed
- **Framing Analysis**: How content is presented
- **Loaded Language**: Examples of emotionally charged language
- **Source Credibility**: Assessment of source reliability
- **Recommendations**: Suggestions for balanced perspective
- **Summary**: Brief analysis summary

## Example Output

```json
{
  "overall_bias": "right",
  "bias_confidence": "high",
  "bias_score": 75,
  "key_indicators": [
    "free market principles",
    "lower taxes",
    "strengthen borders"
  ],
  "political_topics": [
    "economic policy",
    "immigration",
    "regulation"
  ],
  "framing_analysis": "Content frames government intervention as negative and free markets as positive",
  "loaded_language": [
    "hardworking Americans",
    "rule of law"
  ],
  "source_credibility": "medium",
  "recommendations": [
    "Consider alternative economic perspectives",
    "Seek balanced coverage of policy issues"
  ],
  "summary": "Content shows clear conservative/right-leaning bias with emphasis on free market principles and limited government."
}
```

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls and YouTube transcript extraction

## Troubleshooting

### Common Issues

1. **"OpenAI API key not provided"**
   - Make sure you've set the `OPENAI_API_KEY` environment variable
   - Check that the API key is valid and has sufficient credits

2. **"Could not extract transcript from video"**
   - The video might not have subtitles/transcripts available
   - Try a different video or check if the URL is correct

3. **"Analysis failed"**
   - Check your internet connection
   - Verify your OpenAI API key is valid
   - Ensure the text is long enough (minimum 50 characters)

### API Limits

- OpenAI API has rate limits and usage costs
- GPT-3.5-turbo is used by default for cost efficiency
- Consider upgrading to GPT-4 for more detailed analysis

## License

This project is for educational and research purposes. Please ensure you comply with OpenAI's terms of service and YouTube's terms of service when using this application. 