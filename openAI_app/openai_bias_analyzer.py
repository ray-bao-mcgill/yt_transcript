#!/usr/bin/env python3
"""
OpenAI Political Bias Analyzer for YouTube Transcripts
Simple app that uses OpenAI's API to analyze political bias in transcripts
"""

import os
import sys
import json
from typing import Dict, Optional
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_transcript import get_transcript

class OpenAIBiasAnalyzer:
    """
    Simple OpenAI-based political bias analyzer
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI bias analyzer"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_transcript(self, transcript: str, video_title: str = "Unknown Video") -> Dict:
        """
        Analyze political bias in a transcript using OpenAI
        
        Args:
            transcript: The transcript text to analyze
            video_title: Title of the video for context
            
        Returns:
            Dictionary with bias analysis results
        """
        try:
            # Create the prompt for OpenAI
            prompt = self._create_analysis_prompt(transcript, video_title)
            
            # Call OpenAI API
            response = self._call_openai_api(prompt)
            
            if response.get('success'):
                return {
                    'success': True,
                    'video_title': video_title,
                    'transcript_length': len(transcript),
                    'analysis': response['analysis'],
                    'raw_response': response['raw_response']
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error occurred'),
                    'video_title': video_title,
                    'transcript_length': len(transcript)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'video_title': video_title,
                'transcript_length': len(transcript)
            }
    
    def _create_analysis_prompt(self, transcript: str, video_title: str) -> str:
        """Create the prompt for OpenAI analysis"""
        return f"""
You are an expert political analyst. Analyze the following YouTube video transcript for political bias.

Video Title: {video_title}
Transcript Length: {len(transcript)} characters

TRANSCRIPT:
{transcript}

Please provide a comprehensive political bias analysis in the following JSON format:

{{
    "overall_bias": "left/right/center/neutral",
    "bias_confidence": "high/medium/low",
    "bias_score": -100 to 100 (negative = left, positive = right, 0 = neutral),
    "key_indicators": [
        "specific phrases or topics that indicate bias"
    ],
    "political_topics": [
        "main political topics discussed"
    ],
    "framing_analysis": "how the content is framed or presented",
    "loaded_language": [
        "examples of emotionally charged or biased language"
    ],
    "source_credibility": "high/medium/low",
    "recommendations": [
        "suggestions for balanced perspective"
    ],
    "summary": "brief summary of the analysis"
}}

Focus on:
1. Political leanings (left/right/center)
2. Use of loaded language
3. Framing of issues
4. Source credibility
5. Balance of perspectives

Be objective and evidence-based in your analysis.
"""
    
    def _call_openai_api(self, prompt: str) -> Dict:
        """Call OpenAI API with the analysis prompt"""
        try:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert political analyst specializing in bias detection and media analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Try to parse JSON from the response
                try:
                    analysis = json.loads(content)
                    return {
                        'success': True,
                        'analysis': analysis,
                        'raw_response': content
                    }
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw text
                    return {
                        'success': True,
                        'analysis': {
                            'raw_analysis': content,
                            'parse_error': 'Could not parse JSON response'
                        },
                        'raw_response': content
                    }
            else:
                return {
                    'success': False,
                    'error': f'OpenAI API error: {response.status_code} - {response.text}'
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def analyze_youtube_video(self, video_url: str) -> Dict:
        """
        Analyze political bias in a YouTube video
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dictionary with bias analysis results
        """
        try:
            # Extract transcript
            transcript_result = get_transcript(video_url)
            
            if not transcript_result.get('success', False):
                return {
                    'success': False,
                    'error': transcript_result.get('error', 'Could not extract transcript'),
                    'video_url': video_url
                }
            
            transcript = transcript_result['transcript_text']
            video_title = transcript_result.get('title', 'Unknown Video')
            
            # Analyze the transcript
            return self.analyze_transcript(transcript, video_title)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'video_url': video_url
            }


def main():
    """Test the OpenAI bias analyzer"""
    print("OpenAI Political Bias Analyzer")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        analyzer = OpenAIBiasAnalyzer(api_key)
        print("✅ OpenAI Bias Analyzer initialized successfully")
        
        # Test with sample text
        sample_transcript = """
        The government should reduce regulations and allow free market principles to drive economic growth. 
        Lower taxes will stimulate business investment and create jobs for hardworking Americans. 
        We need to strengthen our borders and enforce immigration laws. 
        The rule of law is fundamental to our democracy and national security.
        """
        
        print("\n🔍 Analyzing sample transcript...")
        results = analyzer.analyze_transcript(sample_transcript, "Sample Political Video")
        
        if results['success']:
            print("✅ Analysis completed successfully!")
            print(f"Transcript length: {results['transcript_length']} characters")
            
            analysis = results['analysis']
            if isinstance(analysis, dict):
                print(f"Overall bias: {analysis.get('overall_bias', 'N/A')}")
                print(f"Bias score: {analysis.get('bias_score', 'N/A')}")
                print(f"Confidence: {analysis.get('bias_confidence', 'N/A')}")
                print(f"Summary: {analysis.get('summary', 'N/A')}")
            else:
                print("Raw analysis:", analysis)
        else:
            print(f"❌ Analysis failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 