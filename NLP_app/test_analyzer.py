#!/usr/bin/env python3
"""
Test script for Political Bias Analyzer
Demonstrates the analyzer with sample text
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from political_bias_analyzer import PoliticalBiasAnalyzer

def test_analyzer():
    """Test the political bias analyzer with sample text"""
    
    print("Testing Political Bias Analyzer")
    print("=" * 50)
    
    # Sample texts with different levels of bias
    sample_texts = {
        "Balanced Text": """
        The economic policy debate continues to generate discussion among experts. 
        Some economists argue that the current approach will lead to growth, while 
        others express concerns about potential inflation. Recent data shows mixed 
        results, with employment numbers improving but inflation remaining above 
        target levels. The Federal Reserve has indicated it will continue monitoring 
        the situation and adjust policy as needed. Both supporters and critics of 
        the current policy acknowledge that the economic recovery is complex and 
        multifaceted.
        """,
        
        "Biased Text (Negative)": """
        The disastrous economic policies of this administration have completely 
        destroyed our economy! Everyone knows that these radical socialist policies 
        are a complete failure. The so-called "experts" who support these policies 
        are clearly biased and don't understand basic economics. The data clearly 
        proves that these policies are causing massive inflation and destroying jobs. 
        It's absolutely outrageous that anyone would defend these dangerous policies 
        that are clearly designed to undermine our country's economic stability.
        """,
        
        "Biased Text (Positive)": """
        The amazing economic policies implemented by this administration have 
        completely transformed our economy for the better! These brilliant policies 
        have created unprecedented growth and prosperity. All the experts agree that 
        these policies are revolutionary and will lead to a golden age of economic 
        success. The data clearly shows that these policies are working perfectly 
        and creating millions of jobs. It's absolutely incredible how successful 
        these policies have been in such a short time.
        """
    }
    
    try:
        # Initialize analyzer
        print("Initializing analyzer...")
        analyzer = PoliticalBiasAnalyzer()
        print("✓ Analyzer initialized successfully\n")
        
        # Test each sample text
        for text_name, text in sample_texts.items():
            print(f"Testing: {text_name}")
            print("-" * 30)
            
            # Analyze the text
            results = analyzer.analyze_transcript(text)
            
            if 'error' in results:
                print(f"✗ Error: {results['error']}")
                continue
            
            # Display key results
            bias_assessment = results.get('bias_assessment', {})
            sentiment = results.get('sentiment_analysis', {})
            linguistic = results.get('linguistic_markers', {})
            
            print(f"Bias Level: {bias_assessment.get('bias_level', 'Unknown')}")
            print(f"Bias Score: {bias_assessment.get('overall_bias_score', 0):.2f}")
            print(f"Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
            print(f"Subjectivity: {sentiment.get('subjectivity_score', 0):.2f}")
            print(f"Loaded Language: {linguistic.get('loaded_language_count', 0):.1f} per 1000 words")
            
            # Show bias factors
            bias_factors = bias_assessment.get('bias_factors', [])
            if bias_factors:
                print("Bias Factors:")
                for factor in bias_factors:
                    print(f"  • {factor}")
            
            print()
        
        # Test report generation
        print("Testing Report Generation")
        print("-" * 30)
        
        # Use the balanced text for report generation
        balanced_results = analyzer.analyze_transcript(sample_texts["Balanced Text"])
        report = analyzer.generate_report(balanced_results)
        
        # Show first few lines of the report
        report_lines = report.split('\n')[:20]
        print("Sample Report (first 20 lines):")
        for line in report_lines:
            print(line)
        print("...")
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_youtube_integration():
    """Test YouTube video analysis (requires internet connection)"""
    
    print("\nTesting YouTube Integration")
    print("=" * 50)
    
    # This is a sample YouTube URL - replace with a real one for testing
    sample_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll
    
    try:
        from political_bias_analyzer import analyze_youtube_video
        
        print(f"Attempting to analyze: {sample_url}")
        print("Note: This will only work if the video has available transcripts")
        
        results = analyze_youtube_video(sample_url)
        
        if 'error' in results:
            print(f"Expected error (no transcript): {results['error']}")
        else:
            print("✓ YouTube analysis successful!")
            bias_assessment = results.get('bias_assessment', {})
            print(f"Bias Level: {bias_assessment.get('bias_level', 'Unknown')}")
            print(f"Bias Score: {bias_assessment.get('overall_bias_score', 0):.2f}")
        
    except Exception as e:
        print(f"✗ YouTube test failed: {e}")

if __name__ == "__main__":
    test_analyzer()
    
    # Uncomment the line below to test YouTube integration
    # test_youtube_integration() 