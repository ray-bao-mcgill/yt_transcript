#!/usr/bin/env python3
"""
Test script for Improved Political Bias Analyzer
Demonstrates the enhanced bias detection with highly political texts
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from political_bias_analyzer import PoliticalBiasAnalyzer

def test_improved_analyzer():
    """Test the improved political bias analyzer with highly political texts"""
    
    print("Testing Improved Political Bias Analyzer")
    print("=" * 60)
    
    # Sample texts with different levels of political bias
    sample_texts = {
        "Highly Biased Conservative": """
        The radical left-wing Democrats are destroying our country with their socialist agenda! 
        These corrupt politicians are pushing dangerous policies that will bankrupt America. 
        Everyone knows that liberal policies have failed everywhere they've been tried. 
        The fake news media is spreading lies about conservatives while covering up the truth. 
        We need to take back our country from these radical extremists who want to destroy our 
        traditional values and replace them with their communist ideology. The evidence clearly 
        shows that conservative policies work while liberal policies fail. It's absolutely 
        outrageous what these Democrats are doing to our great nation!
        """,
        
        "Highly Biased Liberal": """
        The right-wing Republicans are absolutely destroying our democracy with their fascist 
        agenda! These corrupt politicians are pushing dangerous policies that will harm millions 
        of Americans. Everyone knows that conservative policies have failed and hurt working 
        families. The corporate media is spreading lies about progressives while covering up 
        the truth about Republican corruption. We need to fight back against these radical 
        extremists who want to destroy our democratic institutions and replace them with their 
        authoritarian ideology. The data clearly proves that progressive policies work while 
        conservative policies fail. It's absolutely disgusting what these Republicans are doing 
        to our country!
        """,
        
        "Moderately Political": """
        The current political climate is quite divisive, with both parties taking extreme 
        positions on important issues. Some people believe that the government should play 
        a larger role in healthcare, while others think the private sector should handle it. 
        The economy has been affected by recent policy changes, and experts disagree on the 
        best approach. Immigration policy continues to be a controversial topic that divides 
        the nation. Both sides have valid points, but the extreme rhetoric from both parties 
        makes it difficult to find common ground.
        """,
        
        "Neutral/Non-Political": """
        The weather today is quite pleasant with temperatures in the mid-70s. The local 
        farmers market has fresh produce available, including tomatoes, lettuce, and various 
        fruits. The community center is hosting a cooking class this weekend where people 
        can learn to make traditional recipes. The library has extended its hours to 
        accommodate more visitors. Local businesses are reporting good sales this month, 
        which is positive for the local economy.
        """
    }
    
    try:
        # Initialize analyzer
        print("Initializing improved analyzer...")
        analyzer = PoliticalBiasAnalyzer()
        print("✓ Analyzer initialized successfully\n")
        
        # Test each sample text
        for text_name, text in sample_texts.items():
            print(f"Testing: {text_name}")
            print("-" * 40)
            
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
            print(f"Bias Score: {bias_assessment.get('overall_bias_score', 0):.3f}")
            print(f"Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
            print(f"Subjectivity: {sentiment.get('subjectivity_score', 0):.3f}")
            print(f"Loaded Language: {linguistic.get('loaded_language_count', 0):.1f} per 1000 words")
            print(f"Political Content: {linguistic.get('political_content_score', 0):.1f} per 1000 words")
            print(f"Partisan Language: {linguistic.get('partisan_language', 0):.1f} per 1000 words")
            print(f"Us vs Them: {linguistic.get('us_vs_them_phrases', 0):.1f} per 1000 words")
            
            # Show bias factors
            bias_factors = bias_assessment.get('bias_factors', [])
            if bias_factors:
                print("Bias Factors:")
                for factor in bias_factors:
                    print(f"  • {factor}")
            
            print()
        
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_analyzer() 