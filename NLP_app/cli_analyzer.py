#!/usr/bin/env python3
"""
Command Line Interface for Political Bias Analysis
Provides easy access to analyze YouTube videos and text for political bias
"""

import argparse
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from political_bias_analyzer import PoliticalBiasAnalyzer, analyze_youtube_video

def main():
    parser = argparse.ArgumentParser(
        description='Analyze YouTube videos and text for political bias using NLP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_analyzer.py --youtube "https://www.youtube.com/watch?v=example"
  python cli_analyzer.py --text "Your text here to analyze for political bias"
  python cli_analyzer.py --youtube "https://youtu.be/example" --output results.json
  python cli_analyzer.py --text-file transcript.txt --verbose
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--youtube', '-y', 
                           help='YouTube video URL to analyze')
    input_group.add_argument('--text', '-t', 
                           help='Text to analyze for political bias')
    input_group.add_argument('--text-file', '-f', 
                           help='File containing text to analyze')
    
    # Output options
    parser.add_argument('--output', '-o', 
                       help='Output file for results (JSON format)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only display the human-readable report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip displaying the human-readable report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    if args.verbose:
        print("Initializing Political Bias Analyzer...")
    
    try:
        analyzer = PoliticalBiasAnalyzer()
        if args.verbose:
            print("✓ Analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Get text to analyze
    text_to_analyze = None
    source_info = None
    
    if args.youtube:
        if args.verbose:
            print(f"Analyzing YouTube video: {args.youtube}")
        
        results = analyze_youtube_video(args.youtube)
        
        if 'error' in results:
            print(f"✗ Error: {results['error']}")
            sys.exit(1)
        
        text_to_analyze = results.get('transcript_text', '')
        source_info = f"YouTube video: {args.youtube}"
        
    elif args.text:
        text_to_analyze = args.text
        source_info = "Custom text input"
        
    elif args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text_to_analyze = f.read()
            source_info = f"Text file: {args.text_file}"
        except Exception as e:
            print(f"✗ Error reading file {args.text_file}: {e}")
            sys.exit(1)
    
    if not text_to_analyze or len(text_to_analyze.strip()) < 50:
        print("✗ Error: Text too short for meaningful analysis (minimum 50 characters)")
        sys.exit(1)
    
    # Perform analysis
    if args.verbose:
        print(f"Analyzing {len(text_to_analyze)} characters of text...")
    
    results = analyzer.analyze_transcript(text_to_analyze, args.youtube)
    
    if 'error' in results:
        print(f"✗ Analysis error: {results['error']}")
        sys.exit(1)
    
    # Display results
    if not args.report_only:
        print("\n" + "="*60)
        print("POLITICAL BIAS ANALYSIS RESULTS")
        print("="*60)
        
        # Basic info
        print(f"Source: {source_info}")
        print(f"Text Length: {results.get('transcript_length', 0)} characters")
        print(f"Analysis Date: {results.get('analysis_timestamp', 'Unknown')}")
        print()
        
        # Bias assessment
        bias_assessment = results.get('bias_assessment', {})
        print("OVERALL ASSESSMENT:")
        print(f"  Bias Level: {bias_assessment.get('bias_level', 'Unknown')}")
        print(f"  Bias Score: {bias_assessment.get('overall_bias_score', 0):.2f}")
        print(f"  Confidence: {bias_assessment.get('confidence', 'Unknown')}")
        print()
        
        # Key factors
        bias_factors = bias_assessment.get('bias_factors', [])
        if bias_factors:
            print("KEY BIAS FACTORS:")
            for factor in bias_factors:
                print(f"  • {factor}")
            print()
        
        # Quick stats
        sentiment = results.get('sentiment_analysis', {})
        linguistic = results.get('linguistic_markers', {})
        
        print("QUICK STATS:")
        print(f"  Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        print(f"  Subjectivity: {sentiment.get('subjectivity_score', 0):.2f}")
        print(f"  Loaded Language: {linguistic.get('loaded_language_count', 0):.1f} per 1000 words")
        print()
    
    # Display detailed report
    if not args.no_report:
        report = analyzer.generate_report(results)
        print(report)
    
    # Save results to file
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {args.output}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")
    
    if args.verbose:
        print("\n✓ Analysis completed successfully")

if __name__ == '__main__':
    main() 