#!/usr/bin/env python3
"""
Simple YouTube Analysis Interface
"""

from youtube_transcript import get_transcript

def analyze_video(video_url: str):
    """Simple function to analyze a YouTube video"""
    
    # Get transcript
    print(f"Getting transcript for: {video_url}")
    transcript_result = get_transcript(video_url)
    
    print(transcript_result['transcript_text'])
    
    if not transcript_result["success"]:
        print(f"Error: {transcript_result['error']}")
        return

def main():
    """Simple interactive mode"""
    print("YouTube Analysis Tool")
    
    url = input("Enter YouTube URL: ")
    
    analyze_video(url)

if __name__ == "__main__":
    main()