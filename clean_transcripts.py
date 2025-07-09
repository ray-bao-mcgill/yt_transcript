#!/usr/bin/env python3
"""
Clean transcript files by removing timestamps
Removes timestamp patterns like "00:00:00.000" from transcript files
"""

import os
import re
import shutil
from pathlib import Path

def clean_transcript_file(file_path):
    """Remove timestamps from a transcript file and save the cleaned version."""
    
    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_path = str(file_path) + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Remove timestamps using regex
    # Pattern matches: HH:MM:SS.mmm followed by space and text
    # Also handles "No text" lines
    cleaned_lines = []
    for line in content.split('\n'):
        if line.strip():
            # Remove timestamp pattern: HH:MM:SS.mmm at the beginning
            cleaned_line = re.sub(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s+', '', line)
            # Skip lines that are just "No text"
            if cleaned_line.strip() != "No text":
                cleaned_lines.append(cleaned_line)
    
    # Write cleaned content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Cleaned: {file_path}")
    return len(cleaned_lines)

def main():
    """Clean all transcript files in the data directory."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("Data directory not found!")
        return
    
    transcript_files = list(data_dir.glob("*.txt"))
    
    if not transcript_files:
        print("No transcript files found in data directory!")
        return
    
    print(f"Found {len(transcript_files)} transcript files to clean:")
    for file_path in transcript_files:
        print(f"  - {file_path.name}")
    
    print("\nCleaning files...")
    total_lines = 0
    
    for file_path in transcript_files:
        try:
            lines_cleaned = clean_transcript_file(file_path)
            total_lines += lines_cleaned
        except Exception as e:
            print(f"Error cleaning {file_path}: {e}")
    
    print(f"\nCleaning complete! Processed {total_lines} total lines.")
    print("Original files have been backed up with .backup extension.")

if __name__ == "__main__":
    main() 