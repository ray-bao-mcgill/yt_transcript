#!/usr/bin/env python3
"""
Enhanced YouTube Transcript Extractor with Speech-to-Text Fallback
Takes a YouTube URL and returns the full transcript
Includes workarounds for YouTube Shorts and speech-to-text for videos without subtitles
"""

import re
import os
import tempfile
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Import speech-to-text libraries (optional)
try:
    import yt_dlp
    import whisper
    SPEECH_TO_TEXT_AVAILABLE = True
except ImportError:
    SPEECH_TO_TEXT_AVAILABLE = False
    print("Warning: Speech-to-text libraries not available. Install yt-dlp and whisper for full functionality.")

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats including Shorts"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        r'youtu\.be\/([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume the input is already a video ID
    return url

def is_shorts_url(url):
    """Check if the URL is a YouTube Shorts URL"""
    return 'youtube.com/shorts/' in url or 'youtu.be/' in url

def get_regular_video_url(shorts_url):
    """Convert Shorts URL to regular video URL"""
    video_id = extract_video_id(shorts_url)
    return f"https://www.youtube.com/watch?v={video_id}"

def download_audio(video_url, output_path):
    """Download audio from YouTube video"""
    if not SPEECH_TO_TEXT_AVAILABLE:
        return False
        
    # Try different audio download methods
    methods = [
        # Method 1: Try with FFmpeg postprocessing
        {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True
        },
        # Method 2: Try without FFmpeg (direct audio download)
        {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',
            'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        },
        # Method 3: Try with minimal options
        {
            'format': 'bestaudio',
            'outtmpl': output_path.replace('.mp3', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
    ]
    
    for i, ydl_opts in enumerate(methods):
        try:
            print(f"   Trying audio download method {i+1}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Check if any audio file was created
            base_path = output_path.replace('.mp3', '')
            possible_extensions = ['.mp3', '.m4a', '.webm', '.wav', '.ogg']
            
            for ext in possible_extensions:
                if os.path.exists(base_path + ext):
                    # If we got a different format, convert or use as is
                    if ext != '.mp3':
                        print(f"   Audio downloaded as {ext}, using as is")
                        return True
                    return True
            
            print(f"   Method {i+1} completed but no audio file found")
            
        except Exception as e:
            error_msg = str(e)
            print(f"   Method {i+1} failed: {error_msg}")
            
            # If it's an FFmpeg error, try the next method
            if "ffprobe" in error_msg.lower() or "ffmpeg" in error_msg.lower():
                print("   FFmpeg not available, trying alternative method...")
                continue
            else:
                # For other errors, try the next method anyway
                continue
    
    print("   All audio download methods failed")
    return False

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    if not SPEECH_TO_TEXT_AVAILABLE:
        return None
        
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")  # Use base model for speed
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def find_audio_file(temp_dir):
    """Find any audio file in the temporary directory"""
    possible_extensions = ['.mp3', '.m4a', '.webm', '.wav', '.ogg']
    
    for ext in possible_extensions:
        for filename in os.listdir(temp_dir):
            if filename.endswith(ext):
                return os.path.join(temp_dir, filename)
    
    return None

def get_video_metadata(video_url):
    """Get video metadata as fallback"""
    if not SPEECH_TO_TEXT_AVAILABLE:
        return None
        
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            # Combine title, description, and tags
            title = info.get('title', '')
            description = info.get('description', '')
            tags = info.get('tags', [])
            
            # Create minimal transcript from metadata
            metadata_text = f"{title}\n\n{description}"
            if tags:
                metadata_text += f"\n\nTags: {', '.join(tags[:10])}"  # Limit tags
            
            return {
                "title": title,
                "description": description,
                "tags": tags,
                "text": metadata_text,
                "video_id": info.get('id')
            }
                
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None

def get_transcript(video_id_or_url, languages=None):
    """
    Get transcript for a YouTube video with multiple fallback methods
    
    Args:
        video_id_or_url: YouTube video ID or URL
        languages: List of language codes to try (default: ['en', 'en-US', 'en-GB'])
    
    Returns:
        Dictionary with transcript data
    """
    if languages is None:
        languages = ['en', 'en-US', 'en-GB']
    
    video_id = extract_video_id(video_id_or_url)
    is_shorts = is_shorts_url(video_id_or_url)
    
    print(f"🔍 Attempting to get transcript for video: {video_id}")
    
    # Method 1: Try YouTube Transcript API
    print("📝 Method 1: Trying YouTube Transcript API...")
    transcript_result = try_youtube_transcript_api(video_id, languages, is_shorts)
    
    if transcript_result["success"]:
        print("✅ Success: Found existing transcript")
        return transcript_result
    
    # Method 2: Try speech-to-text with Whisper
    if SPEECH_TO_TEXT_AVAILABLE:
        print("🎤 Method 2: Trying speech-to-text transcription...")
        whisper_result = try_whisper_transcription(video_id_or_url)
        
        if whisper_result["success"]:
            print("✅ Success: Generated transcript using speech-to-text")
            return whisper_result
    
    # Method 3: Try to get video metadata as fallback
    print("📊 Method 3: Trying to extract video metadata...")
    metadata_result = try_metadata_extraction(video_id_or_url)
    
    if metadata_result["success"]:
        print("⚠️  Partial success: Using video metadata (limited content)")
        return metadata_result
    
    # All methods failed
    return {
        "success": False,
        "error": "Could not extract transcript or audio from this video",
        "video_id": video_id,
        "is_shorts": is_shorts,
        "methods_tried": ["youtube_transcript_api", "whisper", "metadata"]
    }

def try_youtube_transcript_api(video_id, languages, is_shorts):
    """Try to get transcript using YouTube Transcript API"""
    approaches = []
    
    if is_shorts:
        approaches.append(("Regular video URL", video_id))
        approaches.append(("Shorts URL", video_id))
    else:
        approaches.append(("Standard URL", video_id))
    
    for approach_name, vid_id in approaches:
        try:
            print(f"   Trying {approach_name}...")
            
            # Get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
            
            # Try to get transcript in preferred languages
            transcript = None
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except:
                    continue
            
            # If no transcript found in preferred languages, get the first available
            if transcript is None:
                transcript = transcript_list.find_transcript(transcript_list)
            
            # Get the transcript data
            transcript_data = transcript.fetch()
            
            # Format as plain text
            formatter = TextFormatter()
            formatted_text = formatter.format_transcript(transcript_data)
            
            return {
                "success": True,
                "video_id": vid_id,
                "language": transcript.language,
                "language_code": transcript.language_code,
                "transcript_text": formatted_text,
                "word_count": len(formatted_text.split()),
                "character_count": len(formatted_text),
                "method": "youtube_transcript_api",
                "approach_used": approach_name,
                "is_shorts": is_shorts
            }
            
        except Exception as e:
            print(f"   ❌ {approach_name} failed: {str(e)}")
            continue
    
    return {"success": False, "error": "No transcript available via YouTube API"}

def try_whisper_transcription(video_url):
    """Try to transcribe video using Whisper"""
    if not SPEECH_TO_TEXT_AVAILABLE:
        return {"success": False, "error": "Speech-to-text libraries not available"}
        
    try:
        # Create temporary directory for audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.mp3")
            
            print("   Downloading audio...")
            if not download_audio(video_url, audio_path):
                return {"success": False, "error": "Failed to download audio"}
            
            # Find the actual audio file (it might have a different extension)
            actual_audio_path = find_audio_file(temp_dir)
            if not actual_audio_path:
                return {"success": False, "error": "Audio file not found after download"}
            
            print(f"   Found audio file: {os.path.basename(actual_audio_path)}")
            print("   Transcribing with Whisper...")
            transcript_text = transcribe_audio(actual_audio_path)
            
            if transcript_text and len(transcript_text.strip()) > 50:
                return {
                    "success": True,
                    "video_id": extract_video_id(video_url),
                    "language": "en",
                    "language_code": "en",
                    "transcript_text": transcript_text,
                    "word_count": len(transcript_text.split()),
                    "character_count": len(transcript_text),
                    "method": "whisper_speech_to_text",
                    "is_shorts": is_shorts_url(video_url),
                    "confidence": "medium"  # Speech-to-text is less accurate than manual transcripts
                }
            else:
                return {"success": False, "error": "Whisper transcription produced insufficient text"}
                
    except Exception as e:
        return {"success": False, "error": f"Whisper transcription failed: {str(e)}"}

def try_metadata_extraction(video_url):
    """Try to extract video metadata as fallback"""
    metadata = get_video_metadata(video_url)
    
    if metadata and len(metadata["text"].strip()) > 20:
        return {
            "success": True,
            "video_id": metadata["video_id"],
            "language": "en",
            "language_code": "en",
            "transcript_text": metadata["text"],
            "word_count": len(metadata["text"].split()),
            "character_count": len(metadata["text"]),
            "method": "metadata_extraction",
            "is_shorts": is_shorts_url(video_url),
            "confidence": "low",
            "warning": "Using video metadata only - limited content available"
        }
    else:
        return {"success": False, "error": "Insufficient metadata available"}

def get_available_languages(video_id_or_url):
    """Get list of available languages for a video"""
    video_id = extract_video_id(video_id_or_url)
    
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        languages = []
        for transcript in transcript_list:
            languages.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
        
        return {
            "success": True,
            "video_id": video_id,
            "available_languages": languages
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id
        }

def main():
    """Main function to run the script"""
    print("Enhanced YouTube Transcript Extractor with Speech-to-Text")
    print("=" * 60)
    
    if not SPEECH_TO_TEXT_AVAILABLE:
        print("⚠️  Speech-to-text libraries not available.")
        print("   Install with: pip install yt-dlp whisper openai-whisper")
        print("   Only YouTube Transcript API will be used.")
        print()
    else:
        print("✅ Speech-to-text libraries available")
        print("💡 For best results, also install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print()
    
    # Get YouTube URL from user
    url = input("Enter YouTube URL or video ID: ").strip()
    
    if not url:
        print("❌ No URL provided")
        return
    
    print(f"\n🔍 Fetching transcript for: {url}")
    
    # Check if it's a Shorts URL
    if is_shorts_url(url):
        print("📱 Detected YouTube Shorts URL")
    
    # Get transcript with fallback methods
    result = get_transcript(url)
    
    if result["success"]:
        print("✅ Transcript retrieved successfully!")
        print(f"📝 Language: {result['language']} ({result['language_code']})")
        print(f"📊 Word count: {result['word_count']}")
        print(f"📏 Character count: {result['character_count']}")
        print(f"🔧 Method used: {result['method']}")
        
        if result.get("confidence"):
            print(f"🎯 Confidence: {result['confidence']}")
        
        if result.get("warning"):
            print(f"⚠️  Warning: {result['warning']}")
        
        print("\n" + "=" * 60)
        print("TRANSCRIPT:")
        print("=" * 60)
        print(result["transcript_text"])
    else:
        print(f"❌ Failed to get transcript: {result['error']}")
        print(f"🔍 Methods tried: {result.get('methods_tried', [])}")
        
        if not SPEECH_TO_TEXT_AVAILABLE:
            print("\n💡 To enable speech-to-text for videos without subtitles:")
            print("   pip install yt-dlp whisper openai-whisper")
        else:
            print("\n💡 If speech-to-text failed, try installing FFmpeg:")
            print("   Windows: Download from https://ffmpeg.org/download.html")
            print("   macOS: brew install ffmpeg")
            print("   Ubuntu/Debian: sudo apt install ffmpeg")

if __name__ == "__main__":
    main() 