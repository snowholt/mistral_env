#!/usr/bin/env python3
"""
OpenAI API Validation Test Script

Purpose:
- Test OpenAI API key validity
- Transcribe voice sample using OpenAI Whisper API
- Generate response using GPT-4o-mini
- Compare quality with local implementation

Usage:
    python test_openai_validation.py

Author: BeautyAI Framework
Date: 2025-10-06
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install openai python-dotenv")
    sys.exit(1)


class OpenAIValidator:
    """Validates OpenAI API functionality for voice processing."""
    
    def __init__(self, env_path: Optional[Path] = None, cache_dir: Optional[Path] = None):
        """
        Initialize validator with API credentials.
        
        Args:
            env_path: Path to .env file (defaults to script directory)
            cache_dir: Directory for caching results (defaults to script directory)
        """
        # Load environment variables
        if env_path is None:
            env_path = Path(__file__).parent / ".env"
        
        load_dotenv(env_path)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.results = {}
        
        # Cache directory for storing results
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def test_api_connection(self) -> bool:
        """
        Test basic API connectivity.
        
        Returns:
            True if connection successful
        """
        print("\n🔍 Testing API Connection...")
        try:
            # Simple models list test
            models = self.client.models.list()
            print(f"✅ API Connection Successful")
            print(f"   Available models: {len(list(models.data))} found")
            return True
        except Exception as e:
            print(f"❌ API Connection Failed: {e}")
            return False
    
    def _get_cache_path(self, audio_path: Path, cache_type: str) -> Path:
        """Get cache file path for audio file."""
        audio_hash = audio_path.name.replace('.', '_')
        return self.cache_dir / f"{audio_hash}_{cache_type}.json"
    
    def transcribe_audio(self, audio_path: Path, language: str = "ar", use_cache: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper API with caching.
        
        Args:
            audio_path: Path to audio file (WebM, WAV, MP3, etc.)
            language: Language code (ar, en, etc.)
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary with transcription results and metrics
        """
        print(f"\n🎤 Transcribing Audio: {audio_path.name}")
        print(f"   File size: {audio_path.stat().st_size / 1024:.2f} KB")
        print(f"   Language: {language}")
        
        # Check cache first
        cache_path = self._get_cache_path(audio_path, "transcription")
        if use_cache and cache_path.exists():
            print(f"   ♻️  Using cached transcription")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
                self.results['transcription'] = cached_result
                return cached_result
        
        try:
            start_time = time.time()
            
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json"
                )
            
            elapsed_time = time.time() - start_time
            
            # Convert segments to serializable format
            segments = getattr(transcription, 'segments', None)
            if segments:
                segments = [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    }
                    for seg in segments
                ]
            
            result = {
                "success": True,
                "text": transcription.text,
                "language": transcription.language,
                "duration": getattr(transcription, 'duration', None),
                "processing_time_sec": elapsed_time,
                "segments": segments
            }
            
            print(f"✅ Transcription Successful ({elapsed_time:.2f}s)")
            print(f"   Detected Language: {result['language']}")
            print(f"   Transcribed Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
            
            if result['duration']:
                print(f"   Audio Duration: {result['duration']:.2f}s")
            
            # Cache the result
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.results['transcription'] = result
            return result
            
        except Exception as e:
            print(f"❌ Transcription Failed: {e}")
            result = {
                "success": False,
                "error": str(e)
            }
            self.results['transcription'] = result
            return result
    
    def generate_response(self, transcription: str, language: str = "ar", use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate AI response using GPT-4o-mini with caching.
        
        Args:
            transcription: User's transcribed text
            language: Language for response
            use_cache: Whether to use cached results if available
            
        Returns:
            Dictionary with response and metrics
        """
        print(f"\n🤖 Generating Response with GPT-4o-mini...")
        
        # Check cache first
        cache_key = f"{transcription[:50]}_{language}"
        cache_path = self.cache_dir / f"response_{hash(cache_key) & 0xFFFFFFFF:08x}.json"
        if use_cache and cache_path.exists():
            print(f"   ♻️  Using cached response")
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
                self.results['response'] = cached_result
                return cached_result
        
        # Language-specific system prompts
        system_prompts = {
            "ar": "أنت مساعد ذكي ومفيد. أجب على الأسئلة بوضوح وبإيجاز.",
            "en": "You are a helpful and intelligent assistant. Answer questions clearly and concisely."
        }
        
        system_prompt = system_prompts.get(language, system_prompts["en"])
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            elapsed_time = time.time() - start_time
            
            result = {
                "success": True,
                "response_text": response.choices[0].message.content,
                "model": response.model,
                "processing_time_sec": elapsed_time,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
            print(f"✅ Response Generated ({elapsed_time:.2f}s)")
            print(f"   Model: {result['model']}")
            print(f"   Tokens: {result['tokens']['total']} (prompt: {result['tokens']['prompt']}, completion: {result['tokens']['completion']})")
            print(f"   Response: {result['response_text'][:150]}{'...' if len(result['response_text']) > 150 else ''}")
            
            # Cache the result
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.results['response'] = result
            return result
            
        except Exception as e:
            print(f"❌ Response Generation Failed: {e}")
            result = {
                "success": False,
                "error": str(e)
            }
            self.results['response'] = result
            return result
    
    def run_full_pipeline(self, audio_path: Path, language: str = "ar") -> Dict[str, Any]:
        """
        Run complete validation pipeline.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Complete results dictionary
        """
        print("\n" + "=" * 70)
        print("🚀 OpenAI API Validation Pipeline")
        print("=" * 70)
        
        pipeline_start = time.time()
        
        # Step 1: Test connection
        if not self.test_api_connection():
            return {"success": False, "error": "API connection failed"}
        
        # Step 2: Transcribe audio
        transcription_result = self.transcribe_audio(audio_path, language)
        if not transcription_result.get("success"):
            return {"success": False, "error": "Transcription failed", "results": self.results}
        
        # Step 3: Generate response
        response_result = self.generate_response(transcription_result["text"], language)
        
        total_time = time.time() - pipeline_start
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        print(f"✅ API Connection: PASSED")
        print(f"✅ Transcription: {'PASSED' if transcription_result['success'] else 'FAILED'}")
        print(f"✅ Response Generation: {'PASSED' if response_result['success'] else 'FAILED'}")
        print(f"\n⏱️  Total Pipeline Time: {total_time:.2f}s")
        
        if transcription_result['success']:
            print(f"   - Transcription: {transcription_result['processing_time_sec']:.2f}s")
        if response_result.get('success'):
            print(f"   - Response Generation: {response_result['processing_time_sec']:.2f}s")
        
        print("\n📝 Full Transcription:")
        print(f"   {transcription_result.get('text', 'N/A')}")
        
        if response_result.get('success'):
            print("\n💬 AI Response:")
            print(f"   {response_result['response_text']}")
        
        print("=" * 70)
        
        self.results['pipeline_total_time'] = total_time
        self.results['success'] = transcription_result['success'] and response_result.get('success', False)
        
        return self.results
    
    def save_results(self, output_path: Optional[Path] = None):
        """Save validation results to JSON file."""
        if output_path is None:
            output_path = Path(__file__).parent / "validation_results.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main execution function."""
    # Configuration - use local copy in test directory
    audio_file = Path(__file__).parent / "q7.webm"
    language = "ar"  # Arabic language
    
    # Validate audio file exists
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    try:
        # Run validation
        validator = OpenAIValidator()
        results = validator.run_full_pipeline(audio_file, language)
        
        # Save results
        validator.save_results()
        
        # Exit with appropriate code
        sys.exit(0 if results.get('success') else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
