#!/usr/bin/env python3
"""
Comprehensive Voice-to-Voice Pipeline Test
==========================================

Test the complete voice-to-voice pipeline:
1. Audio Input → STT (Speech-to-Text)
2. STT → LLM (Chat response)
3. LLM → TTS (Text-to-Speech)
4. Complete pipeline integration

This will verify that all components work together before testing the WebSocket endpoint.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import BeautyAI services
import sys
sys.path.append('/home/lumi/beautyai')

from beautyai_inference.services.voice_to_voice_service import VoiceToVoiceService
from beautyai_inference.services.audio_transcription_service import AudioTranscriptionService
from beautyai_inference.services.text_to_speech_service import TextToSpeechService
from beautyai_inference.services.inference.chat_service import ChatService


class VoiceToVoicePipelineTest:
    """Comprehensive voice-to-voice pipeline test."""
    
    def __init__(self):
        self.test_dir = Path("/home/lumi/beautyai/tests/websocket")
        self.output_dir = self.test_dir / "voice_to_voice_test_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Test audio files
        self.test_files = [
            "/home/lumi/beautyai/voice_tests/input_test_questions/botox.wav",
            "/home/lumi/beautyai/voice_tests/input_test_questions/botox_ar.wav"
        ]
        
        # Services
        self.v2v_service = None
        self.stt_service = None
        self.tts_service = None
        self.chat_service = None
        
        self.test_results = []
    
    async def setup_services(self):
        """Initialize all services."""
        logger.info("🚀 Setting up Voice-to-Voice Pipeline Services")
        logger.info("=" * 80)
        
        try:
            # Initialize voice-to-voice service with content filtering disabled
            logger.info("📥 Initializing Voice-to-Voice Service...")
            self.v2v_service = VoiceToVoiceService(content_filter_strictness="disabled")
            
            # Initialize individual services for component testing
            logger.info("📥 Initializing individual services...")
            self.stt_service = AudioTranscriptionService()
            self.tts_service = TextToSpeechService()
            self.chat_service = ChatService()
            
            logger.info("✅ All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            return False
    
    async def test_individual_components(self):
        """Test each component individually."""
        logger.info("\n🔍 Testing Individual Components")
        logger.info("=" * 50)
        
        component_results = {}
        
        # Test STT
        logger.info("🎤 Testing Speech-to-Text...")
        try:
            # Load STT model
            stt_result = self.stt_service.load_whisper_model("whisper-large-v3-turbo-arabic")
            if stt_result:  # Boolean return
                logger.info("✅ STT model loaded successfully")
                
                # Test transcription with our audio files
                for audio_file in self.test_files:
                    if os.path.exists(audio_file):
                        logger.info(f"📝 Testing transcription: {Path(audio_file).name}")
                        transcription = self.stt_service.transcribe_audio(audio_file)
                        if transcription and transcription.get("text"):
                            logger.info(f"✅ Transcription: {transcription['text'][:100]}...")
                            component_results["stt"] = True
                        else:
                            logger.warning(f"⚠️ Empty transcription for {audio_file}")
                    else:
                        logger.warning(f"⚠️ Audio file not found: {audio_file}")
            else:
                logger.error("❌ STT model failed to load")
                component_results["stt"] = False
                
        except Exception as e:
            logger.error(f"❌ STT test failed: {e}")
            component_results["stt"] = False
        
        # Test TTS
        logger.info("\n🔊 Testing Text-to-Speech...")
        try:
            # Load TTS model
            tts_result = self.tts_service.load_tts_model("coqui-tts-arabic")
            if tts_result:  # Boolean return
                logger.info("✅ TTS model loaded successfully")
                
                # Test generation
                test_text_ar = "مرحباً، هذا اختبار لنظام التحويل من النص إلى الصوت."
                test_text_en = "Hello, this is a test of our text-to-speech system."
                
                for text, lang in [(test_text_ar, "ar"), (test_text_en, "en")]:
                    logger.info(f"🎯 Testing {lang}: {text[:50]}...")
                    audio_bytes = self.tts_service.text_to_speech(
                        text=text,
                        language=lang,
                        voice="female"
                    )
                    if audio_bytes:
                        output_file = self.output_dir / f"tts_test_{lang}.wav"
                        with open(output_file, "wb") as f:
                            f.write(audio_bytes)
                        logger.info(f"✅ TTS generated: {output_file.name} ({len(audio_bytes)} bytes)")
                        component_results["tts"] = True
                    else:
                        logger.warning(f"⚠️ Empty TTS output for {lang}")
            else:
                logger.error("❌ TTS model failed to load")
                component_results["tts"] = False
                
        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")
            component_results["tts"] = False
        
        # Test Chat/LLM
        logger.info("\n💬 Testing Chat/LLM...")
        try:
            # Load chat model
            chat_result = await self.chat_service.load_model("qwen3-unsloth-q4ks")
            if chat_result.get("success", False):
                logger.info("✅ Chat model loaded successfully")
                
                # Test response generation
                test_prompts = [
                    "ما هو البوتوكس؟",
                    "What is botox treatment?",
                    "كم تكلفة علاج البوتوكس؟"
                ]
                
                for prompt in test_prompts:
                    logger.info(f"💭 Testing prompt: {prompt}")
                    response = await self.chat_service.generate_response(
                        message=prompt,
                        max_length=150,
                        thinking_mode=False
                    )
                    if response and response.get("response"):
                        logger.info(f"✅ Response: {response['response'][:100]}...")
                        component_results["chat"] = True
                    else:
                        logger.warning(f"⚠️ Empty response for: {prompt}")
            else:
                logger.error("❌ Chat model failed to load")
                component_results["chat"] = False
                
        except Exception as e:
            logger.error(f"❌ Chat test failed: {e}")
            component_results["chat"] = False
        
        return component_results
    
    async def test_voice_to_voice_pipeline(self):
        """Test the complete voice-to-voice pipeline."""
        logger.info("\n🎭 Testing Complete Voice-to-Voice Pipeline")
        logger.info("=" * 60)
        
        pipeline_results = []
        
        # Test each audio file
        for audio_file in self.test_files:
            if not os.path.exists(audio_file):
                logger.warning(f"⚠️ Audio file not found: {audio_file}")
                continue
            
            file_name = Path(audio_file).name
            logger.info(f"\n🎯 Testing pipeline with: {file_name}")
            logger.info("-" * 40)
            
            start_time = time.time()
            
            try:
                # Process through complete pipeline
                logger.info("🔄 Processing through voice-to-voice pipeline...")
                result = await self.v2v_service.voice_to_voice_file(
                    audio_file=audio_file,
                    language="auto",  # Auto-detect language
                    speaker_voice="female",
                    response_max_length=200,
                    enable_content_filter=False  # Disabled globally
                )
                
                total_time = time.time() - start_time
                
                if result.get("success", False):
                    logger.info(f"✅ Pipeline successful in {total_time:.2f}s")
                    
                    # Log details
                    transcription = result.get("transcription", "")
                    response_text = result.get("response_text", "")
                    output_audio = result.get("output_audio_path", "")
                    
                    logger.info(f"📝 Transcription: {transcription[:100]}...")
                    logger.info(f"💬 Response: {response_text[:100]}...")
                    logger.info(f"🔊 Audio output: {output_audio}")
                    
                    # Copy output to our test directory
                    if output_audio and os.path.exists(output_audio):
                        import shutil
                        test_output = self.output_dir / f"pipeline_{file_name}"
                        shutil.copy2(output_audio, test_output)
                        logger.info(f"📁 Copied to: {test_output}")
                    
                    pipeline_results.append({
                        "file": file_name,
                        "success": True,
                        "time": total_time,
                        "transcription": transcription,
                        "response": response_text,
                        "output_audio": str(test_output) if output_audio else None
                    })
                    
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"❌ Pipeline failed: {error_msg}")
                    pipeline_results.append({
                        "file": file_name,
                        "success": False,
                        "error": error_msg,
                        "time": total_time
                    })
                
            except Exception as e:
                total_time = time.time() - start_time
                logger.error(f"❌ Pipeline exception: {e}")
                pipeline_results.append({
                    "file": file_name,
                    "success": False,
                    "error": str(e),
                    "time": total_time
                })
        
        return pipeline_results
    
    async def test_different_configurations(self):
        """Test different voice-to-voice configurations."""
        logger.info("\n⚙️ Testing Different Configurations")
        logger.info("=" * 50)
        
        config_results = []
        
        # Test configurations
        test_configs = [
            {"language": "ar", "voice": "female", "max_length": 150},
            {"language": "ar", "voice": "male", "max_length": 150},
            {"language": "en", "voice": "female", "max_length": 150},
            {"language": "auto", "voice": "female", "max_length": 200},
        ]
        
        # Use first available audio file
        test_audio = None
        for audio_file in self.test_files:
            if os.path.exists(audio_file):
                test_audio = audio_file
                break
        
        if not test_audio:
            logger.warning("⚠️ No test audio files available for configuration testing")
            return config_results
        
        logger.info(f"📁 Using audio file: {Path(test_audio).name}")
        
        for i, config in enumerate(test_configs, 1):
            logger.info(f"\n🔧 Config {i}: {config}")
            
            try:
                start_time = time.time()
                result = await self.v2v_service.voice_to_voice_file(
                    audio_file=test_audio,
                    language=config["language"],
                    speaker_voice=config["voice"],
                    response_max_length=config["max_length"],
                    enable_content_filter=False
                )
                
                process_time = time.time() - start_time
                
                if result.get("success", False):
                    logger.info(f"✅ Config {i} successful in {process_time:.2f}s")
                    config_results.append({
                        "config": config,
                        "success": True,
                        "time": process_time,
                        "transcription": result.get("transcription", "")[:50] + "...",
                        "response": result.get("response_text", "")[:50] + "..."
                    })
                else:
                    logger.error(f"❌ Config {i} failed: {result.get('error', 'Unknown')}")
                    config_results.append({
                        "config": config,
                        "success": False,
                        "error": result.get("error", "Unknown"),
                        "time": process_time
                    })
                    
            except Exception as e:
                logger.error(f"❌ Config {i} exception: {e}")
                config_results.append({
                    "config": config,
                    "success": False,
                    "error": str(e),
                    "time": 0
                })
        
        return config_results
    
    def generate_test_report(self, component_results, pipeline_results, config_results):
        """Generate comprehensive test report."""
        logger.info("\n📊 VOICE-TO-VOICE PIPELINE TEST REPORT")
        logger.info("=" * 80)
        
        # Component test summary
        logger.info("🔍 Component Test Results:")
        for component, success in component_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"   {component.upper()}: {status}")
        
        # Pipeline test summary
        logger.info("\n🎭 Pipeline Test Results:")
        successful_pipelines = sum(1 for r in pipeline_results if r.get("success", False))
        total_pipelines = len(pipeline_results)
        logger.info(f"   Success Rate: {successful_pipelines}/{total_pipelines} ({successful_pipelines/total_pipelines*100:.1f}%)")
        
        for result in pipeline_results:
            status = "✅" if result.get("success", False) else "❌"
            time_info = f" ({result.get('time', 0):.2f}s)" if result.get("time") else ""
            logger.info(f"   {status} {result['file']}{time_info}")
        
        # Configuration test summary
        logger.info("\n⚙️ Configuration Test Results:")
        successful_configs = sum(1 for r in config_results if r.get("success", False))
        total_configs = len(config_results)
        if total_configs > 0:
            logger.info(f"   Success Rate: {successful_configs}/{total_configs} ({successful_configs/total_configs*100:.1f}%)")
            
            for result in config_results:
                status = "✅" if result.get("success", False) else "❌"
                config_str = str(result['config'])
                time_info = f" ({result.get('time', 0):.2f}s)" if result.get("time") else ""
                logger.info(f"   {status} {config_str}{time_info}")
        
        # Output directory
        logger.info(f"\n📁 Test outputs saved to: {self.output_dir}")
        
        # Overall status
        all_components_pass = all(component_results.values())
        any_pipeline_pass = any(r.get("success", False) for r in pipeline_results)
        
        if all_components_pass and any_pipeline_pass:
            logger.info("\n🎉 OVERALL STATUS: ✅ PIPELINE IS WORKING!")
            logger.info("📋 Next steps:")
            logger.info("   1. ✅ Voice-to-voice pipeline is functional")
            logger.info("   2. Test WebSocket endpoints")
            logger.info("   3. Test with different audio formats")
            logger.info("   4. Performance optimization if needed")
        else:
            logger.info("\n⚠️ OVERALL STATUS: ❌ ISSUES DETECTED")
            logger.info("📋 Issues to resolve:")
            
            if not all_components_pass:
                failed_components = [c for c, success in component_results.items() if not success]
                logger.info(f"   - Fix component issues: {', '.join(failed_components)}")
            
            if not any_pipeline_pass:
                logger.info("   - Resolve pipeline integration issues")
    
    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🚀 Starting Comprehensive Voice-to-Voice Pipeline Test")
        logger.info("=" * 80)
        
        # Setup
        setup_success = await self.setup_services()
        if not setup_success:
            logger.error("❌ Setup failed - aborting tests")
            return
        
        # Run tests
        component_results = await self.test_individual_components()
        pipeline_results = await self.test_voice_to_voice_pipeline()
        config_results = await self.test_different_configurations()
        
        # Generate report
        self.generate_test_report(component_results, pipeline_results, config_results)
        
        logger.info("\n✅ Voice-to-Voice Pipeline Testing Complete!")


async def main():
    """Main test function."""
    test = VoiceToVoicePipelineTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
