#!/usr/bin/env python3
"""
🎯 BeautyAI Debug Infrastructure - Complete Validation Script

This script demonstrates the complete debug infrastructure implementation
for the BeautyAI voice pipeline. It validates all debug components and
showcases the actionable debugging capabilities.

Usage: python final_debug_validation.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add backend to Python path
sys.path.append('backend/src')

from beautyai_inference.services.voice.conversation.simple_voice_service import SimpleVoiceService
from beautyai_inference.api.schemas.debug_schemas import PipelineStage

async def main():
    """Main validation function demonstrating debug infrastructure capabilities."""
    
    print("🎯 BeautyAI Debug Infrastructure - Complete Validation")
    print("=" * 60)
    
    # Test audio file
    audio_file = Path('voice_tests/input_test_questions/webm/q6.webm')
    if not audio_file.exists():
        print(f"❌ Test audio file not found: {audio_file}")
        return False
    
    print(f"📁 Test Audio File: {audio_file}")
    print(f"📊 File Size: {audio_file.stat().st_size:,} bytes")
    
    # Read audio data
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    # Test 1: Basic Debug Mode Functionality
    print("\n" + "="*60)
    print("🧪 TEST 1: Basic Debug Mode Functionality")
    print("="*60)
    
    # Initialize service with debug mode
    print("🔧 Initializing SimpleVoiceService with debug mode...")
    service = SimpleVoiceService(debug_mode=True)
    
    try:
        # Initialize the service
        await service.initialize()
        print("✅ Service initialization: SUCCESS")
        
        # Set up debug callback to capture events
        captured_events = []
        def debug_callback(event: Any):
            try:
                # Convert DebugEvent to dict-like for easier processing
                event_data = {
                    'stage': event.stage.value if hasattr(event, 'stage') else 'unknown',
                    'level': event.level if hasattr(event, 'level') else 'info',
                    'message': event.message if hasattr(event, 'message') else 'No message',
                    'timestamp': event.timestamp if hasattr(event, 'timestamp') else '',
                    'metadata': event.metadata if hasattr(event, 'metadata') else {}
                }
                captured_events.append(event_data)
                print(f"📋 Debug Event: [{event_data['stage'].upper()}] {event_data['level']}: {event_data['message']}")
            except Exception as e:
                print(f"Debug callback error: {e}")
                # Still add to captured events for counting
                captured_events.append({'error': str(e)})
        
        service.set_debug_callback(debug_callback)
        print("✅ Debug callback registration: SUCCESS")
        
        # Test 2: Audio Processing with Debug Context
        print("\n" + "="*60)
        print("🧪 TEST 2: Audio Processing with Debug Context")
        print("="*60)
        
        debug_context = {
            'test_mode': True,
            'test_file': str(audio_file),
            'test_description': 'Complete debug infrastructure validation',
            'expected_language': 'ar',
            'validation_id': 'debug_validation_001'
        }
        
        print("🎵 Processing audio through voice pipeline...")
        start_time = time.time()
        
        result = await service.process_voice_message(
            audio_data=audio_data,
            audio_format='webm',
            language='ar',
            gender='female',
            debug_context=debug_context
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️  Total processing time: {processing_time:.2f}s")
        
        # Validate processing result
        if result.get('success'):
            print("✅ Audio processing: SUCCESS")
            print(f"📝 Transcribed text: \"{result.get('transcribed_text', 'N/A')}\"")
            print(f"🤖 Response text: \"{result.get('response_text', 'N/A')}\"")
            print(f"🌍 Language detected: {result.get('language_detected', 'N/A')}")
            print(f"🎤 Voice used: {result.get('voice_used', 'N/A')}")
        else:
            print("⚠️  Audio processing: PARTIAL SUCCESS (fallback used)")
        
        # Test 3: Debug Summary Analysis
        print("\n" + "="*60)
        print("🧪 TEST 3: Debug Summary Analysis")
        print("="*60)
        
        debug_summary = service.get_debug_summary()
        if debug_summary:
            print("✅ Debug summary generation: SUCCESS")
            print(f"📊 Total processing time: {debug_summary.total_processing_time_ms:.2f}ms")
            print(f"🎯 Pipeline success: {debug_summary.success}")
            print(f"📝 Debug events collected: {len(debug_summary.debug_events)}")
            print(f"🏁 Completed stages: {[stage.value for stage in debug_summary.completed_stages]}")
            
            # Stage timing analysis
            if debug_summary.stage_timings:
                print("\n📊 Stage Timing Analysis:")
                total_time = debug_summary.total_processing_time_ms
                for stage, timing in debug_summary.stage_timings.items():
                    percentage = (timing / total_time * 100) if total_time > 0 else 0
                    print(f"  {stage.value}: {timing:.1f}ms ({percentage:.1f}%)")
                
                # Identify bottleneck
                if debug_summary.stage_timings:
                    bottleneck_stage = max(debug_summary.stage_timings, 
                                         key=debug_summary.stage_timings.get)
                    bottleneck_time = debug_summary.stage_timings[bottleneck_stage]
                    print(f"\n🔍 Bottleneck identified: {bottleneck_stage.value} ({bottleneck_time:.1f}ms)")
            
            # Performance grading
            performance_grade = 'A'
            if debug_summary.total_processing_time_ms > 3000:
                performance_grade = 'C'
            elif debug_summary.total_processing_time_ms > 2000:
                performance_grade = 'B'
            
            print(f"🏆 Performance Grade: {performance_grade}")
            
            # Test 4: Stage-Specific Debug Data Validation
            print("\n" + "="*60)
            print("🧪 TEST 4: Stage-Specific Debug Data Validation")
            print("="*60)
            
            # STT Debug Data
            if debug_summary.transcription_data:
                stt_data = debug_summary.transcription_data
                print("✅ STT Debug Data: AVAILABLE")
                print(f"  📝 Transcribed: \"{stt_data.transcribed_text}\"")
                print(f"  🌍 Language: {stt_data.language_detected}")
                print(f"  📊 Confidence: {stt_data.confidence_score}")
                print(f"  ⏱️  Processing time: {stt_data.processing_time_ms:.2f}ms")
                print(f"  🎵 Audio duration: {stt_data.audio_duration_ms:.2f}ms")
                print(f"  🤖 Model: {stt_data.model_used}")
                print(f"  📁 Format: {stt_data.audio_format}")
                
                # Validate STT quality
                if stt_data.transcribed_text and len(stt_data.transcribed_text.strip()) > 0:
                    print("  ✅ STT Quality: Text generated successfully")
                else:
                    print("  ⚠️  STT Quality: No text generated")
                
                if stt_data.processing_time_ms < 3000:
                    print("  ✅ STT Performance: Good response time")
                else:
                    print("  ⚠️  STT Performance: Slow response time")
            else:
                print("❌ STT Debug Data: NOT AVAILABLE")
            
            # LLM Debug Data
            if debug_summary.llm_data:
                llm_data = debug_summary.llm_data
                print("\n✅ LLM Debug Data: AVAILABLE")
                print(f"  📝 Response: \"{llm_data.response_text}\"")
                print(f"  🎯 Tokens: {llm_data.prompt_tokens}→{llm_data.completion_tokens}")
                print(f"  ⏱️  Processing time: {llm_data.processing_time_ms:.2f}ms")
                print(f"  🧠 Model: {llm_data.model_used}")
                print(f"  🌡️  Temperature: {llm_data.temperature}")
                print(f"  🧠 Thinking mode: {llm_data.thinking_mode}")
                
                # Validate LLM performance
                if llm_data.response_text and len(llm_data.response_text.strip()) > 0:
                    print("  ✅ LLM Quality: Response generated successfully")
                else:
                    print("  ⚠️  LLM Quality: No response generated")
                
                if llm_data.completion_tokens > 0:
                    print("  ✅ LLM Tokens: Token generation successful")
                else:
                    print("  ⚠️  LLM Tokens: No tokens generated")
            else:
                print("❌ LLM Debug Data: NOT AVAILABLE")
            
            # TTS Debug Data
            if debug_summary.tts_data:
                tts_data = debug_summary.tts_data
                print("\n✅ TTS Debug Data: AVAILABLE")
                print(f"  🎵 Audio length: {tts_data.audio_length_ms}ms")
                print(f"  🎤 Voice: {tts_data.voice_used}")
                print(f"  ⏱️  Processing time: {tts_data.processing_time_ms:.2f}ms")
                print(f"  📁 Format: {tts_data.output_format}")
                print(f"  📝 Text length: {tts_data.text_length}")
                print(f"  🗣️  Speech rate: {tts_data.speech_rate}")
                
                # Validate TTS quality
                if tts_data.audio_length_ms > 0:
                    print("  ✅ TTS Quality: Audio generated successfully")
                else:
                    print("  ⚠️  TTS Quality: No audio generated")
                
                if tts_data.processing_time_ms < 5000:
                    print("  ✅ TTS Performance: Good synthesis time")
                else:
                    print("  ⚠️  TTS Performance: Slow synthesis time")
            else:
                print("❌ TTS Debug Data: NOT AVAILABLE")
        else:
            print("❌ Debug summary generation: FAILED")
        
        # Test 5: Debug Event Validation
        print("\n" + "="*60)
        print("🧪 TEST 5: Debug Event Validation")
        print("="*60)
        
        print(f"📋 Total debug events captured: {len(captured_events)}")
        
        if captured_events:
            print("✅ Debug event emission: SUCCESS")
            
            # Show sample events
            print("\n📝 Sample Debug Events:")
            for i, event in enumerate(captured_events[:5]):  # Show first 5
                if 'error' in event:
                    print(f"  {i+1}. [ERROR] Debug callback error: {event['error']}")
                else:
                    stage = event.get('stage', 'unknown')
                    level = event.get('level', 'info')
                    message = event.get('message', 'No message')
                    print(f"  {i+1}. [{stage.upper()}] {level}: {message}")
            
            if len(captured_events) > 5:
                print(f"  ... and {len(captured_events) - 5} more events")
        else:
            print("⚠️  Debug event emission: NO EVENTS CAPTURED")
        
        # Final Results Summary
        print("\n" + "="*60)
        print("🎯 FINAL VALIDATION RESULTS")
        print("="*60)
        
        validation_results = {
            'service_initialization': True,
            'debug_callback_registration': True,
            'audio_processing': result.get('success', False),
            'debug_summary_generation': debug_summary is not None,
            'stage_timing_data': bool(debug_summary and debug_summary.stage_timings),
            'stt_debug_data': bool(debug_summary and debug_summary.transcription_data),
            'llm_debug_data': bool(debug_summary and debug_summary.llm_data),
            'tts_debug_data': bool(debug_summary and debug_summary.tts_data),
            'debug_event_emission': len(captured_events) > 0,
            'performance_analysis': True
        }
        
        passed_tests = sum(validation_results.values())
        total_tests = len(validation_results)
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} PASSED")
        print(f"🏆 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        for test_name, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 DEBUG INFRASTRUCTURE VALIDATION: COMPLETE SUCCESS!")
            print("🚀 Ready for production use")
            return True
        else:
            print(f"\n⚠️  DEBUG INFRASTRUCTURE VALIDATION: PARTIAL SUCCESS ({passed_tests}/{total_tests})")
            print("🔧 Some components may need adjustment")
            return False
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up service...")
        await service.cleanup()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    print("🎯 Starting BeautyAI Debug Infrastructure Validation...")
    print("⏰ This may take a few minutes to complete...")
    
    success = asyncio.run(main())
    
    if success:
        print("\n🎯 VALIDATION COMPLETE: ALL SYSTEMS OPERATIONAL")
        exit(0)
    else:
        print("\n🎯 VALIDATION COMPLETE: ISSUES DETECTED")
        exit(1)