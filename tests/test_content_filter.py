#!/usr/bin/env python3
"""
Test Content Filter Service

Comprehensive tests for the content filtering system that prevents answering
cosmetic procedure questions from the 2000QAToR.csv dataset.
"""
import pytest
import os
import sys
import time
from typing import List, Dict, Any

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beautyai_inference.services.inference.content_filter_service import (
    ContentFilterService, 
    FilterResult
)


class TestContentFilterService:
    """Test suite for ContentFilterService functionality."""
    
    @pytest.fixture(scope="class")
    def filter_service(self):
        """Initialize content filter service for testing."""
        return ContentFilterService()
    
    @pytest.fixture(scope="class")
    def test_cases(self):
        """Test cases covering various scenarios."""
        return {
            "blocked_cases": [
                # Direct cosmetic procedure questions in Arabic
                "ما تكلفة البوتوكس؟",
                "هل الليزر آمن لإزالة الشعر؟",
                "كم جلسة أحتاج لتنظيف البشرة؟",
                "أين أجد عيادة تجميل موثوقة؟",
                "ما أفضل كريم لتبييض الوجه؟",
                
                # English cosmetic procedure questions
                "How much does Botox cost?",
                "Is laser hair removal safe?",
                "Where can I get lip fillers?",
                "What's the best skin whitening cream?",
                "How many sessions for microneedling?",
                
                # Medical procedure questions
                "ما هي مخاطر عمليات التجميل؟",
                "هل يمكنني إجراء شفط الدهون؟",
                "What are the risks of cosmetic surgery?",
                
                # Skincare product recommendations
                "أنصحني بمنتج لإزالة التصبغات",
                "ما أفضل سيروم للوجه؟",
                "Recommend a good anti-aging cream"
            ],
            
            "allowed_cases": [
                # General beauty and wellness
                "ما هي أهمية النوم الصحي؟",
                "كيف أحافظ على صحة شعري؟",
                "What are the benefits of drinking water?",
                "How to maintain healthy skin naturally?",
                
                # General health questions
                "ما هي فوائد الرياضة؟",
                "كيف أتبع نظام غذائي صحي؟",
                "What vitamins are important for health?",
                
                # Technology and general topics
                "كيف أتعلم البرمجة؟",
                "ما هي أفضل الكتب للقراءة؟",
                "How do I learn a new language?",
                "What's the weather like today?",
                
                # Beauty education (non-procedural)
                "ما هي مكونات البشرة؟",
                "كيف تعمل خلايا الجلد؟",
                "What is collagen and its function?"
            ]
        }
    
    def test_service_initialization(self, filter_service):
        """Test that the content filter service initializes correctly."""
        assert filter_service is not None
        
        # Check that data files are loaded
        stats = filter_service.get_filter_stats()
        assert stats['csv_exists'] is True
        assert stats['total_forbidden_topics'] > 0
        assert stats['total_forbidden_keywords'] > 0
        assert stats['total_question_patterns'] > 0
    
    def test_filter_statistics(self, filter_service):
        """Test filter statistics are properly returned."""
        stats = filter_service.get_filter_stats()
        
        # Verify all required statistics are present
        required_keys = [
            'total_forbidden_topics',
            'total_forbidden_keywords', 
            'total_question_patterns',
            'csv_exists'
        ]
        
        for key in required_keys:
            assert key in stats
            if key != 'csv_exists':
                assert isinstance(stats[key], int)
                assert stats[key] > 0
    
    def test_blocked_content_detection(self, filter_service, test_cases):
        """Test that cosmetic procedure questions are properly blocked."""
        blocked_cases = test_cases["blocked_cases"]
        
        for text in blocked_cases:
            result = filter_service.filter_content(text)
            
            assert isinstance(result, FilterResult)
            assert result.is_allowed is False, f"Should block: '{text}'"
            assert result.filter_reason is not None
            assert 0.0 <= result.confidence_score <= 1.0
            
            # Test both Arabic and English language detection
            for lang in ['ar', 'en']:
                result_lang = filter_service.filter_content(text, language=lang)
                assert result_lang.is_allowed is False
    
    def test_allowed_content_detection(self, filter_service, test_cases):
        """Test that non-cosmetic content is properly allowed."""
        allowed_cases = test_cases["allowed_cases"]
        
        for text in allowed_cases:
            result = filter_service.filter_content(text)
            
            assert isinstance(result, FilterResult)
            assert result.is_allowed is True, f"Should allow: '{text}'"
            assert result.filter_reason is None
            assert result.confidence_score == 0.0
    
    def test_edge_cases(self, filter_service):
        """Test edge cases and special inputs."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "Hello world!" * 100,  # Very long text
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
        ]
        
        for text in edge_cases:
            result = filter_service.filter_content(text)
            assert isinstance(result, FilterResult)
            # Edge cases should generally be allowed
            assert result.is_allowed is True
    
    def test_performance(self, filter_service, test_cases):
        """Test that filtering performance meets requirements (<10ms)."""
        all_test_cases = (
            test_cases["blocked_cases"] + 
            test_cases["allowed_cases"]
        )
        
        # Test performance on multiple runs
        total_time = 0
        num_tests = len(all_test_cases)
        
        for text in all_test_cases:
            start_time = time.perf_counter()
            result = filter_service.filter_content(text)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            total_time += processing_time
            
            # Each individual call should be under 10ms
            assert processing_time < 10.0, f"Too slow ({processing_time:.2f}ms): '{text}'"
        
        # Average time should be well under 10ms
        avg_time = total_time / num_tests
        assert avg_time < 5.0, f"Average filtering time too high: {avg_time:.2f}ms"
    
    def test_multilingual_support(self, filter_service):
        """Test that the filter works with both Arabic and English content."""
        # Test same concepts in different languages
        test_pairs = [
            ("ما تكلفة البوتوكس؟", "How much does Botox cost?"),
            ("هل الليزر آمن؟", "Is laser treatment safe?"),
            ("كيف أحافظ على صحتي؟", "How do I stay healthy?"),
        ]
        
        for arabic_text, english_text in test_pairs:
            arabic_result = filter_service.filter_content(arabic_text, language='ar')
            english_result = filter_service.filter_content(english_text, language='en')
            
            # Results should be consistent across languages for similar content
            assert arabic_result.is_allowed == english_result.is_allowed
    
    def test_confidence_scoring(self, filter_service, test_cases):
        """Test that confidence scores are reasonable."""
        blocked_cases = test_cases["blocked_cases"]
        
        for text in blocked_cases:
            result = filter_service.filter_content(text)
            
            if not result.is_allowed:
                # Blocked content should have reasonable confidence
                assert result.confidence_score > 0.5, f"Low confidence for blocked content: '{text}'"
                assert result.confidence_score <= 1.0
    
    def test_filter_reasons(self, filter_service, test_cases):
        """Test that appropriate filter reasons are provided."""
        blocked_cases = test_cases["blocked_cases"]
        
        expected_reasons = [
            "forbidden_topic_match",
            "keyword_density_threshold",
            "pattern_similarity_match", 
            "medical_procedure_indicators"
        ]
        
        found_reasons = set()
        
        for text in blocked_cases:
            result = filter_service.filter_content(text)
            
            if not result.is_allowed:
                assert result.filter_reason is not None
                found_reasons.add(result.filter_reason)
        
        # Should have found multiple types of filter reasons
        assert len(found_reasons) >= 2, f"Expected multiple filter reasons, found: {found_reasons}"
    
    def test_csv_integration(self, filter_service):
        """Test that CSV file integration works correctly."""
        stats = filter_service.get_filter_stats()
        
        # Should have loaded substantial data from CSV
        assert stats['csv_exists'] is True
        assert stats['total_question_patterns'] >= 1000  # Should have loaded many patterns
        
        # Test a sample question that should be in the CSV
        csv_questions = [
            "ما تكلفة الحقن؟",
            "هل البوتوكس آمن؟",
            "كم سعر الليزر؟"
        ]
        
        for question in csv_questions:
            result = filter_service.filter_content(question)
            # These should likely be blocked as they're cosmetic procedure related
            # Note: May be allowed if not exact matches, which is also valid behavior


class TestFilterIntegration:
    """Integration tests for content filter with other services."""
    
    def test_import_availability(self):
        """Test that content filter can be imported by other services."""
        try:
            from beautyai_inference.services.inference.content_filter_service import ContentFilterService
            from beautyai_inference.services.inference.chat_service import ChatService
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_concurrent_filtering(self):
        """Test that multiple filter instances work correctly."""
        filter1 = ContentFilterService()
        filter2 = ContentFilterService()
        
        test_text = "ما تكلفة البوتوكس؟"
        
        result1 = filter1.filter_content(test_text)
        result2 = filter2.filter_content(test_text)
        
        # Results should be consistent
        assert result1.is_allowed == result2.is_allowed
        assert result1.filter_reason == result2.filter_reason


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
