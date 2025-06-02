#!/usr/bin/env python3
"""
Content Filter Demo Script

This script demonstrates how the content filter prevents the model from answering
questions about cosmetic procedures listed in the 2000QAToR.csv file.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beautyai_inference.services.inference.content_filter_service import ContentFilterService


def test_content_filter():
    """Test the content filter with various inputs."""
    
    print("🔒 BeautyAI Content Filter Demo")
    print("=" * 60)
    
    # Initialize the content filter service
    filter_service = ContentFilterService()
    
    # Get filter statistics
    stats = filter_service.get_filter_stats()
    print(f"📊 Filter Statistics:")
    print(f"   - Forbidden Topics: {stats['total_forbidden_topics']}")
    print(f"   - Forbidden Keywords: {stats['total_forbidden_keywords']}")
    print(f"   - Question Patterns: {stats['total_question_patterns']}")
    print(f"   - CSV File Exists: {stats['csv_exists']}")
    print()
    
    # Test cases
    test_cases = [
        # Should be BLOCKED (cosmetic procedure questions)
        {
            "input": "ما تكلفة البوتوكس؟",
            "expected": "blocked",
            "description": "Direct question about Botox cost"
        },
        {
            "input": "هل الليزر آمن لإزالة الشعر؟",
            "expected": "blocked", 
            "description": "Laser treatment safety question"
        },
        {
            "input": "كم جلسة أحتاج لتنظيف البشرة؟",
            "expected": "blocked",
            "description": "Skincare session question"
        },
        {
            "input": "ما أضرار شد الوجه؟",
            "expected": "blocked",
            "description": "Face lift side effects question"
        },
        {
            "input": "متى تظهر نتائج التقشير الكيميائي؟",
            "expected": "blocked",
            "description": "Chemical peeling results timing"
        },
        
        # Should be ALLOWED (general questions)
        {
            "input": "ما هو الطقس اليوم؟",
            "expected": "allowed",
            "description": "Weather question"
        },
        {
            "input": "كيف أتعلم البرمجة؟",
            "expected": "allowed",
            "description": "Programming learning question"
        },
        {
            "input": "ما هي عاصمة فرنسا؟",
            "expected": "allowed",
            "description": "Geography question"
        },
        {
            "input": "أخبرني قصة مضحكة",
            "expected": "allowed", 
            "description": "Entertainment request"
        },
        {
            "input": "ما هي فوائد القراءة؟",
            "expected": "allowed",
            "description": "General knowledge question"
        }
    ]
    
    print("🧪 Testing Content Filter:")
    print("-" * 60)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        user_input = test_case["input"]
        expected = test_case["expected"]
        description = test_case["description"]
        
        # Test the filter
        result = filter_service.filter_content(user_input, language='ar')
        
        # Determine actual result
        actual = "blocked" if not result.is_allowed else "allowed"
        
        # Check if prediction is correct
        is_correct = actual == expected
        if is_correct:
            correct_predictions += 1
        
        # Display result
        status_icon = "✅" if is_correct else "❌"
        action_icon = "🚫" if actual == "blocked" else "✅"
        
        print(f"{status_icon} Test {i:2d}: {action_icon} {actual.upper():<7} | {description}")
        print(f"          Input: {user_input}")
        
        if actual == "blocked":
            print(f"          Reason: {result.filter_reason}")
            print(f"          Confidence: {result.confidence_score:.2f}")
            if result.matched_patterns:
                print(f"          Matched: {result.matched_patterns[:3]}")  # Show first 3 matches
        
        print()
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 60)
    print(f"📈 Results Summary:")
    print(f"   - Total Tests: {total_tests}")
    print(f"   - Correct Predictions: {correct_predictions}")
    print(f"   - Accuracy: {accuracy:.1f}%")
    print()
    
    if accuracy >= 90:
        print("🎉 Excellent! The content filter is working very well.")
    elif accuracy >= 80:
        print("👍 Good! The content filter is working well.")
    elif accuracy >= 70:
        print("⚠️  Fair. The content filter may need some tuning.")
    else:
        print("🔧 The content filter needs significant improvement.")


def interactive_test():
    """Interactive testing mode."""
    
    print("\n🔄 Interactive Testing Mode")
    print("-" * 40)
    print("Enter questions to test the content filter.")
    print("Type 'quit' to exit.")
    print()
    
    filter_service = ContentFilterService()
    
    while True:
        try:
            user_input = input("👤 Enter your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Test the filter
            result = filter_service.filter_content(user_input, language='ar')
            
            if result.is_allowed:
                print("✅ ALLOWED - This question would be processed by the model")
            else:
                print("🚫 BLOCKED - Content filter activated")
                print(f"   Reason: {result.filter_reason}")
                print(f"   Confidence: {result.confidence_score:.2f}")
                print(f"   Response: {result.suggested_response}")
                if result.matched_patterns:
                    print(f"   Matched Patterns: {result.matched_patterns[:3]}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_content_filter()
        print("\n💡 To test interactively, run: python content_filter_demo.py --interactive")
