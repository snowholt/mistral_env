"""
Content filtering service for BeautyAI inference framework.

This service implements content filtering to prevent the model from answering
questions about cosmetic procedures or related topics that are outside the
approved scope defined in the reference CSV file.
"""
import logging
import re
import csv
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from ..base.base_service import BaseService

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_allowed: bool
    filter_reason: Optional[str] = None
    confidence_score: float = 0.0
    matched_patterns: List[str] = None
    suggested_response: Optional[str] = None

    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []


class ContentFilterService(BaseService):
    """
    Content filtering service that prevents models from answering inappropriate questions.
    
    This service:
    - Loads forbidden topics from the reference CSV file
    - Analyzes incoming user queries for restricted content
    - Provides safety responses for filtered content
    - Supports both exact matching and semantic similarity detection
    """
    
    def __init__(self, reference_csv_path: Optional[str] = None):
        super().__init__()
        self.reference_csv_path = reference_csv_path or "/home/lumi/beautyai/refrences/2000QAToR.csv"
        self.forbidden_topics = set()
        self.forbidden_keywords = set()
        self.question_patterns = []
        self.safety_responses = {
            'ar': "عذراً، لا يمكنني الإجابة على أسئلة تتعلق بالإجراءات التجميلية أو العلاجات الطبية. يُرجى استشارة طبيب مختص للحصول على المشورة الطبية المناسبة.",
            'en': "I apologize, but I cannot answer questions related to cosmetic procedures or medical treatments. Please consult with a qualified medical professional for appropriate medical advice."
        }
        self._load_forbidden_content()
    
    def _load_forbidden_content(self) -> None:
        """Load forbidden content from the reference CSV file."""
        try:
            if not Path(self.reference_csv_path).exists():
                logger.warning(f"Reference CSV file not found: {self.reference_csv_path}")
                return
            
            with open(self.reference_csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    question = row.get('السؤال', '').strip()
                    if question:
                        # Extract topics and keywords from questions
                        self._extract_topics_and_keywords(question)
                        self.question_patterns.append(question.lower())
            
            logger.info(f"Loaded {len(self.forbidden_topics)} forbidden topics and {len(self.forbidden_keywords)} keywords")
            logger.info(f"Sample forbidden topics: {list(self.forbidden_topics)[:10]}")
            
        except Exception as e:
            logger.error(f"Error loading forbidden content: {str(e)}")
    
    def _extract_topics_and_keywords(self, question: str) -> None:
        """Extract topics and keywords from a question."""
        # Common cosmetic procedure terms in Arabic
        cosmetic_terms = [
            'البوتوكس', 'بوتوكس', 'السيلوليت', 'حب الشباب', 'الرؤوس السوداء',
            'شد الوجه', 'الشفاه', 'جلسات النضارة', 'تنظيف البشرة', 'تنقية المسامات',
            'تجديد البشرة', 'التجاعيد', 'التصبغات', 'الهيدرافيشل', 'التقشير الهوليودي',
            'الهالات السوداء', 'الماسكات الطبية', 'ترهل الجلد', 'المسام الواسعة',
            'شد الرقبة', 'العناية بالبشرة', 'تكبير الشفاه', 'الميزوثيرابي',
            'نحت الجسم', 'حقن الخدود', 'تفتيح البشرة', 'إزالة الزيوان',
            'حقن الدهون الذاتية', 'الخلايا الجذعية', 'الليزر الكربوني',
            'تنظيف عميق للبشرة', 'الأنف التجميلي', 'علاج البقع الداكنة',
            'الليزر', 'توريد الشفاه', 'تقشير بارد', 'تقشير كيميائي',
            'الندبات', 'النضارة', 'ليزر فراكشنال', 'تفتيح الإبط',
            'تنشيط الدورة الدموية', 'علاج المسام المفتوحة', 'إزالة الوشم',
            'جراحة تجميلية', 'عملية تجميل', 'حقن', 'فيلر', 'كولاجين'
        ]
        
        # Add terms found in the question
        for term in cosmetic_terms:
            if term in question:
                self.forbidden_topics.add(term)
                # Split compound terms into keywords
                words = term.split()
                self.forbidden_keywords.update(words)
        
        # Extract additional keywords using regex patterns
        arabic_words = re.findall(r'[\u0600-\u06FF]+', question)
        for word in arabic_words:
            if len(word) > 2:  # Only consider words longer than 2 characters
                self.forbidden_keywords.add(word)
    
    def filter_content(self, user_input: str, language: str = 'ar') -> FilterResult:
        """
        Filter user input to check if it contains forbidden content.
        
        Args:
            user_input: The user's input text to filter
            language: The language of the input ('ar' for Arabic, 'en' for English)
            
        Returns:
            FilterResult indicating whether the content should be allowed
        """
        user_input_lower = user_input.lower().strip()
        
        if not user_input_lower:
            return FilterResult(is_allowed=True, confidence_score=1.0)
        
        # Check for exact topic matches
        topic_matches = self._check_topic_matches(user_input_lower)
        if topic_matches:
            return FilterResult(
                is_allowed=False,
                filter_reason="Contains forbidden cosmetic procedure topics",
                confidence_score=0.9,
                matched_patterns=topic_matches,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        # Check for keyword density
        keyword_score = self._calculate_keyword_density(user_input_lower)
        if keyword_score > 0.5:  # High concentration of cosmetic-related keywords (raised from 0.3 to 0.5)
            return FilterResult(
                is_allowed=False,
                filter_reason="High density of cosmetic procedure keywords",
                confidence_score=keyword_score,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        # Check for pattern similarity
        pattern_similarity = self._check_pattern_similarity(user_input_lower)
        if pattern_similarity > 0.7:  # High similarity to known forbidden patterns
            return FilterResult(
                is_allowed=False,
                filter_reason="Similar to known forbidden question patterns",
                confidence_score=pattern_similarity,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        # Check for medical/cosmetic procedure indicators
        medical_indicators = self._check_medical_indicators(user_input_lower)
        if medical_indicators:
            return FilterResult(
                is_allowed=False,
                filter_reason="Contains medical/cosmetic procedure indicators",
                confidence_score=0.8,
                matched_patterns=medical_indicators,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        return FilterResult(is_allowed=True, confidence_score=1.0)
    
    def _check_topic_matches(self, text: str) -> List[str]:
        """Check for direct topic matches in the text."""
        matches = []
        for topic in self.forbidden_topics:
            if topic.lower() in text:
                matches.append(topic)
        return matches
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate the density of forbidden keywords in the text."""
        words = re.findall(r'[\u0600-\u06FF\w]+', text)
        if not words:
            return 0.0
        
        forbidden_word_count = 0
        for word in words:
            if word in self.forbidden_keywords:
                forbidden_word_count += 1
        
        return forbidden_word_count / len(words)
    
    def _check_pattern_similarity(self, text: str) -> float:
        """Check similarity to known forbidden question patterns."""
        max_similarity = 0.0
        
        for pattern in self.question_patterns[:100]:  # Check against first 100 patterns for performance
            similarity = self._calculate_text_similarity(text, pattern)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""
        words1 = set(re.findall(r'[\u0600-\u06FF\w]+', text1.lower()))
        words2 = set(re.findall(r'[\u0600-\u06FF\w]+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _check_medical_indicators(self, text: str) -> List[str]:
        """Check for medical/cosmetic procedure indicators."""
        indicators = [
            # Arabic medical/cosmetic indicators
            r'ما\s+تكلفة',  # "What is the cost of"
            r'هل\s+.*\s+آمن',  # "Is ... safe"
            r'متى\s+تظهر\s+نتائج',  # "When do results appear"
            r'كم\s+جلسة',  # "How many sessions"
            r'ما\s+أضرار',  # "What are the harms of"
            r'هل\s+.*\s+مؤلم',  # "Is ... painful"
            r'بدائل\s+طبيعية',  # "Natural alternatives"
            r'العيادة\s+المناسبة',  # "Appropriate clinic"
            r'المواد\s+المستخدمة',  # "Materials used"
            r'آثار\s+جانبية',  # "Side effects"
            r'أفضل\s+مرشح',  # "Best candidate"
            r'تجنب\s+.*\s+بعد',  # "Avoid ... after"
            
            # English medical/cosmetic indicators
            r'cost\s+of\s+.*\s+treatment',
            r'side\s+effects\s+of',
            r'how\s+many\s+sessions',
            r'is\s+.*\s+safe',
            r'when\s+do\s+results',
            r'clinic\s+for\s+.*\s+treatment',
            r'natural\s+alternatives\s+to',
            r'recovery\s+time\s+for'
        ]
        
        matches = []
        for pattern in indicators:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        return matches
    
    def add_custom_forbidden_term(self, term: str) -> None:
        """Add a custom forbidden term to the filter."""
        self.forbidden_topics.add(term.lower())
        logger.info(f"Added custom forbidden term: {term}")
    
    def remove_forbidden_term(self, term: str) -> None:
        """Remove a term from the forbidden list."""
        self.forbidden_topics.discard(term.lower())
        logger.info(f"Removed forbidden term: {term}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the content filter."""
        return {
            "total_forbidden_topics": len(self.forbidden_topics),
            "total_forbidden_keywords": len(self.forbidden_keywords),
            "total_question_patterns": len(self.question_patterns),
            "reference_csv_path": self.reference_csv_path,
            "csv_exists": Path(self.reference_csv_path).exists()
        }
    
    def validate_filter_configuration(self) -> Dict[str, Any]:
        """Validate the filter configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if reference CSV exists
        if not Path(self.reference_csv_path).exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Reference CSV file not found: {self.reference_csv_path}")
        
        # Check if we have loaded forbidden content
        if not self.forbidden_topics:
            validation_result["warnings"].append("No forbidden topics loaded")
        
        if not self.forbidden_keywords:
            validation_result["warnings"].append("No forbidden keywords loaded")
        
        return validation_result
