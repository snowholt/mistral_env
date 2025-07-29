"""
Content filtering service for BeautyAI inference framework.

This service implements content filtering to ensure the model ONLY answers
questions related to medical/beauty/cosmetic topics and blocks sensual/sexual
content or topics outside the medical/beauty domain.
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
    Content filtering service for beauty clinic conversations.
    
    This service:
    - ALLOWS medical/beauty clinic topics (botox, treatments, prices, scheduling)
    - BLOCKS sensual/sexual content and off-topic conversations
    - Keeps conversations focused on medical/beauty clinic context
    - Prevents inappropriate or non-medical discussions
    """
    
    def __init__(self, reference_csv_path: Optional[str] = None, strictness_level: str = "balanced"):
        super().__init__()
        self.reference_csv_path = reference_csv_path or "/home/lumi/beautyai/refrences/2000QAToR.csv"
        self.strictness_level = strictness_level  # "strict", "balanced", "relaxed", "disabled"
        
        # ALLOWED topics (medical/beauty clinic related)
        self.allowed_medical_topics = set()
        self.allowed_medical_keywords = set()
        
        # FORBIDDEN topics (sensual/sexual/off-topic)
        self.forbidden_topics = set()
        self.forbidden_keywords = set()
        
        self.question_patterns = []
        self.safety_responses = {
            'ar': "عذراً، يمكنني فقط الإجابة على الأسئلة المتعلقة بالعيادات التجميلية والعلاجات الطبية. يُرجى طرح أسئلة حول الإجراءات التجميلية أو العلاجات الطبية.",
            'en': "I apologize, but I can only answer questions related to beauty clinics and medical treatments. Please ask questions about cosmetic procedures or medical treatments."
        }
        self._load_content_categories()
    
    def _load_content_categories(self) -> None:
        """Load allowed medical topics and forbidden off-topic content."""
        try:
            # Load medical/beauty topics from CSV (these are ALLOWED)
            csv_path = Path(self.reference_csv_path)
            if csv_path.exists():
                import csv
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question = row.get('question', '') or row.get('السؤال', '')
                        question = question.strip()
                        if question:
                            # Extract medical terms from the question
                            self._extract_medical_keywords(question)
                            self.question_patterns.append(question.lower())
                
                logger.info(f"Loaded {len(self.question_patterns)} medical question patterns from CSV")
            else:
                logger.warning(f"CSV file not found: {self.reference_csv_path}")
            
            # Define core medical/beauty terms that are ALWAYS ALLOWED
            self._load_allowed_medical_terms()
            
            # Define forbidden content (sensual/sexual/off-topic)
            self._load_forbidden_content()
            
            logger.info(f"Loaded {len(self.allowed_medical_topics)} allowed medical topics and {len(self.allowed_medical_keywords)} keywords")
            logger.info(f"Loaded {len(self.forbidden_topics)} forbidden topics and {len(self.forbidden_keywords)} keywords")
            
        except Exception as e:
            logger.error(f"Error loading content categories: {e}")
            # Load minimal defaults if CSV fails
            self._load_allowed_medical_terms()
            self._load_forbidden_content()

    def _load_allowed_medical_terms(self) -> None:
        """Load medical/beauty clinic terms that are ALLOWED."""
        # Medical/beauty clinic terms in Arabic
        allowed_arabic_terms = [
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
            'جراحة تجميلية', 'عملية تجميل', 'حقن', 'فيلر', 'كولاجين',
            'عيادة', 'طبيب', 'علاج', 'تكلفة', 'سعر', 'موعد', 'جلسة',
            'استشارة', 'فحص', 'تشخيص', 'نتائج', 'آثار جانبية', 'مضاعفات'
        ]
        
        # Medical/beauty clinic terms in English
        allowed_english_terms = [
            'botox', 'laser', 'treatment', 'clinic', 'doctor', 'procedure', 'surgery',
            'cosmetic', 'beauty', 'skin', 'facial', 'injection', 'filler', 'peeling',
            'appointment', 'consultation', 'cost', 'price', 'session', 'results',
            'side effects', 'medical', 'dermatology', 'aesthetic', 'plastic surgery'
        ]
        
        self.allowed_medical_topics.update(allowed_arabic_terms + allowed_english_terms)
        
        # Extract keywords from allowed topics
        for topic in self.allowed_medical_topics:
            words = re.findall(r'[\u0600-\u06FF\w]+', topic)
            self.allowed_medical_keywords.update(word for word in words if len(word) > 2)

    def _load_forbidden_content(self) -> None:
        """Load forbidden content (sensual/sexual/off-topic)."""
        # Sensual/sexual content terms
        forbidden_terms = [
            # Arabic sensual/sexual terms
            'جنس', 'جنسي', 'جنسية', 'إغراء', 'إثارة', 'شهوة', 'رغبة جنسية',
            'علاقة حميمة', 'حب', 'عشق', 'غرام', 'هيام', 'وله', 'عاطفة',
            'قبلة', 'عناق', 'لمس', 'احتكاك', 'ملامسة', 'تقبيل',
            
            # English sensual/sexual terms
            'sex', 'sexual', 'sexy', 'seductive', 'intimate', 'romance', 'romantic',
            'love', 'kiss', 'hug', 'touch', 'caress', 'desire', 'attraction',
            'flirt', 'dating', 'relationship', 'partner', 'boyfriend', 'girlfriend',
            
            # Off-topic general conversation
            'طقس', 'weather', 'رياضة', 'sports', 'سياسة', 'politics', 'أخبار', 'news',
            'طعام', 'food', 'طبخ', 'cook', 'cooking', 'recipe', 'وصفة', 'سفر', 'travel', 
            'موسيقى', 'music', 'فيلم', 'movie', 'كتاب', 'book', 'لعبة', 'game', 
            'تسوق', 'shopping', 'سيارة', 'car', 'pasta', 'meal', 'restaurant'
        ]
        
        self.forbidden_topics.update(forbidden_terms)
        
        # Extract keywords from forbidden topics
        for topic in self.forbidden_topics:
            words = re.findall(r'[\u0600-\u06FF\w]+', topic)
            self.forbidden_keywords.update(word for word in words if len(word) > 2)

    def _extract_medical_keywords(self, question: str) -> None:
        """Extract medical keywords from questions (these are ALLOWED)."""
        # Extract Arabic and English words
        words = re.findall(r'[\u0600-\u06FF\w]+', question.lower())
        
        for word in words:
            if len(word) > 2:  # Only consider words longer than 2 characters
                self.allowed_medical_keywords.add(word)
    
    def filter_content(self, user_input: str, language: str = 'ar') -> FilterResult:
        """
        Filter user input for beauty clinic appropriateness.
        
        NEW LOGIC:
        - ALLOWS: Medical/beauty clinic topics (botox, treatments, prices, scheduling)
        - BLOCKS: Sensual/sexual content and off-topic conversations
        
        Args:
            user_input: The user's input text to filter
            language: The language of the input ('ar' for Arabic, 'en' for English)
            
        Returns:
            FilterResult indicating whether the content should be allowed
        """
        user_input_lower = user_input.lower().strip()
        
        if not user_input_lower:
            return FilterResult(is_allowed=True, confidence_score=1.0)
        
        # Check strictness level
        if self.strictness_level == "disabled":
            return FilterResult(is_allowed=True, confidence_score=1.0)
        
        # Step 1: Check for FORBIDDEN content (sensual/sexual/off-topic)
        forbidden_result = self._check_forbidden_content(user_input_lower, language)
        if forbidden_result:
            return forbidden_result
        
        # Step 2: Check for ALLOWED medical content
        allowed_result = self._check_medical_content(user_input_lower, language)
        if allowed_result:
            return allowed_result
        
        # Step 3: If content is neither clearly forbidden nor clearly medical, 
        # be permissive and allow it (might be general medical question)
        return FilterResult(
            is_allowed=True, 
            confidence_score=0.6,
            filter_reason="Neutral content - allowing as potentially medical-related"
        )
    
    def _check_forbidden_content(self, text: str, language: str) -> Optional[FilterResult]:
        """Check if content contains forbidden topics (sensual/sexual/off-topic)."""
        
        # Check for direct forbidden topic matches
        forbidden_matches = []
        for topic in self.forbidden_topics:
            if topic.lower() in text:
                forbidden_matches.append(topic)
        
        if forbidden_matches:
            return FilterResult(
                is_allowed=False,
                filter_reason=f"Contains forbidden content: {forbidden_matches[:3]}",
                confidence_score=0.9,
                matched_patterns=forbidden_matches[:5],
                suggested_response=self.safety_responses.get(language, self.safety_responses['ar'])
            )
        
        # Check for sensual/sexual indicators
        sensual_patterns = [
            # Arabic patterns
            r'حب[ي]?ب[يت]?', r'عشق', r'غرام', r'عاطف[ةه]', r'رومانس[يه]',
            r'قبل[ةه]', r'أقبل', r'تقبيل', r'عناق', r'لمس', r'ملامس[ةه]', r'إثار[ةه]', r'إغراء',
            r'حبيب[يت]?', r'حبيبة', r'عشيق[ة]?', r'رومانسي[ة]?',
            
            # English patterns  
            r'love', r'romance', r'romantic', r'kiss', r'hug', r'touch', r'caress',
            r'sexy', r'seductive', r'intimate', r'flirt', r'dating', r'boyfriend', r'girlfriend',
            r'cook', r'food', r'recipe', r'pasta', r'meal'  # Food/cooking related
        ]
        
        for pattern in sensual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return FilterResult(
                    is_allowed=False,
                    filter_reason=f"Contains sensual/romantic content: {pattern}",
                    confidence_score=0.85,
                    matched_patterns=[pattern],
                    suggested_response=self.safety_responses.get(language, self.safety_responses['ar'])
                )
        
        # Check for general off-topic conversations
        off_topic_density = self._calculate_off_topic_density(text)
        if off_topic_density > 0.7:  # More than 70% off-topic keywords
            return FilterResult(
                is_allowed=False,
                filter_reason=f"Off-topic conversation (density: {off_topic_density:.2f})",
                confidence_score=0.8,
                matched_patterns=["high_off_topic_density"],
                suggested_response=self.safety_responses.get(language, self.safety_responses['ar'])
            )
        
        return None  # No forbidden content found
    
    def _check_medical_content(self, text: str, language: str) -> Optional[FilterResult]:
        """Check if content contains medical/beauty clinic topics (ALLOWED)."""
        
        # Check for direct medical topic matches
        medical_matches = []
        for topic in self.allowed_medical_topics:
            if topic.lower() in text:
                medical_matches.append(topic)
        
        if medical_matches:
            return FilterResult(
                is_allowed=True,
                confidence_score=0.95,
                filter_reason=f"Contains medical/beauty topics: {medical_matches[:3]}"
            )
        
        # Check for medical question patterns
        medical_patterns = [
            # Arabic medical question patterns
            r'ما (تكلفة|سعر|ثمن)', r'كم (يكلف|سعر|ثمن)', r'هل.*آمن', r'متى تظهر نتائج',
            r'كم جلسة', r'أين (أجد|يمكن)', r'ما أفضل (علاج|طريقة)', r'كيف (أعالج|علاج)',
            r'(عيادة|طبيب|دكتور)', r'(استشارة|فحص|موعد)', r'(علاج|إجراء|عملية)',
            
            # English medical question patterns  
            r'(cost|price) of', r'how much (does|for)', r'is.*safe', r'when.*results',
            r'how many sessions', r'where (can|to find)', r'best (treatment|way)',
            r'how to treat', r'(clinic|doctor)', r'(consultation|appointment)',
            r'(treatment|procedure|surgery)'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return FilterResult(
                    is_allowed=True,
                    confidence_score=0.9,
                    filter_reason=f"Medical question pattern: {pattern}"
                )
        
        # Check medical keyword density
        medical_density = self._calculate_medical_density(text)
        if medical_density > 0.3:  # More than 30% medical keywords
            return FilterResult(
                is_allowed=True,
                confidence_score=0.8,
                filter_reason=f"High medical content density: {medical_density:.2f}"
            )
        
        return None  # No clear medical content found
    
    def _calculate_medical_density(self, text: str) -> float:
        """Calculate the density of medical keywords in the text."""
        words = re.findall(r'[\u0600-\u06FF\w]+', text)
        if not words:
            return 0.0
        
        medical_word_count = 0
        for word in words:
            if word.lower() in self.allowed_medical_keywords:
                medical_word_count += 1
        
        return medical_word_count / len(words)
    
    def _calculate_off_topic_density(self, text: str) -> float:
        """Calculate the density of off-topic keywords in the text."""
        words = re.findall(r'[\u0600-\u06FF\w]+', text)
        if not words:
            return 0.0
        
        off_topic_word_count = 0
        for word in words:
            if word.lower() in self.forbidden_keywords:
                off_topic_word_count += 1
        
        return off_topic_word_count / len(words)
        
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
        
    def set_strictness_level(self, level: str) -> None:
        """Set the strictness level of the content filter."""
        valid_levels = ["strict", "balanced", "relaxed", "disabled"]
        if level in valid_levels:
            self.strictness_level = level
            logger.info(f"Content filter strictness level set to: {level}")
        else:
            logger.warning(f"Invalid strictness level: {level}. Valid levels: {valid_levels}")
    
    def get_strictness_level(self) -> str:
        """Get the current strictness level."""
        return self.strictness_level
    
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
            "strictness_level": self.strictness_level,
            "total_allowed_medical_topics": len(self.allowed_medical_topics),
            "total_allowed_medical_keywords": len(self.allowed_medical_keywords),
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
        
        # Check if we have loaded allowed medical content
        if not self.allowed_medical_topics:
            validation_result["warnings"].append("No allowed medical topics loaded")
        
        if not self.allowed_medical_keywords:
            validation_result["warnings"].append("No allowed medical keywords loaded")
        
        # Check if we have loaded forbidden content
        if not self.forbidden_topics:
            validation_result["warnings"].append("No forbidden topics loaded")
        
        return validation_result
