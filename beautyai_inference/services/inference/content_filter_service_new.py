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
    Content filtering service that ensures responses stay within medical/beauty domain.
    
    This service:
    - ALLOWS: Medical, beauty, cosmetic, skincare, aesthetic questions and treatments
    - ALLOWS: Questions about prices, scheduling, procedures, botox, clinics, etc.
    - BLOCKS: Sexual/sensual content and topics completely outside medical/beauty domain
    """
    
    def __init__(self, reference_csv_path: Optional[str] = None, strictness_level: str = "balanced"):
        super().__init__()
        self.reference_csv_path = reference_csv_path or "/home/lumi/beautyai/refrences/2000QAToR.csv"
        self.strictness_level = strictness_level  # "strict", "balanced", "relaxed", "disabled"
        
        # Medical/beauty topics and keywords (ALLOWED)
        self.allowed_medical_topics = set()
        self.allowed_medical_keywords = set()
        
        # Sexual/sensual and off-topic content (BLOCKED)
        self.forbidden_topics = set()
        self.forbidden_keywords = set()
        
        self.safety_responses = {
            'ar': "عذراً، يمكنني فقط الإجابة على الأسئلة المتعلقة بالطب التجميلي والعناية بالبشرة والجمال. يُرجى طرح سؤال متعلق بهذا المجال.",
            'en': "I apologize, but I can only answer questions related to cosmetic medicine, skincare, and beauty. Please ask a question related to this field."
        }
        self._initialize_content_categories()
    
    def _initialize_content_categories(self) -> None:
        """Initialize allowed and forbidden content categories."""
        self._load_allowed_medical_content()
        self._load_forbidden_content()
        logger.info(f"Content filter initialized - Allowed: {len(self.allowed_medical_keywords)} medical keywords, Forbidden: {len(self.forbidden_keywords)} inappropriate keywords")
    
    def _load_allowed_medical_content(self) -> None:
        """Load medical/beauty content that is ALLOWED."""
        # Medical and beauty terms in Arabic
        medical_beauty_terms_ar = [
            # Beauty and skincare
            'البشرة', 'الجلد', 'الوجه', 'الجمال', 'العناية', 'النضارة', 'التنظيف',
            'حب الشباب', 'البثور', 'الرؤوس السوداء', 'المسام', 'التجاعيد', 'التصبغات',
            'الهالات السوداء', 'البقع الداكنة', 'الندبات', 'الترهل', 'الشيخوخة',
            
            # Cosmetic procedures
            'البوتوكس', 'بوتوكس', 'الفيلر', 'فيلر', 'الحقن', 'حقن', 'الليزر', 'ليزر',
            'التقشير', 'تقشير', 'الميزوثيرابي', 'ميزوثيرابي', 'الهيدرافيشل', 'هيدرافيشل',
            'التقشير الكيميائي', 'التقشير الهوليودي', 'الليزر الكربوني', 'ليزر فراكشنال',
            'الكولاجين', 'كولاجين', 'الخلايا الجذعية', 'خلايا جذعية',
            
            # Body treatments
            'نحت الجسم', 'شد الجسم', 'إزالة الدهون', 'تفتيت الدهون', 'السيلوليت',
            'شد الوجه', 'شد الرقبة', 'تكبير الشفاه', 'توريد الشفاه', 'حقن الشفاه',
            'تكبير الثدي', 'تصغير الثدي', 'شد البطن', 'عملية تجميل',
            
            # Medical terms
            'طبيب', 'دكتور', 'عيادة', 'مستشفى', 'مركز طبي', 'عيادة تجميل',
            'استشارة طبية', 'فحص طبي', 'تشخيص', 'علاج', 'عملية', 'إجراء',
            'جراحة تجميلية', 'طب تجميلي', 'تجميل',
            
            # Scheduling and business
            'موعد', 'حجز', 'تكلفة', 'سعر', 'ثمن', 'عرض', 'باقة', 'جلسة',
            'متابعة', 'نتائج', 'مدة العلاج', 'فترة النقاهة', 'التعافي',
            
            # Questions and informational
            'ما هو', 'كيف', 'متى', 'أين', 'لماذا', 'من', 'هل', 'الفرق بين',
            'أنواع', 'أسباب', 'أعراض', 'فوائد', 'أضرار', 'آثار جانبية', 'مخاطر'
        ]
        
        # Medical and beauty terms in English
        medical_beauty_terms_en = [
            # Beauty and skincare
            'skin', 'face', 'beauty', 'skincare', 'facial', 'complexion', 'cleansing',
            'acne', 'pimples', 'blackheads', 'pores', 'wrinkles', 'pigmentation',
            'dark circles', 'dark spots', 'scars', 'sagging', 'aging', 'anti-aging',
            
            # Cosmetic procedures
            'botox', 'filler', 'injection', 'laser', 'peeling', 'peel', 'mesotherapy',
            'hydrafacial', 'chemical peel', 'carbon laser', 'fractional laser',
            'collagen', 'stem cells', 'plasma', 'radiofrequency', 'ultrasound',
            
            # Body treatments
            'body contouring', 'body sculpting', 'liposuction', 'fat removal', 'cellulite',
            'facelift', 'neck lift', 'lip augmentation', 'lip filler', 'breast augmentation',
            'breast reduction', 'tummy tuck', 'cosmetic surgery', 'plastic surgery',
            
            # Medical terms
            'doctor', 'physician', 'clinic', 'hospital', 'medical center', 'cosmetic clinic',
            'consultation', 'examination', 'diagnosis', 'treatment', 'procedure', 'surgery',
            'medical', 'cosmetic', 'aesthetic', 'dermatology', 'dermatologist',
            
            # Scheduling and business
            'appointment', 'booking', 'cost', 'price', 'fee', 'package', 'session',
            'follow-up', 'results', 'duration', 'recovery', 'healing', 'downtime',
            
            # Questions and informational
            'what is', 'how', 'when', 'where', 'why', 'who', 'difference between',
            'types of', 'causes', 'symptoms', 'benefits', 'risks', 'side effects'
        ]
        
        # Add all terms to allowed sets
        self.allowed_medical_topics.update(medical_beauty_terms_ar + medical_beauty_terms_en)
        self.allowed_medical_keywords.update(medical_beauty_terms_ar + medical_beauty_terms_en)
        
        # Add common medical keywords (any individual words)
        for term in medical_beauty_terms_ar + medical_beauty_terms_en:
            words = term.split()
            self.allowed_medical_keywords.update(words)
    
    def _load_forbidden_content(self) -> None:
        """Load sexual/sensual and off-topic content that is FORBIDDEN."""
        # Sexual and sensual content (Arabic and English)
        forbidden_terms = [
            # Sexual content in Arabic
            'جنس', 'جنسي', 'جنسية', 'حب', 'عشق', 'غرام', 'شهوة', 'إثارة',
            'ممارسة', 'علاقة حميمة', 'قبلة', 'عناق', 'لمسة', 'إغراء', 'إغواء',
            
            # Sexual content in English
            'sex', 'sexual', 'intimate', 'romance', 'romantic', 'love', 'kiss',
            'touch', 'seduce', 'seduction', 'arousal', 'desire', 'lust', 'passion',
            'erotic', 'sensual', 'sexy', 'hot', 'attraction', 'flirt', 'dating',
            
            # Completely off-topic subjects
            'طبخ', 'طعام', 'كرة القدم', 'رياضة', 'سياسة', 'اقتصاد', 'تكنولوجيا',
            'cooking', 'food', 'football', 'sports', 'politics', 'economy', 'technology',
            'programming', 'computer', 'software', 'hardware', 'gaming', 'music',
            'movies', 'entertainment', 'travel', 'vacation', 'weather', 'news'
        ]
        
        self.forbidden_topics.update(forbidden_terms)
        self.forbidden_keywords.update(forbidden_terms)
        
        # Add individual words from compound terms
        for term in forbidden_terms:
            words = term.split()
            if len(words) > 1:
                self.forbidden_keywords.update(words)
    
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
        
        # Check strictness level
        if self.strictness_level == "disabled":
            return FilterResult(is_allowed=True, confidence_score=1.0)
        
        # First, check for explicitly forbidden content (sexual/sensual)
        forbidden_matches = self._check_forbidden_content(user_input_lower)
        if forbidden_matches:
            return FilterResult(
                is_allowed=False,
                filter_reason="Contains inappropriate or sexual content",
                confidence_score=0.9,
                matched_patterns=forbidden_matches,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        # Check if content is related to medical/beauty domain
        is_medical_beauty_related = self._is_medical_beauty_related(user_input_lower)
        
        if not is_medical_beauty_related:
            return FilterResult(
                is_allowed=False,
                filter_reason="Content is outside medical/beauty domain",
                confidence_score=0.8,
                suggested_response=self.safety_responses.get(language, self.safety_responses['en'])
            )
        
        # If it's medical/beauty related and not forbidden, allow it
        return FilterResult(is_allowed=True, confidence_score=1.0)
    
    def _check_forbidden_content(self, text: str) -> List[str]:
        """Check for explicitly forbidden sexual/sensual content."""
        matches = []
        
        for forbidden_term in self.forbidden_topics:
            if forbidden_term.lower() in text:
                matches.append(forbidden_term)
        
        return matches
    
    def _is_medical_beauty_related(self, text: str) -> bool:
        """Check if the text is related to medical/beauty domain."""
        # Extract words from the text
        words = re.findall(r'[\u0600-\u06FF\w]+', text.lower())
        
        if not words:
            return False
        
        # Count medical/beauty related words
        medical_word_count = 0
        for word in words:
            if word in self.allowed_medical_keywords:
                medical_word_count += 1
        
        # Also check for medical phrases
        phrase_matches = 0
        for topic in self.allowed_medical_topics:
            if len(topic.split()) > 1 and topic.lower() in text:
                phrase_matches += 1
        
        # Calculate relevance score
        total_relevant = medical_word_count + phrase_matches
        relevance_ratio = total_relevant / len(words) if words else 0
        
        # Adjust threshold based on strictness
        if self.strictness_level == "strict":
            threshold = 0.3  # At least 30% of words must be medical/beauty related
        elif self.strictness_level == "balanced":
            threshold = 0.2  # At least 20% of words must be medical/beauty related  
        else:  # "relaxed"
            threshold = 0.1  # At least 10% of words must be medical/beauty related
        
        # Also allow if we have any exact medical phrase match
        return relevance_ratio >= threshold or phrase_matches > 0
    
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
    
    def add_allowed_term(self, term: str) -> None:
        """Add a custom allowed medical/beauty term."""
        self.allowed_medical_topics.add(term.lower())
        self.allowed_medical_keywords.add(term.lower())
        logger.info(f"Added allowed medical term: {term}")
    
    def add_forbidden_term(self, term: str) -> None:
        """Add a custom forbidden term."""
        self.forbidden_topics.add(term.lower())
        self.forbidden_keywords.add(term.lower())
        logger.info(f"Added forbidden term: {term}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the content filter."""
        return {
            "strictness_level": self.strictness_level,
            "allowed_medical_topics": len(self.allowed_medical_topics),
            "allowed_medical_keywords": len(self.allowed_medical_keywords),
            "forbidden_topics": len(self.forbidden_topics),
            "forbidden_keywords": len(self.forbidden_keywords),
            "reference_csv_path": self.reference_csv_path
        }
    
    def validate_filter_configuration(self) -> Dict[str, Any]:
        """Validate the filter configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if we have loaded content
        if not self.allowed_medical_keywords:
            validation_result["valid"] = False
            validation_result["errors"].append("No allowed medical keywords loaded")
        
        if not self.forbidden_keywords:
            validation_result["warnings"].append("No forbidden keywords loaded")
        
        return validation_result
