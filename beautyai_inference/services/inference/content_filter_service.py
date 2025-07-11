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
        
        # Add common medical keywords from compound terms (be more selective)
        medical_keywords_to_add = []
        for term in medical_beauty_terms_ar + medical_beauty_terms_en:
            # Only add individual words that are clearly medical/beauty related
            words = term.split()
            for word in words:
                if len(word) > 3 and word not in ['the', 'and', 'or', 'of', 'في', 'من', 'إلى', 'على']:
                    # Skip common non-medical words
                    non_medical_words = ['best', 'good', 'better', 'أفضل', 'جيد', 'أحسن', 'how', 'what', 'كيف', 'ما']
                    if word not in non_medical_words:
                        medical_keywords_to_add.append(word)
        
        self.allowed_medical_keywords.update(medical_keywords_to_add)
    
    def _load_forbidden_content(self) -> None:
        """Load sexual/sensual and off-topic content that is FORBIDDEN."""
        # Sexual and sensual content (be more specific to avoid false positives)
        forbidden_terms = [
            # Sexual content in Arabic (use more specific terms)
            'جنس', 'جنسي', 'جنسية', 'عشق', 'غرام', 'شهوة', 'إثارة جنسية',
            'ممارسة الجنس', 'علاقة حميمة', 'إغراء جنسي', 'إغواء جنسي',
            'رومانسي', 'عاطفي', 'صديق', 'صديقة', 'حبيب', 'حبيبة',
            'زواج', 'خطوبة', 'موعد غرامي', 'لقاء رومانسي',
            
            # Sexual content in English (specific terms)
            'sex', 'sexual', 'intimate relationship', 'romance', 'romantic', 
            'boyfriend', 'girlfriend', 'dating', 'love affair', 'seduction',
            'kiss', 'kissing', 'touching', 'seduce', 'arousal', 'desire', 'lust', 'passion',
            'erotic', 'sensual', 'sexy', 'hot body', 'attraction', 'flirt', 'flirting',
            'marriage', 'wedding', 'date night', 'romantic dinner',
            
            # Completely off-topic subjects (not medical/beauty related)
            'طبخ', 'طعام', 'مطبخ', 'وصفة', 'كرة القدم', 'رياضة', 'سياسة', 'اقتصاد', 
            'تكنولوجيا', 'حاسوب', 'برمجة', 'ألعاب', 'موسيقى', 'أفلام', 'سفر',
            'طقس', 'أخبار', 'سيارات', 'مواصلات', 'تعليم', 'دراسة', 'عمل', 'وظيفة',
            
            'cooking', 'food', 'recipe', 'kitchen', 'restaurant', 'football', 'soccer',
            'sports', 'politics', 'economy', 'technology', 'programming', 'computer', 
            'software', 'hardware', 'gaming', 'music', 'movies', 'entertainment', 
            'travel', 'vacation', 'weather', 'news', 'cars', 'transportation',
            'education', 'study', 'work', 'job', 'career'
        ]
        
        self.forbidden_topics.update(forbidden_terms)
        self.forbidden_keywords.update(forbidden_terms)
    
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
        
        # Check for forbidden phrases first (more specific)
        forbidden_phrases = [
            'علاقة حميمة', 'ممارسة الجنس', 'إثارة جنسية', 'إغراء جنسي', 'موعد غرامي',
            'intimate relationship', 'sexual intercourse', 'sexual arousal', 'romantic date'
        ]
        
        for phrase in forbidden_phrases:
            if phrase.lower() in text:
                matches.append(phrase)
        
        # Then check individual forbidden terms, but be more careful
        for forbidden_term in self.forbidden_topics:
            if len(forbidden_term) > 3 and forbidden_term.lower() in text:
                # Skip if it's part of a medical term
                medical_exceptions = [
                    'حب الشباب',  # acne (contains 'حب' but is medical)
                    'love handles',  # medical term for body fat
                ]
                
                is_medical_context = False
                for medical_term in medical_exceptions:
                    if medical_term.lower() in text:
                        is_medical_context = True
                        break
                
                if not is_medical_context:
                    matches.append(forbidden_term)
        
        return matches
    
    def _is_medical_beauty_related(self, text: str) -> bool:
        """Check if the text is related to medical/beauty domain."""
        # Extract words from the text
        words = re.findall(r'[\u0600-\u06FF\w]+', text.lower())
        
        if not words:
            return False
        
        # Check for explicit medical/beauty phrases first
        medical_phrases = [
            'حب الشباب', 'عيادة تجميل', 'طب تجميلي', 'عملية تجميل', 'علاج البشرة',
            'cosmetic surgery', 'plastic surgery', 'skin treatment', 'beauty clinic',
            'aesthetic medicine', 'dermatology'
        ]
        
        for phrase in medical_phrases:
            if phrase.lower() in text:
                return True
        
        # Count medical/beauty related words
        medical_word_count = 0
        for word in words:
            # Check the word as-is
            if word in self.allowed_medical_keywords:
                medical_word_count += 1
            # For Arabic words, also check without definite article prefix "ال"
            elif word.startswith('ال') and len(word) > 2:
                word_without_al = word[2:]  # Remove "ال" prefix
                if word_without_al in self.allowed_medical_keywords:
                    medical_word_count += 1
            # For Arabic words, also check without other common prefixes
            elif word.startswith('و') and len(word) > 1:  # "و" prefix (and)
                word_without_wa = word[1:]
                if word_without_wa in self.allowed_medical_keywords:
                    medical_word_count += 1
                # Also check without "ال" after removing "و"
                elif word_without_wa.startswith('ال') and len(word_without_wa) > 2:
                    word_clean = word_without_wa[2:]
                    if word_clean in self.allowed_medical_keywords:
                        medical_word_count += 1
        
        # Calculate relevance score
        relevance_ratio = medical_word_count / len(words) if words else 0
        
        # Adjust threshold based on strictness and text length
        if len(words) <= 5:  # Short texts need higher medical word density
            if self.strictness_level == "strict":
                threshold = 0.6  # 60% of words must be medical for short texts
            elif self.strictness_level == "balanced":
                threshold = 0.4  # 40% of words must be medical for short texts
            else:  # "relaxed"
                threshold = 0.2  # 20% of words must be medical for short texts (more lenient)
        else:  # Longer texts can have lower density
            if self.strictness_level == "strict":
                threshold = 0.3  # 30% of words must be medical
            elif self.strictness_level == "balanced":
                threshold = 0.15  # 15% of words must be medical (more lenient)
            else:  # "relaxed"
                threshold = 0.1  # 10% of words must be medical
        
        # For very short questions, be more lenient if they contain question words
        if len(words) <= 3:
            question_words = ['ما', 'كيف', 'متى', 'أين', 'لماذا', 'هل', 'what', 'how', 'when', 'where', 'why', 'is', 'are']
            has_question_word = any(word in question_words for word in words)
            if has_question_word and medical_word_count > 0:
                return True
        
        # Special case for beauty/medical facility inquiries
        clinic_words = ['عيادة', 'مركز', 'clinic', 'center']
        beauty_words = ['تجميل', 'جمال', 'beauty', 'cosmetic', 'aesthetic']
        service_words = ['خدمات', 'خدمة', 'services', 'service']
        
        has_clinic = any(any(clinic_word in word for clinic_word in clinic_words) for word in words)
        has_beauty = any(any(beauty_word in word for beauty_word in beauty_words) for word in words)
        has_service = any(any(service_word in word for service_word in service_words) for word in words)
        
        if has_clinic and (has_beauty or has_service):
            return True
        
        return relevance_ratio >= threshold
    
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
