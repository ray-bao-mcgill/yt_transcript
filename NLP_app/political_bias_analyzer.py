#!/usr/bin/env python3
"""
Political Bias Analysis for YouTube Transcripts
A comprehensive NLP-based approach to detect political bias using multiple techniques:
- Sentiment analysis and emotional markers
- Topic modeling and discourse analysis
- Linguistic markers and framing analysis
- Contextual analysis and semantic similarity
- Source credibility and fact-checking indicators
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    import spacy
    from spacy.tokens import Doc, Token
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with: pip install textblob")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import AutoModelForTokenClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install transformers torch")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import nltk
    from nltk.corpus import stopwords, subjectivity
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Install with: pip install nltk")

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_transcript import get_transcript


class PoliticalBiasAnalyzer:
    """
    Comprehensive political bias analysis using multiple NLP techniques
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the analyzer with required models and data"""
        self.nlp = None
        self.sentiment_analyzer = None
        self.topic_model = None
        self.vectorizer = None
        
        # Political bias indicators
        self.bias_indicators = self._load_bias_indicators()
        
        # Initialize NLP models
        self._initialize_models(model_name)
        
        # Analysis results storage
        self.results = {}
        
    def _load_bias_indicators(self) -> Dict:
        """Load comprehensive bias indicators and linguistic markers"""
        return {
            # Emotional and subjective language markers
            'emotional_markers': {
                'intense_emotions': ['outrageous', 'disgusting', 'shocking', 'terrifying', 'amazing', 'incredible', 'horrible', 'wonderful', 'fantastic', 'awful'],
                'subjective_adjectives': ['obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely', 'absolutely', 'completely', 'totally'],
                'loaded_language': ['radical', 'extreme', 'dangerous', 'threat', 'crisis', 'scandal', 'corrupt', 'evil', 'good', 'bad', 'wrong', 'right', 'terrible', 'excellent'],
                'us_vs_them': ['they', 'them', 'those people', 'the other side', 'opposition', 'enemy', 'opponent', 'liberals', 'conservatives', 'democrats', 'republicans'],
                'authority_claims': ['experts say', 'studies show', 'research proves', 'scientists agree', 'data shows', 'statistics prove', 'everyone knows']
            },
            
            # Political framing indicators
            'framing_indicators': {
                'economic_framing': ['economy', 'jobs', 'taxes', 'business', 'market', 'growth', 'recession', 'inflation', 'unemployment', 'wealth', 'poverty'],
                'moral_framing': ['values', 'morals', 'ethics', 'right', 'wrong', 'good', 'evil', 'sin', 'virtue', 'family', 'tradition'],
                'security_framing': ['security', 'safety', 'protection', 'defense', 'threat', 'danger', 'terrorism', 'crime', 'law', 'order'],
                'social_framing': ['community', 'society', 'people', 'citizens', 'public', 'social', 'welfare', 'healthcare', 'education'],
                'political_framing': ['democracy', 'freedom', 'rights', 'government', 'policy', 'election', 'vote', 'campaign', 'politician', 'party']
            },
            
            # Bias detection patterns
            'bias_patterns': {
                'one_sided_arguments': ['always', 'never', 'everyone knows', 'nobody believes', 'everyone agrees', 'no one disputes', 'clearly', 'obviously'],
                'straw_man': ['some people say', 'critics claim', 'opponents argue', 'they say', 'liberals think', 'conservatives believe'],
                'false_equivalence': ['both sides', 'equally', 'same thing', 'no difference', 'just as bad', 'just as good'],
                'confirmation_bias': ['proves my point', 'as expected', 'just as I said', 'I told you so', 'this confirms'],
                'appeal_to_authority': ['expert', 'scientist', 'study', 'research', 'data', 'statistics', 'official', 'authority']
            },
            
            # Political content indicators
            'political_content': {
                'political_terms': ['politics', 'political', 'government', 'administration', 'congress', 'senate', 'president', 'election', 'campaign', 'vote', 'voting'],
                'partisan_terms': ['democrat', 'republican', 'liberal', 'conservative', 'left', 'right', 'progressive', 'traditional'],
                'policy_terms': ['policy', 'legislation', 'bill', 'law', 'regulation', 'reform', 'change', 'agenda'],
                'controversial_topics': ['abortion', 'guns', 'immigration', 'climate', 'healthcare', 'taxes', 'welfare', 'defense', 'foreign policy']
            },
            
            # Left-leaning indicators
            'left_indicators': {
                'progressive_terms': ['progressive', 'liberal', 'democrat', 'left', 'socialist', 'socialism', 'equality', 'equity', 'diversity', 'inclusion', 'social justice'],
                'left_policies': ['universal healthcare', 'medicare for all', 'green new deal', 'climate action', 'renewable energy', 'minimum wage', 'workers rights', 'union', 'social security'],
                'left_framing': ['systemic', 'privilege', 'oppression', 'marginalized', 'underrepresented', 'social welfare', 'public education', 'environmental protection'],
                'left_criticism': ['capitalism', 'corporations', 'big business', 'wealth inequality', 'income gap', 'tax cuts for rich', 'trickle down', 'deregulation']
            },
            
            # Right-leaning indicators
            'right_indicators': {
                'conservative_terms': ['conservative', 'republican', 'right', 'traditional', 'patriotic', 'american values', 'family values', 'religious', 'christian'],
                'right_policies': ['free market', 'deregulation', 'tax cuts', 'small government', 'defense spending', 'border security', 'law and order', 'second amendment', 'pro life'],
                'right_framing': ['personal responsibility', 'individual liberty', 'free enterprise', 'national security', 'traditional marriage', 'religious freedom'],
                'right_criticism': ['socialism', 'big government', 'welfare state', 'entitlements', 'liberal media', 'fake news', 'deep state', 'globalism']
            },
            
            # Source credibility indicators
            'credibility_indicators': {
                'factual_claims': ['according to', 'data shows', 'statistics indicate', 'study found', 'research shows', 'evidence suggests'],
                'opinion_markers': ['I think', 'I believe', 'in my opinion', 'it seems to me', 'I feel', 'I would say'],
                'uncertainty_markers': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems like', 'appears to'],
                'qualifiers': ['some', 'many', 'most', 'often', 'sometimes', 'rarely', 'usually', 'generally']
            }
        }
    
    def _initialize_models(self, model_name: str):
        """Initialize NLP models and pipelines"""
        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                print(f"Loaded spaCy model: {model_name}")
            except OSError:
                print(f"spaCy model {model_name} not found. Please install with: python -m spacy download {model_name}")
                self.nlp = None
        
        # Initialize sentiment analyzer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("Loaded sentiment analyzer")
            except Exception as e:
                print(f"Could not load sentiment analyzer: {e}")
                self.sentiment_analyzer = None
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('subjectivity', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                print("Initialized NLTK components")
            except Exception as e:
                print(f"Could not initialize NLTK: {e}")
        
        # Initialize topic modeling
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            print("Initialized topic modeling components")
    
    def analyze_transcript(self, transcript_text: str, video_url: str = None) -> Dict:
        """
        Perform comprehensive political bias analysis on transcript
        
        Args:
            transcript_text: The transcript text to analyze
            video_url: Optional video URL for context
            
        Returns:
            Dictionary containing comprehensive bias analysis results
        """
        if not transcript_text or len(transcript_text.strip()) < 50:
            return {"error": "Transcript too short or empty for meaningful analysis"}
        
        print("Starting comprehensive political bias analysis...")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(transcript_text)
        
        # Perform various analyses
        results = {
            "video_url": video_url,
            "transcript_length": len(transcript_text),
            "cleaned_length": len(cleaned_text),
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            
            # Core analyses
            "sentiment_analysis": self._analyze_sentiment(cleaned_text),
            "linguistic_markers": self._analyze_linguistic_markers(cleaned_text),
            "topic_analysis": self._analyze_topics(cleaned_text),
            "framing_analysis": self._analyze_framing(cleaned_text),
            "credibility_analysis": self._analyze_credibility(cleaned_text),
            "contextual_analysis": self._analyze_context(cleaned_text),
            
            # Overall bias assessment
            "bias_assessment": {},
            "recommendations": []
        }
        
        # Calculate overall bias score
        results["bias_assessment"] = self._calculate_overall_bias(results)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        self.results = results
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)  # Remove special characters
        text = text.lower().strip()
        
        return text
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and emotional content"""
        results = {
            "overall_sentiment": "neutral",
            "sentiment_scores": {},
            "emotional_intensity": 0.0,
            "subjectivity_score": 0.0,
            "sentiment_distribution": {}
        }
        
        # TextBlob sentiment analysis
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            results["sentiment_scores"]["textblob"] = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            results["subjectivity_score"] = blob.sentiment.subjectivity
        
        # Transformers sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
                sentence_sentiments = []
                
                for sentence in sentences[:50]:  # Limit to avoid memory issues
                    if len(sentence.strip()) > 10:
                        sentiment_result = self.sentiment_analyzer(sentence)
                        sentence_sentiments.append(sentiment_result[0])
                
                if sentence_sentiments:
                    # Aggregate sentiment scores
                    avg_scores = defaultdict(float)
                    for result in sentence_sentiments:
                        for score in result:
                            avg_scores[score['label']] += score['score']
                    
                    for label in avg_scores:
                        avg_scores[label] /= len(sentence_sentiments)
                    
                    results["sentiment_scores"]["transformers"] = dict(avg_scores)
                    
                    # Determine overall sentiment
                    if avg_scores.get('positive', 0) > avg_scores.get('negative', 0):
                        results["overall_sentiment"] = "positive"
                    elif avg_scores.get('negative', 0) > avg_scores.get('positive', 0):
                        results["overall_sentiment"] = "negative"
            except Exception as e:
                print(f"Error in transformers sentiment analysis: {e}")
        
        # Calculate emotional intensity
        emotional_words = self.bias_indicators['emotional_markers']['intense_emotions']
        emotional_count = sum(1 for word in emotional_words if word in text.lower())
        results["emotional_intensity"] = emotional_count / len(text.split()) * 1000  # Per 1000 words
        
        return results
    
    def _analyze_linguistic_markers(self, text: str) -> Dict:
        """Analyze linguistic markers that indicate bias"""
        results = {
            "loaded_language_count": 0,
            "subjective_markers": 0,
            "authority_claims": 0,
            "us_vs_them_phrases": 0,
            "one_sided_arguments": 0,
            "straw_man_arguments": 0,
            "false_equivalence": 0,
            "confirmation_bias": 0,
            "appeal_to_authority": 0,
            "political_content_score": 0,
            "partisan_language": 0,
            "political_topics_detected": [],
            "controversial_topics_detected": [],
            "left_indicators_count": 0,
            "right_indicators_count": 0,
            "left_indicators_detected": [],
            "right_indicators_detected": [],
            "contributing_words": [],
            "bias_contributors": {}
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count various linguistic markers (raw counts, not normalized)
        for category, markers in self.bias_indicators['emotional_markers'].items():
            if category == 'loaded_language':
                detected = [word for word in words if word in markers]
                results['loaded_language_count'] = len(detected)
                results['contributing_words'].extend(detected)
            elif category == 'subjective_adjectives':
                detected = [word for word in words if word in markers]
                results['subjective_markers'] = len(detected)
                results['contributing_words'].extend(detected)
            elif category == 'authority_claims':
                detected = [phrase for phrase in markers if phrase in text_lower]
                results['authority_claims'] = len(detected)
                results['contributing_words'].extend(detected)
            elif category == 'us_vs_them':
                detected = [word for word in words if word in markers]
                results['us_vs_them_phrases'] = len(detected)
                results['contributing_words'].extend(detected)
        
        # Count bias patterns (raw counts)
        for pattern, phrases in self.bias_indicators['bias_patterns'].items():
            detected = [phrase for phrase in phrases if phrase in text_lower]
            if pattern == 'one_sided_arguments':
                results['one_sided_arguments'] = len(detected)
            elif pattern == 'straw_man':
                results['straw_man_arguments'] = len(detected)
            elif pattern == 'false_equivalence':
                results['false_equivalence'] = len(detected)
            elif pattern == 'confirmation_bias':
                results['confirmation_bias'] = len(detected)
            elif pattern == 'appeal_to_authority':
                results['appeal_to_authority'] = len(detected)
            results['contributing_words'].extend(detected)
        
        # Analyze political content (raw counts and topic detection)
        political_score = 0
        partisan_score = 0
        political_topics = []
        controversial_topics = []
        
        for category, terms in self.bias_indicators['political_content'].items():
            detected_terms = [word for word in words if word in terms]
            count = len(detected_terms)
            
            if category == 'partisan_terms':
                partisan_score += count
                political_topics.extend(detected_terms)
            elif category == 'political_terms':
                political_score += count
                political_topics.extend(detected_terms)
            elif category == 'policy_terms':
                political_score += count
                political_topics.extend(detected_terms)
            elif category == 'controversial_topics':
                political_score += count
                controversial_topics.extend(detected_terms)
            
            results['contributing_words'].extend(detected_terms)
        
        # Analyze left-leaning indicators
        left_score = 0
        left_indicators = []
        for category, terms in self.bias_indicators['left_indicators'].items():
            detected_terms = [word for word in words if word in terms]
            count = len(detected_terms)
            left_score += count
            left_indicators.extend(detected_terms)
            results['contributing_words'].extend(detected_terms)
            results['bias_contributors'][f'left_{category}'] = detected_terms
        
        # Analyze right-leaning indicators
        right_score = 0
        right_indicators = []
        for category, terms in self.bias_indicators['right_indicators'].items():
            detected_terms = [word for word in words if word in terms]
            count = len(detected_terms)
            right_score += count
            right_indicators.extend(detected_terms)
            results['contributing_words'].extend(detected_terms)
            results['bias_contributors'][f'right_{category}'] = detected_terms
        
        results['political_content_score'] = political_score
        results['partisan_language'] = partisan_score
        results['political_topics_detected'] = list(set(political_topics))  # Remove duplicates
        results['controversial_topics_detected'] = list(set(controversial_topics))
        results['left_indicators_count'] = left_score
        results['right_indicators_count'] = right_score
        results['left_indicators_detected'] = list(set(left_indicators))
        results['right_indicators_detected'] = list(set(right_indicators))
        results['contributing_words'] = list(set(results['contributing_words']))  # Remove duplicates
        
        return results
    
    def _analyze_topics(self, text: str) -> Dict:
        """Analyze topics and themes in the text"""
        results = {
            "main_topics": [],
            "topic_diversity": 0.0,
            "political_topics": [],
            "controversial_topics": []
        }
        
        if not SKLEARN_AVAILABLE:
            return results
        
        try:
            # Extract sentences for topic modeling
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 3:
                return results
            
            # Create document-term matrix
            dtm = self.vectorizer.fit_transform(sentences)
            
            # Topic modeling with LDA
            lda = LatentDirichletAllocation(n_components=min(5, len(sentences)), random_state=42)
            lda.fit(dtm)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:]]
                topics.append({
                    "topic_id": topic_idx,
                    "top_words": top_words,
                    "weight": topic.max()
                })
            
            results["main_topics"] = topics
            
            # Calculate topic diversity
            topic_distribution = lda.transform(dtm)
            results["topic_diversity"] = np.std(topic_distribution).mean()
            
            # Identify political and controversial topics
            political_keywords = ['politics', 'government', 'election', 'vote', 'democrat', 'republican', 
                                'policy', 'law', 'congress', 'senate', 'president', 'campaign']
            
            controversial_keywords = ['controversy', 'scandal', 'corruption', 'protest', 'riot', 
                                    'violence', 'crime', 'investigation', 'allegation']
            
            for topic in topics:
                topic_text = ' '.join(topic['top_words']).lower()
                if any(keyword in topic_text for keyword in political_keywords):
                    results["political_topics"].append(topic)
                if any(keyword in topic_text for keyword in controversial_keywords):
                    results["controversial_topics"].append(topic)
                    
        except Exception as e:
            print(f"Error in topic analysis: {e}")
        
        return results
    
    def _analyze_framing(self, text: str) -> Dict:
        """Analyze how topics are framed in the text"""
        results = {
            "framing_categories": {},
            "dominant_frames": [],
            "frame_balance": 0.0
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count framing indicators
        for frame_type, keywords in self.bias_indicators['framing_indicators'].items():
            count = sum(1 for word in words if word in keywords)
            results["framing_categories"][frame_type] = count
        
        # Normalize by text length
        word_count = len(words)
        for frame_type in results["framing_categories"]:
            results["framing_categories"][frame_type] /= word_count * 1000
        
        # Identify dominant frames
        sorted_frames = sorted(results["framing_categories"].items(), 
                             key=lambda x: x[1], reverse=True)
        results["dominant_frames"] = [frame[0] for frame in sorted_frames[:3]]
        
        # Calculate frame balance (how evenly distributed frames are)
        frame_values = list(results["framing_categories"].values())
        if frame_values:
            results["frame_balance"] = 1 - (max(frame_values) / sum(frame_values) if sum(frame_values) > 0 else 0)
        
        return results
    
    def _analyze_credibility(self, text: str) -> Dict:
        """Analyze source credibility and fact-checking indicators"""
        results = {
            "factual_claims": 0,
            "opinion_markers": 0,
            "uncertainty_markers": 0,
            "qualifiers": 0,
            "credibility_score": 0.0
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count credibility indicators
        for indicator_type, markers in self.bias_indicators['credibility_indicators'].items():
            if indicator_type == 'factual_claims':
                results['factual_claims'] = sum(1 for phrase in markers if phrase in text_lower)
            elif indicator_type == 'opinion_markers':
                results['opinion_markers'] = sum(1 for phrase in markers if phrase in text_lower)
            elif indicator_type == 'uncertainty_markers':
                results['uncertainty_markers'] = sum(1 for word in words if word in markers)
            elif indicator_type == 'qualifiers':
                results['qualifiers'] = sum(1 for word in words if word in markers)
        
        # Calculate credibility score
        total_indicators = sum(results.values())
        if total_indicators > 0:
            factual_weight = results['factual_claims'] * 2  # Positive weight
            opinion_weight = results['opinion_markers'] * -1  # Negative weight
            uncertainty_weight = results['uncertainty_markers'] * 0.5  # Slight positive
            qualifier_weight = results['qualifiers'] * 0.5  # Slight positive
            
            results['credibility_score'] = (factual_weight + opinion_weight + 
                                          uncertainty_weight + qualifier_weight) / total_indicators
        
        return results
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze contextual factors and semantic relationships"""
        results = {
            "sentence_complexity": 0.0,
            "vocabulary_diversity": 0.0,
            "argument_structure": "unclear",
            "contextual_consistency": 0.0
        }
        
        if not NLTK_AVAILABLE:
            return results
        
        try:
            # Analyze sentence complexity
            sentences = sent_tokenize(text)
            if sentences:
                avg_sentence_length = np.mean([len(s.split()) for s in sentences])
                results["sentence_complexity"] = avg_sentence_length
            
            # Analyze vocabulary diversity
            words = word_tokenize(text.lower())
            unique_words = set(words)
            results["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
            
            # Analyze argument structure
            argument_indicators = ['because', 'therefore', 'however', 'although', 'but', 'so']
            argument_count = sum(1 for word in words if word in argument_indicators)
            
            if argument_count > len(sentences) * 0.3:
                results["argument_structure"] = "well_structured"
            elif argument_count > len(sentences) * 0.1:
                results["argument_structure"] = "moderately_structured"
            else:
                results["argument_structure"] = "poorly_structured"
                
        except Exception as e:
            print(f"Error in contextual analysis: {e}")
        
        return results
    
    def _calculate_overall_bias(self, analysis_results: Dict) -> Dict:
        """Calculate overall bias assessment based on political content and presentation"""
        bias_score = 0.0
        bias_factors = []
        
        # Step 1: Analyze Political Content
        linguistic = analysis_results.get('linguistic_markers', {})
        political_content = linguistic.get('political_content_score', 0)
        partisan_language = linguistic.get('partisan_language', 0)
        controversial_topics = len(linguistic.get('controversial_topics_detected', []))
        
        # Analyze left/right leaning
        left_indicators = linguistic.get('left_indicators_count', 0)
        right_indicators = linguistic.get('right_indicators_count', 0)
        left_words = linguistic.get('left_indicators_detected', [])
        right_words = linguistic.get('right_indicators_detected', [])
        
        # Determine political leaning
        political_leaning = "Neutral"
        leaning_strength = 0.0
        if left_indicators > right_indicators and left_indicators > 0:
            political_leaning = "Left-leaning"
            leaning_strength = (left_indicators - right_indicators) / max(left_indicators, 1)
        elif right_indicators > left_indicators and right_indicators > 0:
            political_leaning = "Right-leaning"
            leaning_strength = (right_indicators - left_indicators) / max(right_indicators, 1)
        elif left_indicators == right_indicators and left_indicators > 0:
            political_leaning = "Mixed/Partisan"
            leaning_strength = 0.5
        
        # Determine if content is political
        is_political_content = political_content > 0 or partisan_language > 0 or controversial_topics > 0
        
        if not is_political_content:
            # If no political content, bias is minimal
            return {
                "overall_bias_score": 0.05,  # Very low bias for non-political content
                "bias_level": "Low",
                "bias_factors": ["Non-political content"],
                "political_leaning": "Neutral",
                "leaning_strength": 0.0,
                "contributing_words": linguistic.get('contributing_words', []),
                "confidence": "high"
            }
        
        # Step 2: Analyze Presentation (How political content is presented)
        sentiment = analysis_results.get('sentiment_analysis', {})
        subjectivity = sentiment.get('subjectivity_score', 0)
        loaded_language = linguistic.get('loaded_language_count', 0)
        subjective_markers = linguistic.get('subjective_markers', 0)
        us_vs_them = linguistic.get('us_vs_them_phrases', 0)
        
        # Calculate presentation bias
        presentation_bias = 0.0
        
        # High subjectivity indicates bias
        if subjectivity > 0.7:
            presentation_bias += 0.4
            bias_factors.append("Very high subjectivity")
        elif subjectivity > 0.5:
            presentation_bias += 0.2
            bias_factors.append("High subjectivity")
        elif subjectivity > 0.3:
            presentation_bias += 0.1
            bias_factors.append("Moderate subjectivity")
        
        # Loaded language indicates bias
        if loaded_language > 5:
            presentation_bias += 0.3
            bias_factors.append("Heavy use of loaded language")
        elif loaded_language > 2:
            presentation_bias += 0.15
            bias_factors.append("Moderate use of loaded language")
        elif loaded_language > 0:
            presentation_bias += 0.05
            bias_factors.append("Some loaded language")
        
        # Subjective markers indicate bias
        if subjective_markers > 3:
            presentation_bias += 0.2
            bias_factors.append("Heavy use of subjective markers")
        elif subjective_markers > 1:
            presentation_bias += 0.1
            bias_factors.append("Moderate use of subjective markers")
        
        # Us vs them language indicates bias
        if us_vs_them > 2:
            presentation_bias += 0.25
            bias_factors.append("Divisive us vs them language")
        elif us_vs_them > 0:
            presentation_bias += 0.1
            bias_factors.append("Some divisive language")
        
        # Step 3: Analyze Political Intensity
        political_intensity = 0.0
        
        # Partisan language is a strong indicator
        if partisan_language > 3:
            political_intensity += 0.4
            bias_factors.append("Heavy partisan language")
        elif partisan_language > 1:
            political_intensity += 0.2
            bias_factors.append("Moderate partisan language")
        elif partisan_language > 0:
            political_intensity += 0.1
            bias_factors.append("Some partisan language")
        
        # Controversial topics increase intensity
        if controversial_topics > 2:
            political_intensity += 0.2
            bias_factors.append("Multiple controversial topics")
        elif controversial_topics > 0:
            political_intensity += 0.1
            bias_factors.append("Controversial topics present")
        
        # Political content volume
        if political_content > 10:
            political_intensity += 0.2
            bias_factors.append("High political content")
        elif political_content > 5:
            political_intensity += 0.1
            bias_factors.append("Moderate political content")
        
        # Step 4: Calculate Final Bias Score
        # Bias = Political Intensity × Presentation Bias
        if political_intensity > 0 and presentation_bias > 0:
            bias_score = min(political_intensity * presentation_bias * 2, 1.0)  # Scale and cap at 1.0
        else:
            bias_score = max(political_intensity, presentation_bias)
        
        # Ensure minimum bias for political content
        if is_political_content and bias_score < 0.1:
            bias_score = 0.1
            if not bias_factors:
                bias_factors.append("Political content detected")
        
        # Determine bias level
        if bias_score < 0.2:
            bias_level = "Low"
        elif bias_score < 0.5:
            bias_level = "Moderate"
        elif bias_score < 0.8:
            bias_level = "High"
        else:
            bias_level = "Very High"
        
        return {
            "overall_bias_score": bias_score,
            "bias_level": bias_level,
            "bias_factors": bias_factors,
            "political_content_detected": is_political_content,
            "political_intensity": political_intensity,
            "presentation_bias": presentation_bias,
            "political_leaning": political_leaning,
            "leaning_strength": leaning_strength,
            "left_indicators": left_indicators,
            "right_indicators": right_indicators,
            "left_words": left_words,
            "right_words": right_words,
            "contributing_words": linguistic.get('contributing_words', []),
            "bias_contributors": linguistic.get('bias_contributors', {}),
            "confidence": "high" if is_political_content else "medium"
        }
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        bias_assessment = analysis_results.get('bias_assessment', {})
        bias_level = bias_assessment.get('bias_level', 'Unknown')
        
        if bias_level in ['High', 'Very High']:
            recommendations.append("Consider seeking multiple perspectives on this topic")
            recommendations.append("Verify factual claims with independent sources")
            recommendations.append("Be aware of potential emotional manipulation")
        
        linguistic = analysis_results.get('linguistic_markers', {})
        if linguistic.get('loaded_language_count', 0) > 5:
            recommendations.append("High use of loaded language detected - consider more neutral language")
        
        if linguistic.get('one_sided_arguments', 0) > 3:
            recommendations.append("One-sided arguments detected - consider opposing viewpoints")
        
        credibility = analysis_results.get('credibility_analysis', {})
        if credibility.get('credibility_score', 0) < -0.2:
            recommendations.append("Low credibility indicators - verify sources and claims")
        
        if not recommendations:
            recommendations.append("Content appears relatively balanced - still verify important claims")
        
        return recommendations
    
    def generate_report(self, results: Dict = None) -> str:
        """Generate a human-readable report from analysis results"""
        if results is None:
            results = self.results
        
        if not results:
            return "No analysis results available"
        
        report = []
        report.append("=" * 60)
        report.append("POLITICAL BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Basic info
        if results.get('video_url'):
            report.append(f"Video URL: {results['video_url']}")
        report.append(f"Transcript Length: {results.get('transcript_length', 0)} characters")
        report.append(f"Analysis Date: {results.get('analysis_timestamp', 'Unknown')}")
        report.append("")
        
        # Overall bias assessment
        bias_assessment = results.get('bias_assessment', {})
        report.append("OVERALL BIAS ASSESSMENT:")
        report.append(f"Bias Level: {bias_assessment.get('bias_level', 'Unknown')}")
        report.append(f"Bias Score: {bias_assessment.get('overall_bias_score', 0):.2f}")
        report.append(f"Confidence: {bias_assessment.get('confidence', 'Unknown')}")
        report.append(f"Political Leaning: {bias_assessment.get('political_leaning', 'Unknown')}")
        report.append(f"Leaning Strength: {bias_assessment.get('leaning_strength', 0):.2f}")
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS:")
        bias_factors = bias_assessment.get('bias_factors', [])
        if bias_factors:
            for factor in bias_factors:
                report.append(f"• {factor}")
        else:
            report.append("• No significant bias indicators detected")
        report.append("")
        
        # Political leaning breakdown
        if bias_assessment.get('political_leaning') != 'Neutral':
            report.append("POLITICAL LEANING BREAKDOWN:")
            report.append(f"Left Indicators: {bias_assessment.get('left_indicators', 0)}")
            report.append(f"Right Indicators: {bias_assessment.get('right_indicators', 0)}")
            if bias_assessment.get('left_words'):
                report.append(f"Left-leaning Words: {', '.join(bias_assessment['left_words'])}")
            if bias_assessment.get('right_words'):
                report.append(f"Right-leaning Words: {', '.join(bias_assessment['right_words'])}")
            report.append("")
        
        # Contributing words
        if bias_assessment.get('contributing_words'):
            report.append("BIAS CONTRIBUTING WORDS:")
            contributing_words = bias_assessment.get('contributing_words', [])
            report.append(f"Total: {len(contributing_words)} words")
            report.append(f"Key Words: {', '.join(contributing_words[:15])}")
            if len(contributing_words) > 15:
                report.append(f"... and {len(contributing_words) - 15} more")
            report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS:")
        
        # Sentiment
        sentiment = results.get('sentiment_analysis', {})
        report.append(f"Sentiment: {sentiment.get('overall_sentiment', 'Unknown')}")
        report.append(f"Subjectivity: {sentiment.get('subjectivity_score', 0):.2f}")
        report.append(f"Emotional Intensity: {sentiment.get('emotional_intensity', 0):.1f} per 1000 words")
        report.append("")
        
        # Linguistic markers
        linguistic = results.get('linguistic_markers', {})
        report.append("LINGUISTIC MARKERS (per 1000 words):")
        report.append(f"Loaded Language: {linguistic.get('loaded_language_count', 0):.1f}")
        report.append(f"Subjective Markers: {linguistic.get('subjective_markers', 0):.1f}")
        report.append(f"Authority Claims: {linguistic.get('authority_claims', 0):.1f}")
        report.append("")
        
        # Topics
        topics = results.get('topic_analysis', {})
        if topics.get('main_topics'):
            report.append("MAIN TOPICS:")
            for topic in topics['main_topics'][:3]:
                report.append(f"• {', '.join(topic['top_words'][:5])}")
            report.append("")
        
        # Framing
        framing = results.get('framing_analysis', {})
        if framing.get('dominant_frames'):
            report.append("DOMINANT FRAMES:")
            for frame in framing['dominant_frames']:
                report.append(f"• {frame.replace('_', ' ').title()}")
            report.append("")
        
        # Credibility
        credibility = results.get('credibility_analysis', {})
        report.append("CREDIBILITY INDICATORS:")
        report.append(f"Credibility Score: {credibility.get('credibility_score', 0):.2f}")
        report.append(f"Factual Claims: {credibility.get('factual_claims', 0)}")
        report.append(f"Opinion Markers: {credibility.get('opinion_markers', 0)}")
        report.append("")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"• {rec}")
            report.append("")
        
        report.append("=" * 60)
        report.append("End of Report")
        report.append("=" * 60)
        
        return "\n".join(report)


def analyze_youtube_video(video_url: str) -> Dict:
    """
    Convenience function to analyze a YouTube video for political bias
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Analysis results dictionary
    """
    # Get transcript
    print(f"Extracting transcript from: {video_url}")
    transcript_result = get_transcript(video_url)
    
    if not transcript_result.get("success"):
        return {"error": f"Failed to get transcript: {transcript_result.get('error', 'Unknown error')}"}
    
    transcript_text = transcript_result.get("transcript_text", "")
    if not transcript_text:
        return {"error": "No transcript text available"}
    
    # Analyze for political bias
    analyzer = PoliticalBiasAnalyzer()
    results = analyzer.analyze_transcript(transcript_text, video_url)
    
    # Add the transcript text to the results
    results["transcript_text"] = transcript_text
    
    return results


def main():
    """Main function for command-line usage"""
    print("Political Bias Analysis for YouTube Transcripts")
    print("=" * 50)
    
    # Get video URL
    video_url = input("Enter YouTube URL: ").strip()
    
    if not video_url:
        print("No URL provided. Exiting.")
        return
    
    # Analyze the video
    results = analyze_youtube_video(video_url)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Generate and display report
    analyzer = PoliticalBiasAnalyzer()
    report = analyzer.generate_report(results)
    print("\n" + report)
    
    # Save results to file
    output_file = f"bias_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main() 