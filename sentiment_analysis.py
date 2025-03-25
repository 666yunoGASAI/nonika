import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
import re
import string
import emoji
import joblib
import os
from pathlib import Path

# Download required NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Enhanced emoji sentiment dictionary (expanded)
EMOJI_SENTIMENT = {
    # Highly positive
    "😍": 1.0, "🥰": 1.0, "❤️": 0.9, "😁": 0.9, "🔥": 0.8, 
    "💯": 0.9, "✨": 0.7, "🙏": 0.7, "👑": 0.8, "🎉": 0.9,
    
    # Moderately positive
    "👍": 0.7, "😊": 0.7, "🤣": 0.8, "😂": 0.7, "😆": 0.6,
    "👏": 0.6, "🥳": 0.8, "🤩": 0.9, "😎": 0.6, "💪": 0.7,
    
    # Slightly positive
    "👌": 0.4, "🙂": 0.3, "😉": 0.3, "😄": 0.5, "☺️": 0.4,
    "😃": 0.5, "🤭": 0.3, "💕": 0.6, "💓": 0.6, "💖": 0.6,
    
    # Neutral
    "🤔": 0.0, "😐": 0.0, "🙄": -0.1, "😶": 0.0, "🤷": 0.0,
    "⭐": 0.2, "📱": 0.0, "📸": 0.1, "🎵": 0.1, "🤨": -0.1,
    
    # Slightly negative
    "😒": -0.4, "😕": -0.3, "😟": -0.4, "😬": -0.3, "😧": -0.4,
    "😓": -0.4, "😨": -0.5, "😥": -0.4, "😔": -0.4, "🙁": -0.3,
    
    # Moderately negative
    "😢": -0.6, "😭": -0.6, "😡": -0.7, "👎": -0.7, "😠": -0.6,
    "😤": -0.6, "😩": -0.6, "💔": -0.7, "😫": -0.6, "😖": -0.5,
    
    # Highly negative
    "😱": -0.8, "🤬": -0.9, "😈": -0.7, "💀": -0.7, "🤢": -0.8,
    "🤮": -0.9, "😷": -0.6, "🙅": -0.7, "👿": -0.8, "🤡": -0.5,
}

# TikTok-specific sentiment lexicon
TIKTOK_LEXICON = {
    # Positive TikTok slang
    'slay': 0.8, 'fire': 0.7, 'lit': 0.7, 'goated': 0.9, 'based': 0.7,
    'bussin': 0.8, 'valid': 0.6, 'vibes': 0.5, 'iconic': 0.8, 'ate': 0.7,
    'fax': 0.5, 'facts': 0.5, 'bet': 0.4, 'fyp': 0.3, 'foryou': 0.3,
    'relatable': 0.5, 'talented': 0.7, 'queen': 0.7, 'king': 0.7, 'legend': 0.8,
    'periodt': 0.6, 'period': 0.6, 'win': 0.7, 'viral': 0.4, 'trending': 0.4,
    'clean': 0.5, 'chef kiss': 0.8, 'no cap': 0.3, 'sheesh': 0.7, 'glowing': 0.6,
    'stan': 0.5, 'vibe check': 0.4, 'rent free': 0.3, 'main character': 0.6,
    'elite': 0.7, 'chef\'s kiss': 0.8, 'baddie': 0.6, 'gem': 0.7, 'masterpiece': 0.9,
    
    # Negative TikTok slang
    'cringe': -0.7, 'flop': -0.8, 'mid': -0.5, 'ratio': -0.6, 'dead': -0.4,
    'basic': -0.5, 'ick': -0.7, 'yikes': -0.6, 'cap': -0.5, 'sus': -0.4,
    'cancel': -0.7, 'clickbait': -0.6, 'toxic': -0.7, 'cheugy': -0.5, 'copium': -0.4,
    'cursed': -0.6, 'clout chasing': -0.7, 'fake': -0.7, 'fraud': -0.8, 'scam': -0.8,
    'disappointing': -0.7, 'trash': -0.8, 'nightmare': -0.7, 'unfollow': -0.6, 'worst': -0.8,
    'shadowban': -0.7, 'shadow ban': -0.7, 'triggered': -0.5, 'cancelled': -0.7,
    'flopped': -0.8, 'overrated': -0.6, 'boring': -0.6, 'annoying': -0.7, 'wtf': -0.6
}

# Function to preprocess text for sentiment analysis
def preprocess_for_sentiment(text):
    """
    Preprocess text specifically for sentiment analysis, preserving emoticons and key phrases.
    """
    if not isinstance(text, str):
        return {"processed_text": "", "emojis": "", "demojized_text": ""}
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    
    # Replace user mentions with token
    text = re.sub(r'@\w+', ' USER ', text)
    
    # Extract and save emojis before modifying text
    emojis_found = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
    # Convert emojis to text representations
    text_with_emoji_names = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation except ! and ? which can indicate sentiment
    text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return {
        'processed_text': text,
        'emojis': emojis_found,
        'demojized_text': text_with_emoji_names
    }

# VADER Sentiment Analysis
def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER with optimizations for social media content.
    """
    if not text or not isinstance(text, str):
        return "Neutral (0.00)"
    
    sid = SentimentIntensityAnalyzer()
    
    # VADER works better with original capitalization and punctuation
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    
    # Adjust thresholds for social media content which tends to be more polarized
    if compound >= 0.05:
        return f"Positive ({compound:.2f})"
    elif compound <= -0.05:
        return f"Negative ({compound:.2f})"
    else:
        return f"Neutral ({compound:.2f})"

# Advanced Emoji Sentiment Analysis
def analyze_emoji_sentiment(emoji_text):
    """
    Analyze sentiment from emojis using a predefined dictionary.
    """
    if not emoji_text:
        return 0.0
    
    total_score = 0
    count = 0
    
    # Analyze each emoji in the text
    for char in emoji_text:
        if char in EMOJI_SENTIMENT:
            total_score += EMOJI_SENTIMENT[char]
            count += 1
    
    # If no known emojis were found
    if count == 0:
        return 0.0
        
    return total_score / count

def train_with_labeled_data():
    """
    Train sentiment model using manually labeled data.
    Returns the trained model or None if no labeled data exists.
    """
    try:
        # Try to load labeled data
        labeled_data = pd.read_csv('sentiment_corrections.csv')
        
        if len(labeled_data) < 10:
            # Not enough data yet
            return None
            
        # Get base training data
        base_texts, base_labels = generate_training_data()
        
        # Extract labeled comments and sentiments
        labeled_texts = labeled_data['Comment'].tolist()
        labeled_sentiments = labeled_data['Corrected_Sentiment'].tolist()
        
        # Combine base and labeled data
        all_texts = base_texts + labeled_texts
        all_labels = base_labels + labeled_sentiments
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        pipeline.fit(all_texts, all_labels)
        
        # Save the custom model
        try:
            joblib.dump(pipeline, "custom_sentiment_model.joblib")
        except:
            pass
            
        return pipeline
        
    except Exception as e:
        # If any error occurs (like file not found), return None
        print(f"Error training with labeled data: {e}")
        return None

# Advanced lexicon-based sentiment with TikTok-specific terms
def analyze_lexicon_sentiment(text):
    """
    Analyze sentiment using TikTok-specific lexicon.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Also check for multi-word phrases
    bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
    
    total_score = 0
    count = 0
    
    # Check for words
    for word in words:
        if word in TIKTOK_LEXICON:
            total_score += TIKTOK_LEXICON[word]
            count += 1
    
    # Check for phrases
    for phrase in bigrams:
        if phrase in TIKTOK_LEXICON:
            total_score += TIKTOK_LEXICON[phrase]
            count += 1
    
    # If no sentiment words were found
    if count == 0:
        return 0.0
        
    return total_score / count

# Generate enhanced training data for ML models
def generate_training_data():
    """
    Generate enhanced training data for sentiment analysis models.
    """
    # Positive examples with TikTok-specific vocabulary
    positive_examples = [
        "i love this", "this is amazing", "great video", "awesome content", "love it",
        "so good", "incredible", "fantastic", "perfect", "the best",
        "outstanding", "excellent", "wonderful", "brilliant", "fabulous",
        "impressive", "superb", "exceptional", "terrific", "top notch",
        "this is fire", "absolutely slaying", "you ate this up", "living for this content",
        "this is bussin fr", "goated", "no cap this is lit", "iconic", "this is so based",
        "vibes are immaculate", "period queen", "talented af", "true masterpiece",
        "obsessed with this", "chef's kiss", "rent free in my mind", "sheesh",
        "main character energy", "elite content", "pop off", "stan forever"
    ]
    
    # Negative examples with TikTok-specific vocabulary
    negative_examples = [
        "i hate this", "this is terrible", "awful video", "bad content", "dislike it",
        "so bad", "horrible", "disappointing", "worst ever", "waste of time",
        "useless", "pathetic", "terrible", "awful", "dreadful",
        "poor quality", "unbearable", "rubbish", "lame", "disgusting",
        "this is cringe", "major flop", "mid at best", "giving me the ick",
        "big yikes", "that's cap", "kinda sus", "toxic behavior", "clickbait",
        "basic content", "cursed content", "nightmare fuel", "so fake",
        "actually delusional", "flopped hard", "pure trash", "annoying af",
        "get this off my fyp", "make it stop", "instant unfollow"
    ]
    
    # Neutral examples with TikTok-specific vocabulary
    neutral_examples = [
        "okay", "not sure", "maybe", "average", "alright",
        "not bad", "so so", "ordinary", "standard", "mediocre",
        "fair", "tolerable", "passable", "reasonable", "moderate",
        "neither good nor bad", "acceptable", "adequate", "middle of the road", "nothing special",
        "just scrolling", "pov: me watching", "idk about this", "no thoughts",
        "for legal reasons that's a joke", "here before it blows up", "algorithm bring me back",
        "wondering if", "anyone else notice", "first time seeing this", "interesting concept",
        "what's the song", "trying to understand", "need more context", "still processing",
        "commenting for the algorithm", "the algorithm blessed me"
    ]
    
    # Create training data
    texts = positive_examples + negative_examples + neutral_examples
    labels = (["Positive"] * len(positive_examples) + 
             ["Negative"] * len(negative_examples) + 
             ["Neutral"] * len(neutral_examples))
    
    return texts, labels

# Train multiple models for ensemble sentiment analysis
def train_ensemble_model():
    """
    Train an ensemble of ML models for sentiment analysis.
    Save the model for reuse.
    """
    # Check if model already exists
    model_path = Path("tiktok_sentiment_model.joblib")
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except:
            pass  # If loading fails, train a new model
    
    # Generate training data
    texts, labels = generate_training_data()
    
    # Create feature extraction and model pipeline
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2)
    
    # Create ensemble of models
    estimators = [
        ('nb', MultinomialNB()),
        ('svc', LinearSVC(C=1.0, class_weight='balanced', dual=False))
    ]
    
    # Build voting classifier
    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', ensemble)
    ])
    
    # Train the model
    pipeline.fit(texts, labels)
    
    # Save the model
    try:
        joblib.dump(pipeline, model_path)
    except:
        pass  # If saving fails, just continue
    
    return pipeline

# Function that uses ML model to predict sentiment
def predict_sentiment_ml(text_series):
    """
    Use trained ensemble model to predict sentiment.
    
    Args:
        text_series: Pandas Series containing processed text
    
    Returns:
        Series of sentiment predictions with confidence
    """
    # Load or train the model
    model = train_ensemble_model()
    
    # Convert to list if it's a single string
    if isinstance(text_series, str):
        text_series = [text_series]
    
    # Ensure all inputs are strings
    text_series = [str(text) if text is not None else "" for text in text_series]
    
    # Predict class
    predictions = model.predict(text_series)
    
    # Try to get probabilities if possible (not all models support predict_proba)
    try:
        probabilities = model.predict_proba(text_series)
        confidence_scores = np.max(probabilities, axis=1)
        # Format results with confidence scores
        result = [f"{pred} ({conf:.2f})" for pred, conf in zip(predictions, confidence_scores)]
    except:
        # If predict_proba not available, use fixed confidence
        result = [f"{pred} (0.80)" for pred in predictions]
    
    if len(result) == 1 and isinstance(text_series, list) and len(text_series) == 1:
        return result[0]
        
    return pd.Series(result)

# Replace your existing train_mnb_model function with this one

def train_mnb_model(text_series):
    """
    Trains a MultinomialNB model on given text and returns predictions.
    First tries to use custom labeled data if available.
    
    Args:
        text_series: Pandas Series containing processed text
    
    Returns:
        Series of sentiment predictions
    """
    # Try to use custom labeled model first
    custom_model = train_with_labeled_data()
    
    if custom_model is not None:
        # We have enough labeled data, use the custom model
        pipeline = custom_model
    else:
        # Fall back to the regular training data
        train_texts, train_labels = generate_training_data()
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        pipeline.fit(train_texts, train_labels)
    
    # Convert to list if it's a single string
    if isinstance(text_series, str):
        text_series = [text_series]
    
    # Ensure all inputs are strings
    text_series = [str(text) if text is not None else "" for text in text_series]
    
    # Predict on input texts
    predictions = pipeline.predict(text_series)
    
    # Add confidence scores
    try:
        probabilities = pipeline.predict_proba(text_series)
        confidence_scores = np.max(probabilities, axis=1)
    except:
        # Fixed confidence if predict_proba fails
        confidence_scores = [0.8] * len(predictions)
    
    # Format results with confidence scores
    result = [f"{pred} ({conf:.2f})" for pred, conf in zip(predictions, confidence_scores)]
    
    if len(result) == 1 and isinstance(text_series, list) and len(text_series) == 1:
        return result[0]
        
    return pd.Series(result)

# Comprehensive sentiment analysis function
def combined_sentiment_analysis(text_series):
    """
    Combines multiple sentiment analysis techniques for improved accuracy.
    
    Args:
        text_series: Pandas Series containing text
    
    Returns:
        Series of combined sentiment results
    """
    # Convert to list if it's a single string
    single_input = False
    if isinstance(text_series, str):
        text_series = [text_series]
        single_input = True
    
    results = []
    
    for text in text_series:
        if not isinstance(text, str) or not text:
            results.append("Neutral (0.00)")
            continue
        
        # Get VADER sentiment
        sid = SentimentIntensityAnalyzer()
        vader_scores = sid.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Extract emojis
        emojis_found = ''.join(c for c in text if c in emoji.EMOJI_DATA)
        
        # Get emoji sentiment if emojis exist
        emoji_score = 0
        if emojis_found:
            emoji_score = analyze_emoji_sentiment(emojis_found)
        
        # Get TikTok lexicon sentiment
        lexicon_score = analyze_lexicon_sentiment(text)
        
        # Weight the scores
        weights = {
            'vader': 0.6,  # VADER has highest weight
            'emoji': 0.2,  # Emojis are important in TikTok content
            'lexicon': 0.2  # TikTok-specific lexicon
        }
        
        final_score = (
            vader_compound * weights['vader'] +
            emoji_score * weights['emoji'] +
            lexicon_score * weights['lexicon']
        )
        
        # Adjust thresholds for TikTok content which tends to be more polarized
        if final_score >= 0.05:
            results.append(f"Positive ({final_score:.2f})")
        elif final_score <= -0.05:
            results.append(f"Negative ({final_score:.2f})")
        else:
            results.append(f"Neutral ({final_score:.2f})")
    
    if single_input:
        return results[0]
        
    return pd.Series(results)

# Enhanced sentiment analysis with ensemble approach
def enhanced_sentiment_analysis(text_series):
    """
    Ensemble approach combining multiple sentiment analysis methods.
    
    Args:
        text_series: Pandas Series or string containing text
    
    Returns:
        Combined sentiment results
    """
    # Convert to list if it's a single string
    single_input = False
    if isinstance(text_series, str):
        text_series = [text_series]
        single_input = True
    
    results = []
    
    for text in text_series:
        if not isinstance(text, str) or not text:
            results.append("Neutral (0.00)")
            continue
        
        # Process text for analysis
        processed = preprocess_for_sentiment(text)
        clean_text = processed['processed_text']
        
        # Get scores from different methods
        
        # VADER sentiment (rule-based)
        vader_sentiment = analyze_sentiment_vader(text)
        vader_score = float(re.search(r'\(([-+]?\d+\.\d+)\)', vader_sentiment).group(1))
        
        # ML model prediction
        try:
            ml_sentiment = predict_sentiment_ml(clean_text)
            ml_score_match = re.search(r'\(([-+]?\d+\.\d+)\)', ml_sentiment)
            ml_score = float(ml_score_match.group(1)) if ml_score_match else 0.0
            
            # Convert categorical to numerical (-1 to 1 scale)
            if "Positive" in ml_sentiment:
                ml_score = abs(ml_score)
            elif "Negative" in ml_sentiment:
                ml_score = -abs(ml_score)
            else:
                ml_score = 0.0
        except:
            # If ML prediction fails
            ml_score = 0.0
        
        # Emoji analysis
        emoji_score = analyze_emoji_sentiment(processed['emojis'])
        
        # TikTok lexicon analysis
        lexicon_score = analyze_lexicon_sentiment(text)
        
        # Weight the scores
        weights = {
            'vader': 0.4,   # VADER is reliable for social media
            'ml': 0.3,      # ML model captures patterns
            'emoji': 0.15,  # Emojis are important in TikTok
            'lexicon': 0.15 # TikTok-specific language
        }
        
        # Calculate weighted ensemble score
        final_score = (
            vader_score * weights['vader'] +
            ml_score * weights['ml'] +
            emoji_score * weights['emoji'] +
            lexicon_score * weights['lexicon']
        )
        
        # Determine sentiment category
        if final_score >= 0.05:
            results.append(f"Positive ({final_score:.2f})")
        elif final_score <= -0.05:
            results.append(f"Negative ({final_score:.2f})")
        else:
            results.append(f"Neutral ({final_score:.2f})")
    
    if single_input:
        return results[0]
    
    return pd.Series(results)

def ensemble_sentiment_analysis(texts):
    """
    Ensemble of all sentiment analysis methods.
    """
    # Handle different input types
    single_input = False
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    elif isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    results = []
    
    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            results.append("Neutral (0.00)")
            continue
        
        # Get results from all methods
        vader_result = analyze_sentiment_vader(text)
        enhanced_result = enhanced_sentiment_analysis(text)
        
        # Get ML results if model is trained
        if ml_model is not None:
            ml_result = predict_mnb_model(text)
        else:
            # Use enhanced as substitute if ML not available
            ml_result = enhanced_result
        
        # Extract scores
        vader_score = _extract_sentiment_score(vader_result)
        ml_score = _extract_sentiment_score(ml_result)
        enhanced_score = _extract_sentiment_score(enhanced_result)
        
        # Get transformer result if available
        try:
            from transformers import pipeline
            sentiment_pipeline = pipeline('sentiment-analysis')
            transformer_result = sentiment_pipeline(text)[0]
            
            # Map pipeline result to our scoring system
            if transformer_result['label'] == 'POSITIVE':
                transformer_score = transformer_result['score']
            elif transformer_result['label'] == 'NEGATIVE':
                transformer_score = -transformer_result['score']
            else:
                transformer_score = 0.0
                
            # Weighted average with transformer
            weights = {'vader': 0.2, 'ml': 0.2, 'enhanced': 0.3, 'transformer': 0.3}
            final_score = (
                weights['vader'] * vader_score +
                weights['ml'] * ml_score +
                weights['enhanced'] * enhanced_score +
                weights['transformer'] * transformer_score
            )
        except:
            # Weighted average without transformer
            weights = {'vader': 0.3, 'ml': 0.3, 'enhanced': 0.4}
            final_score = (
                weights['vader'] * vader_score +
                weights['ml'] * ml_score +
                weights['enhanced'] * enhanced_score
            )
        
        # Convert to sentiment label with score
        if final_score >= 0.05:
            results.append(f"Positive ({final_score:.2f})")
        elif final_score <= -0.05:
            results.append(f"Negative ({final_score:.2f})")
        else:
            results.append(f"Neutral ({final_score:.2f})")
    
    # Return based on input type
    if single_input:
        return results[0]
    else:
        return pd.Series(results)

# Function to get sentiment scores breakdown
def get_sentiment_breakdown(text):
    """
    Get detailed breakdown of sentiment scores from different methods.
    
    Args:
        text: String text to analyze
    
    Returns:
        Dictionary with sentiment scores from each method
    """
    if not isinstance(text, str) or not text:
        return {
            "vader": 0.0,
            "emoji": 0.0,
            "lexicon": 0.0,
            "ml": 0.0,
            "final": 0.0,
            "sentiment": "Neutral"
        }
    
    # Process text
    processed = preprocess_for_sentiment(text)
    
    # VADER sentiment
    sid = SentimentIntensityAnalyzer()
    vader_scores = sid.polarity_scores(text)
    vader_score = vader_scores['compound']
    
    # Emoji sentiment
    emoji_score = analyze_emoji_sentiment(processed['emojis'])
    
    # Lexicon sentiment
    lexicon_score = analyze_lexicon_sentiment(text)
    
    # ML model prediction
    try:
        ml_sentiment = predict_sentiment_ml(processed['processed_text'])
        ml_score_match = re.search(r'\(([-+]?\d+\.\d+)\)', ml_sentiment)
        ml_score = float(ml_score_match.group(1)) if ml_score_match else 0.0
        
        # Convert categorical to numerical (-1 to 1 scale)
        if "Positive" in ml_sentiment:
            ml_score = abs(ml_score)
        elif "Negative" in ml_sentiment:
            ml_score = -abs(ml_score)
        else:
            ml_score = 0.0
    except:
        ml_score = 0.0
    
    # Weight the scores
    weights = {
        'vader': 0.4,
        'ml': 0.3,
        'emoji': 0.15,
        'lexicon': 0.15
    }
    
    # Calculate final score
    final_score = (
        vader_score * weights['vader'] +
        ml_score * weights['ml'] +
        emoji_score * weights['emoji'] +
        lexicon_score * weights['lexicon']
    )
    
    # Determine sentiment
    if final_score >= 0.05:
        sentiment = "Positive"
    elif final_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        "vader": vader_score,
        "emoji": emoji_score,
        "lexicon": lexicon_score,
        "ml": ml_score,
        "final": final_score,
        "sentiment": sentiment
    }
