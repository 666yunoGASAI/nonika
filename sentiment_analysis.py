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
import langdetect
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

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
    'flopped': -0.8, 'overrated': -0.6, 'boring': -0.6, 'annoying': -0.7, 'wtf': -0.6,
    
    
    # Negative product-specific terms
    'defective': -0.8, 'sira': -0.8, 'broken': -0.8, 'not working': -0.8, 
    'budol': -0.9, 'peke': -0.8, 'counterfeit': -0.8, 'knockoff': -0.8,
    'overpriced': -0.7, 'mahal': -0.6, 'sobrang mahal': -0.8, 'not worth it': -0.7, 
    'hindi sulit': -0.7, 'sayang': -0.7, 'waste': -0.8, 'regret': -0.8,
    'misleading': -0.7, 'false advertising': -0.8, 'refund': -0.5, 'return': -0.5,
    'terrible': -0.8, 'avoid': -0.7, 'stay away': -0.8, 'pangit': -0.7,
    'marupok': -0.7, 'madaling masira': -0.8, 'walang kwenta': -0.8, 'useless': -0.8,
    'expired': -0.8, 'expiry': -0.7, 'arrived damaged': -0.8, 'doa': -0.8,
    'not as described': -0.7, 'hindi kapareho': -0.7, 'catfishing': -0.8,
    
    # Shopping platform specific terms
    'shopee scam': -0.9, 'lazada scam': -0.9, 'tiktok shop scam': -0.9,
    'shopee budol': -0.8, 'lazada budol': -0.8, 'tiktok shop budol': -0.8,
    'cod scam': -0.9, 'cash on delivery scam': -0.9,
    'seller ghosted': -0.8, 'hindi nagrereply': -0.7, 'no response': -0.7,
    'cancellation': -0.6, 'cancelled order': -0.6, 'delayed shipping': -0.6,
    
    # Positive product-specific terms
    'sulit': 0.7, 'worth it': 0.7, 'quality': 0.7, 'excellent': 0.8, 'legit': 0.8,
    'authentic': 0.7, 'original': 0.7, 'recommended': 0.7, 'mura': 0.6, 'affordable': 0.6,
    'fast delivery': 0.7, 'mabilis dumating': 0.7, 'responsive seller': 0.7,
    'generous seller': 0.7, 'exceeded expectations': 0.8, 'free gift': 0.6,
    'good packaging': 0.6, 'secure packaging': 0.6, 'well packed': 0.6
}
FILIPINO_LEXICON = {
    # Positive Filipino/Taglish words
    'ganda': 0.7, 'astig': 0.8, 'galing': 0.8, 'maganda': 0.7,
    'husay': 0.7, 'ang galing': 0.8, 'ang cute': 0.7, 'ang ganda': 0.8,
    'idol': 0.7, 'petmalu': 0.8, 'lodi': 0.7, 'solid': 0.7, 'lupet': 0.8,
    'panalo': 0.8, 'sana all': 0.6, 'nakakatuwa': 0.7, 'bongga': 0.7,
    
    # Negative Filipino/Taglish words
    'panget': -0.7, 'pangit': -0.7, 'chaka': -0.7, 'nakakabwisit': -0.8,
    'basura': -0.8, 'bastos': -0.7, 'walang kwenta': -0.8, 'epal': -0.7,
    'tanga': -0.8, 'bobo': -0.8, 'tae': -0.7, 'gago': -0.8, 'gaga': -0.8,
    'ulol': -0.8, 'pakyu': -0.9, 'bwisit': -0.7, 'nakakaasar': -0.7,
    'kadiri': -0.7, 'bulok': -0.7, 'tarantado': -0.8, 'pokpok': -0.8,
    
    # Filipino troll/bait words (common on TikTok)
    'lutang': -0.7, 'lenlen': -0.7, 'dilawan': -0.6, 'pinklawan': -0.6, 
    'yellowtard': -0.7, 'kulto': -0.8, 'bayaran': -0.7, 'paid': -0.6,
    'troll': -0.7, 'fake news': -0.7, 'apologist': -0.6, 'fanatic': -0.6,

        # Add more negative/troll Filipino words
    'engot': -0.7, 'inutil': -0.7, 'gunggong': -0.8, 'hangal': -0.7, 
    'ungas': -0.8, 'ugok': -0.7, 'ulol': -0.8, 'abnormal': -0.7,
    'baliw': -0.7, 'sira ulo': -0.8, 'sinungaling': -0.7, 'budol': -0.7,
    'balimbing': -0.6, 'traydor': -0.7, 'duwag': -0.7, 'takot': -0.5,
    'manahimik ka': -0.7, 'tumahimik ka': -0.7, 'tumigil ka': -0.7,
    'magsara': -0.6, 'umalis': -0.5, 'pabebe': -0.6, 'feeling': -0.6,
    'epal': -0.7, 'attention': -0.5, 'papansin': -0.6, 'praning': -0.6,
    'plastik': -0.7, 'fake': -0.7, 'peke': -0.7, 'poser': -0.6,
    
    # Additional political troll terms
    'bobotante': -0.8, 'abnoy': -0.8, 'lugaw': -0.7, 'laylayan': -0.6,
    'gurang': -0.7, 'matanda': -0.6, 'luka-luka': -0.7, 'oligarch': -0.7,
    'elitista': -0.7, 'tuta': -0.8, 'sundalong kanin': -0.7,
    'puppet': -0.7, 'diktador': -0.7, 'magnanakaw': -0.8, 'adik': -0.7,
    'drugas': -0.7, 'addict': -0.7, 'korap': -0.8, 'corrupt': -0.8,
    
    # Sarcastic terms often used in trolling
    'maka diyos': -0.5, 'disente': -0.5, 'itong': -0.4, 'heto': -0.4,
    'talaga naman': -0.5, 'asa pa': -0.6, 'good luck': -0.5, 'lmao': -0.6,
    'tatawa': -0.4, 'natawa': -0.5, 'patawa': -0.5, 'aliw': -0.4,
    'naaliw': -0.5, 'pinapatawa': -0.5, 'katatawa': -0.5

    
}

TROLL_PATTERNS = [
    # Basic patterns
    r'(ha){3,}',              # hahaha patterns
    r'(he){3,}',              # hehehe patterns
    r'!!+',                   # multiple exclamation marks
    r'\?\?+',                 # multiple question marks - fixed to escape ?
    r'[A-Z]{3,}',             # ALL CAPS WORDS
    r'\.{3,}',                # multiple periods (ellipsis)
    r'😂{2,}',                # multiple laughing emojis
    r'🤡|💀|🫠|💩',           # emojis commonly used in trolling
    r'(lutang|bobo|tanga)\s*(ka|talaga|naman)', # common insults 
    r'(\w+)(?:(?:\s+\1){2,})',  # repeated words (e.g., "galing galing galing")
    r'respect\s*my\s*opinion',  # common troll defense
    
    # Additional patterns - with fixes
    r'#(bbm|sara|leni|kakampink|pinklawan|dilawan|duterte|dutz|marcos)',  # Political hashtags
    r'(bbm|marcos|duterte|dutz|leni|lugaw|lutang)\s*pa\s*more',  # Political trolling phrase
    r'(bobo|tanga|gago)\s*(ka\s*ba|naman|talaga|amp)',  # Enhanced insult patterns  
    r'(haha|hihi|huhu|hehe){2,}',  # Repeated laugh patterns (hahahaha, etc)
    r'\?{2,}',                  # Fixed question mark patterns in Filipino comments
    r'(\s*ha){3,}',            # Enhanced "ha ha ha" pattern
    r'(\s*eh){3,}',            # "eh eh eh" pattern
    r'(naman){2,}',            # Repeated "naman"
    r'(edi\s*wow|edi\s*ikaw\s*na)',  # Sarcastic expressions
    r'(dami\s*alam|dami\s*satsat)',  # Dismissive phrases
    r'(feeling|filang)\s*(expert|galing|maganda|pogi|matalino)',  # Sarcastic compliment
    r'(ayaw|gusto|trip)\s*ko\s*yan[!]+',  # Exaggerated statements
    r'(fake|imbento|gawa-gawa|kasinungalingan)\s*(news|yan)',  # Fake news accusations
    r'(delawan|dilawan|yellowtard|pinklawan)',  # Political faction insults
    r'(paid|bayad|bayaran)',  # Paid troll accusations
    r'(communist|komunista|npa|terorista)',  # Political labeling
    r'(wala\s*kang\s*alam|wala\s*kang\s*karapatan)',  # Dismissive statements
    r'(kabobohan|katangahan|kaululan)',  # Name-calling
    r'\b(ok|sige|sure)\s*na\s*yan\s*for\s*you',  # Dismissive agreement
    r'#(tiktokfamous|viralvideo|foryoupage|fyp)',  # Hashtag baiting
        # Product-specific troll patterns
    r'(fake|peke|scam|budol|lokohan|manloloko)\s*(product|item|seller)',  # Fake product accusations
    r'(over|sobrang)\s*(priced|mahal)',  # Overpriced complaints
    r'(don\'t|wag|huwag)\s*(buy|order|bilhin)',  # Discouraging purchases
    r'(waste|sayang)\s*(of|ng)\s*(money|pera)',  # Money waste claims
    r'(worst|pinakamasamang)\s*(purchase|bili|product)',  # Extreme negative claims
    r'(returns?|refunds?)\s*(denied|rejected|hindi)',  # Return/refund complaints
    r'(buyer|customer)\s*(beware|ingat)',  # Warning other customers
    r'(marketing|ad|advertisement)\s*(scam|lie|kasinungalingan)',  # Marketing dishonesty claims
    r'(not|hindi)\s*(worth|sulit)',  # Value complaints
    r'(broken|sira|defective)\s*(on|upon|pagka)\s*(arrival|dating|deliver)',  # DOA claims
    
    # Exaggerated reviews
    r'(never|hindi\s*na\s*ulit)\s*(buying|bibili)',  # Never buying again
    r'(regret|nagsisisi)\s*(buying|purchase)',  # Purchase regret
    r'(0|zero)\s*(stars|rating)',  # Zero rating claims
    r'(this|ito)\s*(ain\'t|hindi)\s*(it|maganda|okay)',  # Dismissive language
    
    # Suspicious behavior patterns
    r'(all|lahat\s*ng)\s*(reviews|ratings|comments)\s*(fake|peke|paid|bayad)',  # Fake review accusations
    r'(shop|store)\s*(paying|nagbabayad)\s*for\s*(good|positive|5\s*star)',  # Paid review accusations
    r'(daming|andaming|ang\s*dami\s*ng)\s*(tanga|bobo|uto-uto)',  # Insulting other customers
    r'(obvious|halatang)\s*(paid|bayad)',  # Calling out paid endorsements
    
    # Specific product claim patterns
    r'(expired|expiry|lumang)\s*(product|item)',  # Expired products
    r'(fake|peke|counterfeit|pirated|class\s*[a-z])',  # Counterfeit accusations
    r'(factory|manufacturer)\s*(defect|reject)',  # Factory defects
    r'(not|hindi)\s*(authentic|original|tunay)',  # Authenticity questions
    r'(expectations)\s*vs\s*(reality)',  # Expectation vs reality
    r'(order|expectation)\s*vs\s*(received|reality)',  # Order vs received
    
    # Mocking patterns
    r'(laughing|tumawa|natawa)\s*so\s*(hard|much)',  # Mocking reactions
    r'(clown|circus|joke)\s*(emoji|face)?',  # Calling product/seller a joke
    r'(imagine|isipin)\s*(paying|nagbayad)',  # Mocking buyers
    
    # Extreme comparison claims
    r'(better|mas\s*maganda)\s*(off|pa)\s*(buying|bumili)',  # Better off elsewhere claims
    r'(local|china|chinese)\s*(products|alternatives)\s*(better|mas\s*maganda)',  # Comparison to alternatives
    r'(could|pwede)\s*(have|sana)\s*(bought|bumili)',  # Regret comparisons
    
    # General catastrophizing
    r'(worst|pinaka\s*malala|pinakamasamang)\s*(product|experience)',  # Extreme negative statements
    r'(completely|totally|absolutely)\s*(useless|worthless|walang\s*kwenta)',  # Extreme uselessness claims
    r'(biggest|pinakamalaking)\s*(mistake|regret)',  # Extreme regret claims
    
    # Filipino-specific patterns
    r'(budol|na-budol|nabudol)',  # Scammed
    r'(sayang|nasayang)\s*(pera|money|funds)',  # Wasted money
    r'(lubog|nilubog|nilubog)\s*(pera|money)',  # Sunken cost
    r'(pinagkakitaan|kita|kumikita)\s*(lang)',  # Just for profit
    r'(puro|pure)\s*(kalokohan|pangako|promises)',  # Empty promises
    r'(panloloko|manloloko|pangloloko)',  # Deception terms
]
def validate_regex_patterns(patterns):
    """
    Validates a list of regex patterns and returns only those that compile successfully.
    
    Args:
        patterns: List of regex pattern strings
        
    Returns:
        List of valid regex patterns
    """
    valid_patterns = []
    for i, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
            valid_patterns.append(pattern)
        except re.error as e:
            print(f"Invalid regex pattern #{i}: '{pattern}'")
            print(f"Error: {e}")
            # Either skip the pattern or try to fix it
            # Here we'll skip it for safety
    
    return valid_patterns
# Function to detect language
def detect_language(text):
    """
    Detect if text is Tagalog, English, or Taglish (mixed).
    Returns 'tl' for Tagalog, 'en' for English, 'mixed' for Taglish, or 'unknown'.
    """
    if not isinstance(text, str) or not text:
        return 'unknown'
    
    # Common Filipino markers that help identify Tagalog/Taglish
    filipino_markers = ['ang', 'ng', 'mga', 'sa', 'ko', 'mo', 'ka', 'naman', 'po',
                        'na', 'ay', 'yung', 'ito', 'yan', 'siya', 'ikaw', 'ako']
    
    # Count Filipino marker words
    words = text.lower().split()
    filipino_word_count = sum(1 for word in words if word in filipino_markers)
    filipino_lexicon_words = sum(1 for word in words if word in FILIPINO_LEXICON)
    
    # If significant Filipino markers are found
    if filipino_word_count >= 2 or filipino_lexicon_words >= 2:
        # Simple check for English words
        english_markers = ['the', 'of', 'and', 'to', 'is', 'in', 'it', 'you', 'that']
        english_count = sum(1 for word in words if word in english_markers)
        
        if english_count >= 2:
            return 'mixed'  # Likely Taglish
        else:
            return 'tl'     # Likely Filipino/Tagalog
    
    # Try standard language detection (this can be error-prone for short texts)
    try:
        return detect(text)
    except:
        # If language detection fails
        return 'unknown'
    
    # Function to detect troll patterns
def detect_troll_patterns(text):
    """
    Detect patterns commonly found in troll comments.
    Returns a score from 0 (not troll-like) to 1 (highly troll-like).
    """
    if not isinstance(text, str) or not text:
        return 0.0
    
    # Initialize score
    troll_score = 0.0
    
    # Check for pattern matches
    pattern_matches = 0
    for pattern in TROLL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches += 1
    
    # Increase impact of pattern matches (changed from 0.1 to 0.15 per match)
    # Normalize score based on matches (max impact 0.75 instead of 0.6)
    if pattern_matches > 0:
        troll_score += min(0.75, pattern_matches * 0.15)
    
    # Check for extreme sentiment words (common in trolling)
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count strong negative Filipino words
    strong_negative_count = sum(1 for word in words if word in FILIPINO_LEXICON 
                               and FILIPINO_LEXICON[word] <= -0.7)
    
    # Increase impact of negative words (changed from 0.1 to 0.15 per word)
    # Add to score based on strong negative words (max impact 0.5 instead of 0.4)
    if strong_negative_count > 0:
        troll_score += min(0.5, strong_negative_count * 0.15)
    
    return min(1.0, troll_score)  # Cap at 1.0

def has_excessive_formatting(text):
    """
    Check if text has excessive formatting typical of troll comments
    """
    # Check for ALL CAPS (if the comment is mostly uppercase)
    words = text.split()
    if len(words) >= 2:  # Only check comments with at least 2 words
        uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        uppercase_ratio = uppercase_words / len(words)
        if uppercase_ratio > 0.5:  # If more than 50% of words are ALL CAPS
            return 0.5
    
    # Check for excessive punctuation
    punctuation_count = sum(1 for char in text if char in '!?.')
    if len(text) > 0:
        punctuation_ratio = punctuation_count / len(text)
        if punctuation_ratio > 0.1:  # If more than 10% of the text is punctuation
            return 0.4
    
    # Check for repeated characters (like "hahahaha" or "!!!!!")
    if re.search(r'(.)\1{3,}', text):  # Same character repeated 4+ times
        return 0.3
        
    return 0.0
def analyze_emoji_sentiment_for_trolls(emoji_text):
    """
    Check if the emojis in a comment indicate potential trolling
    Returns a troll score from 0 to 1 based on emojis only
    """
    if not emoji_text:
        return 0.0
    
    # Emojis commonly used in trolling
    troll_emojis = {
        "🤡": 0.8,  # Clown face - very common in trolling
        "💀": 0.6,  # Skull - often used mockingly
        "😂": 0.4,  # Laughing crying - when used excessively
        "🤣": 0.4,  # Rolling on floor laughing - when used excessively
        "🫠": 0.5,  # Melting face - often used sarcastically
        "💩": 0.7,  # Poop emoji - directly insulting
        "🙄": 0.5,  # Eye roll - dismissive
        "🤦": 0.5,  # Facepalm - dismissive
        "🤪": 0.4,  # Zany face - mocking
        "🥴": 0.4,  # Woozy face - often used mockingly
        "👋": 0.3,  # Waving hand - dismissive in context
        "😴": 0.4,  # Sleeping - dismissive
        "🤓": 0.5,  # Nerd face - often used mockingly
    }
    
    total_score = 0
    count = 0
    
    # Count occurrences of each emoji
    emoji_counts = {}
    for char in emoji_text:
        if char in emoji_counts:
            emoji_counts[char] += 1
        else:
            emoji_counts[char] = 1
    
    # Calculate total score based on emojis and their counts
    for emoji, count in emoji_counts.items():
        if emoji in troll_emojis:
            # Higher score for repeated troll emojis
            repetition_factor = min(2.0, 1.0 + (0.2 * (count - 1)))
            total_score += troll_emojis[emoji] * repetition_factor
            count += 1
    
    # If no troll emojis were found
    if count == 0:
        return 0.0
        
    # Normalize to 0-1 range
    return min(1.0, total_score / count)

class TrollDetector:
    """
    A class to detect troll comments with more contextual awareness
    """
    def __init__(self):
        # Initialize troll detection patterns and thresholds
        self.troll_patterns = {
            'caps_ratio': 0.7,  # Ratio of uppercase characters
            'punctuation_repeat': r'[!?]{3,}',  # Repeated punctuation
            'char_repeat': r'(.)\1{4,}',  # Repeated characters
            'troll_phrases': ['troll', 'spam', 'fake', 'bot', 'reported', 'scam']
        }
        
        # Load custom troll phrase dictionaries
        self.tagalog_troll_phrases = [
            'fake news', 'bayaran', 'trolls', 'paid', 'propaganda',
            'spam', 'fake account', 'report', 'scam', 'mang-aasar'
        ]
    
    def check_caps_ratio(self, text):
        """Check ratio of uppercase characters"""
        if not text:
            return 0
        caps_count = sum(1 for c in text if c.isupper())
        return caps_count / len(text)
    
    def check_punctuation(self, text):
        """Check for excessive punctuation"""
        matches = re.findall(self.troll_patterns['punctuation_repeat'], text)
        return len(matches) > 0
    
    def check_char_repetition(self, text):
        """Check for character repetition"""
        matches = re.findall(self.troll_patterns['char_repeat'], text)
        return len(matches) > 0
    
    def check_troll_phrases(self, text, language='english'):
        """Check for presence of troll-related phrases"""
        text_lower = text.lower()
        phrases = (self.tagalog_troll_phrases if language == 'tagalog' 
                  else self.troll_patterns['troll_phrases'])
        return any(phrase in text_lower for phrase in phrases)
    
    def calculate_troll_score(self, text, language='english'):
        """Calculate overall troll score"""
        scores = {
            'caps': self.check_caps_ratio(text) * 0.3,
            'punctuation': float(self.check_punctuation(text)) * 0.2,
            'repetition': float(self.check_char_repetition(text)) * 0.2,
            'phrases': float(self.check_troll_phrases(text, language)) * 0.3
        }
        return sum(scores.values())

def analyze_for_trolling(text):
    """
    Analyze text for troll-like behavior.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary containing troll analysis results
    """
    detector = TrollDetector()
    
    # Detect language (simplified)
    is_tagalog = any(word in text.lower() for word in ['ang', 'ng', 'mga', 'sa', 'ko', 'ako'])
    language = 'tagalog' if is_tagalog else 'english'
    
    # Calculate troll score
    troll_score = detector.calculate_troll_score(text, language)
    
    # Determine if text is from a troll
    is_troll = troll_score > 0.6
    
    return {
        'is_troll': is_troll,
        'troll_score': troll_score,
        'language': language
    }

def analyze_comment_with_trolling(text, language_mode=None):
    """
    Analyzes comment for both sentiment and troll detection.
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Dictionary with sentiment and troll information
    """
    # Get standard sentiment analysis with language preference
    sentiment = analyze_sentiment_with_language_preference(text, language_mode)
    
    # Get troll analysis
    troll_analysis = analyze_for_trolling(text)
    
    # Extract just the base sentiment and score WITHOUT adding troll info
    sentiment_parts = sentiment.split(' (')
    base_sentiment = sentiment_parts[0]
    score = float(sentiment_parts[1].rstrip(')'))
    
    # Return sentiment and troll information completely separately
    return {
        'sentiment_text': base_sentiment,  # Just the sentiment without any extra info
        'sentiment_score': score,  # Just the score
        'is_troll': troll_analysis['is_troll'],
        'troll_score': troll_analysis['troll_score'],
        'language': troll_analysis['language']
    }

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return f"Positive ({scores['compound']:.2f})"
    elif scores['compound'] <= -0.05:
        return f"Negative ({scores['compound']:.2f})"
    else:
        return f"Neutral ({scores['compound']:.2f})"

def train_mnb_model(texts):
    """Train a Multinomial Naive Bayes model on the texts."""
    # Placeholder implementation
    return ["Neutral (0.00)" for _ in texts]

def combined_sentiment_analysis(text):
    """Combine multiple sentiment analysis methods."""
    return analyze_sentiment_vader(text)

def enhanced_sentiment_analysis(text):
    """Enhanced sentiment analysis with additional features."""
    return analyze_sentiment_vader(text)

def get_sentiment_breakdown(text):
    """Get detailed sentiment breakdown."""
    return {
        'vader': 0.0,
        'ml': 0.0,
        'emoji': 0.0,
        'lexicon': 0.0,
        'final': 0.0
    }

def is_tagalog(text):
    """Check if text is in Tagalog."""
    # Placeholder implementation
    return False

# Function to preprocess text for sentiment analysis
def preprocess_for_sentiment(text):
    """
    Preprocess text specifically for sentiment analysis, preserving emoticons and key phrases.
    """
    if not isinstance(text, str):
        return {"processed_text": "", "emojis": "", "demojized_text": "", "language": "unknown"}
    
    # Detect language - ADD THIS LINE
    language = detect_language(text)
    
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
        'demojized_text': text_with_emoji_names,
        'language': language  
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
def analyze_lexicon_sentiment(text, language=None):
    """
    Analyze sentiment using TikTok and Filipino-specific lexicon.
    
    Args:
        text: Text to analyze
        language: Optional language code. If None, will detect internally.
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Detect language if not provided
    if language is None:
        # Use the detect_language function
        language = detect_language(text)
    
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Also check for multi-word phrases
    bigrams = [words[i] + ' ' + words[i+1] for i in range(len(words)-1)]
    
    total_score = 0
    count = 0
    
    # Initialize lexicons with a default
    lexicons = [TIKTOK_LEXICON, FILIPINO_LEXICON]
    
    # Then modify the order based on detected language
    if language == 'tl':
        # Filipino content - check Filipino lexicon first, then TikTok
        lexicons = [FILIPINO_LEXICON, TIKTOK_LEXICON]
    elif language == 'mixed':
        # Taglish content - check both lexicons with equal priority
        lexicons = [FILIPINO_LEXICON, TIKTOK_LEXICON]
    # Default for 'en' or 'unknown' is already set above
    
    # Check for words in the appropriate lexicons
    for word in words:
        for lexicon in lexicons:
            if word in lexicon:
                total_score += lexicon[word]
                count += 1
                break  # Found in one lexicon, no need to check the other
    
    # Check for phrases
    for phrase in bigrams:
        for lexicon in lexicons:
            if phrase in lexicon:
                total_score += lexicon[phrase]
                count += 1
                break
    
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
        
        # Process text - this now includes language detection
        processed = preprocess_for_sentiment(text)
        detected_language = processed['language']  # Get the detected language
        
        # Get VADER sentiment
        sid = SentimentIntensityAnalyzer()
        vader_scores = sid.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Extract emojis
        emojis_found = processed['emojis']
        
        # Get emoji sentiment if emojis exist
        emoji_score = 0
        if emojis_found:
            emoji_score = analyze_emoji_sentiment(emojis_found)
        
        # Get TikTok lexicon sentiment - pass the detected language
        lexicon_score = analyze_lexicon_sentiment(text, detected_language)
        
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

# Example usage code for testing troll detection
def test_troll_detection():
    test_comments = [
        "I really love this content, so helpful!",
        "HAHAHAHA ANG BOBO MO NAMAN!!! 🤡🤡🤡",
        "Ang ganda ng video na to, very informative",
        "dilawan ka siguro bayaran ng mga aquino!!!!!",
        "This is mid at best, kinda disappointing",
        "RESPECT MY OPINION NALANG PO MGA SNOWFLAKE 🤡🤡🤡",
        "Neutral comment just saying hello",
        "walang kwentang content tapos ang panget pa ng presentation",
        "Sana all ganito kagaling mag explain",
    ]
    
    for comment in test_comments:
        result = analyze_for_trolling(comment)
        print(f"\nComment: {comment}")
        print(f"Language: {result['language']}")
        print(f"Troll Score: {result['troll_score']:.2f} (Is Troll: {result['is_troll']})")
        print(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})")

# To run the test when the script is executed directly
if __name__ == "__main__":
    # You can call your test function here
    test_troll_detection()
    
    # Or add your own test cases
    print("\nTesting a single comment:")
    result = analyze_for_trolling("Ang pangit ng content mo, walang kwenta!!!")
    print(f"Troll Score: {result['troll_score']:.2f} (Is Troll: {result['is_troll']})")

__all__ = [
    'TrollDetector',
    'analyze_sentiment_vader',
    'train_mnb_model',
    'combined_sentiment_analysis',
    'enhanced_sentiment_analysis',
    'get_sentiment_breakdown',
    'analyze_for_trolling',
    'analyze_comment_with_trolling'
]
