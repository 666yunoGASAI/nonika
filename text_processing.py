import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stopwords
english_stop_words = set(stopwords.words('english'))

# Add custom stopwords specific to TikTok comments
tiktok_stopwords = {
    'tiktok', 'comment', 'duet', 'stitch', 'fyp', 'foryou', 'foryoupage', 
    'fy', 'viral', 'trending', 'follow', 'like', 'share', 'comment',
    'pov', 'fyp️', 'foryou️', 'fypシ'
}
english_stop_words.update(tiktok_stopwords)

# Tagalog stopwords
tagalog_stop_words = {
    'ang', 'ng', 'sa', 'na', 'at', 'ay', 'mga', 'ko', 'siya', 'mo', 'ito', 'po',
    'ako', 'ka', 'niya', 'si', 'ni', 'kayo', 'kami', 'tayo', 'sila', 'nila',
    'namin', 'natin', 'kanila', 'kaniya', 'ating', 'amin', 'akin', 'iyo', 'nito',
    'dito', 'diyan', 'doon', 'kung', 'pag', 'para', 'dahil', 'upang', 'nga',
    'lang', 'lamang', 'din', 'rin', 'pa', 'pala', 'ba', 'naman', 'kasi', 'hindi',
    'huwag', 'wag', 'oo', 'hindi', 'yung', 'yan', 'yun'
}

# TikTok-specific Tagalog stopwords
tagalog_tiktok_stopwords = {
    'charot', 'charr', 'char', 'skl', 'emz', 'mema', 'borde', 'forda', 
    'ferson', 'fordalinis', 'skeri', 'pov', 'mhie', 'bhie', 'mars', 
    'accla', 'teh', 'sis', 'dzai', 'ghorl', 'mumsh', 'siz'
}
tagalog_stop_words.update(tagalog_tiktok_stopwords)

def is_tagalog(text):
    """
    Simple heuristic to detect if text is primarily in Tagalog.
    Uses common Tagalog function words as indicators.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        bool: True if text appears to be in Tagalog, False otherwise
    """
    if not isinstance(text, str) or not text:
        return False
    
    # Common Tagalog function words as detection markers
    tagalog_markers = {
        'ang', 'ng', 'sa', 'ay', 'mga', 'na', 'naman', 'po', 'ko', 'mo', 
        'siya', 'niya', 'ito', 'iyon', 'yung', 'yan', 'din', 'rin', 'pa',
        'kung', 'para', 'kasi', 'dahil', 'nga', 'wag', 'huwag'
    }
    
    # Clean and tokenize text
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    
    # Count Tagalog marker words
    tagalog_word_count = sum(1 for word in words if word in tagalog_markers)
    
    # If at least 15% of words are Tagalog markers or at least 2 Tagalog markers are present
    # in a short message, consider it Tagalog
    if len(words) > 0:
        if (tagalog_word_count / len(words) >= 0.15) or (tagalog_word_count >= 2 and len(words) <= 10):
            return True
    
    return False

def clean_text(text):
    """
    Basic text cleaning function.
    
    Args:
        text (str): Text to clean
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove user mentions (common in social media)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_remove_stopwords(text):
    """
    Tokenize text, remove stopwords, and lemmatize.
    Handles both English and Tagalog text.
    
    Args:
        text (str): Text to process
    
    Returns:
        list: List of processed tokens
    """
    if not isinstance(text, str):
        return []
    
    # Detect language
    is_tag_text = is_tagalog(text)
    
    # Choose appropriate stopwords based on language
    stop_words = tagalog_stop_words if is_tag_text else english_stop_words
    
    # Tokenize
    tokens = word_tokenize(text)
    
    if is_tag_text:
        # For Tagalog, just remove stopwords (no lemmatization)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    else:
        # For English, remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                if word not in stop_words and len(word) > 2]
    
    return tokens

def extract_hashtags(text):
    """
    Extract hashtags from text.
    
    Args:
        text (str): Text to process
    
    Returns:
        list: List of hashtags
    """
    if not isinstance(text, str):
        return []
        
    hashtag_pattern = r'#(\w+)'
    hashtags = re.findall(hashtag_pattern, text)
    return hashtags

def extract_mentions(text):
    """
    Extract user mentions from text.
    
    Args:
        text (str): Text to process
    
    Returns:
        list: List of user mentions
    """
    if not isinstance(text, str):
        return []
        
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text)
    return mentions

def generate_ngrams(tokens, n=2):
    """
    Generate n-grams from tokens.
    
    Args:
        tokens (list): List of tokens
        n (int): Size of n-grams
    
    Returns:
        list: List of n-grams
    """
    if not tokens or len(tokens) < n:
        return []
        
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append('_'.join(tokens[i:i+n]))
    return ngrams

def preserve_emoticons(text):
    """
    Preserve emoticons in text during preprocessing.
    
    Args:
        text (str): Text to process
    
    Returns:
        str: Text with emoticons preserved
    """
    if not isinstance(text, str):
        return ""
    
    # Common emoticons
    emoticons = {
        ':)': ' HAPPY_FACE ', ':-)': ' HAPPY_FACE ', '(:': ' HAPPY_FACE ', 
        ':(': ' SAD_FACE ', ':-(': ' SAD_FACE ', '):': ' SAD_FACE ',
        ';)': ' WINKING_FACE ', ';-)': ' WINKING_FACE ',
        ':D': ' LAUGHING_FACE ', ':-D': ' LAUGHING_FACE ',
        ':p': ' TONGUE_FACE ', ':P': ' TONGUE_FACE ', ':-p': ' TONGUE_FACE ', ':-P': ' TONGUE_FACE ',
        ':o': ' SURPRISED_FACE ', ':O': ' SURPRISED_FACE ', ':-o': ' SURPRISED_FACE ', ':-O': ' SURPRISED_FACE ',
        '<3': ' HEART ', '</3': ' BROKEN_HEART ',
        '(y)': ' THUMBS_UP ', '(n)': ' THUMBS_DOWN '
    }
    
    # Replace emoticons with tokens
    for emoticon, token in emoticons.items():
        text = text.replace(emoticon, token)
    
    return text

def detect_tiktok_slang(text):
    """
    Detect TikTok-specific slang in text (works for both English and Tagalog).
    
    Args:
        text (str): Text to process
    
    Returns:
        list: List of TikTok slang terms found
    """
    if not isinstance(text, str):
        return []
    
    # English TikTok slang terms
    english_tiktok_slang = {
        'fyp', 'foryou', 'foryoupage', 'xyzbca', 'viral', 'trending', 
        'slay', 'periodt', 'bussin', 'rizz', 'mid', 'goated', 'based',
        'fire', 'lit', 'cap', 'no cap', 'cringe', 'sus', 'ick', 'iykyk',
        'flop', 'W', 'L', 'chief', 'rent free', 'understood the assignment',
        'main character', 'vibe check', 'simp', 'cheugy', 'ratio',
        'living rent free', 'ate', 'clean', 'atp', 'fax', 'hits different'
    }
    
    # Tagalog TikTok slang terms
    tagalog_tiktok_slang = {
        'angat', 'pak', 'pak ganern', 'lodi', 'petmalu', 'awra', 'shookt', 
        'labyu', 'labs', 'poging', 'gandang', 'sana all', 'sanaol', 'apaka ganda',
        'keri', 'angas', 'witty', 'kyot', 'kyut', 'naurrr', 'peri true', 'trulalu', 
        'deserve', 'slayyy', 'slayyyy', 'mhie', 'bhie', 'mars', 'accla', 'teh',
        'sis', 'dzai', 'ghorl', 'mumsh', 'siz', 'charot', 'charr', 'char', 'skl', 
        'emz', 'borde', 'forda', 'ferson', 'fordalinis', 'mema', 'skeri'
    }
    
    # Combine both sets
    all_tiktok_slang = english_tiktok_slang.union(tagalog_tiktok_slang)
    
    # Tokenize text
    words = text.lower().split()
    
    # Check for single word slang
    slang_found = [word for word in words if word in all_tiktok_slang]
    
    # Check for multi-word slang phrases
    text_lower = text.lower()
    for phrase in all_tiktok_slang:
        if ' ' in phrase and phrase in text_lower:
            slang_found.append(phrase)
    
    return slang_found

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