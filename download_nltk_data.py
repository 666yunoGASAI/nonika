import os
import nltk
import ssl
import urllib.request
import zipfile
import io

# Create data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
corpus_dir = os.path.join(nltk_data_dir, 'corpora')
if not os.path.exists(corpus_dir):
    os.makedirs(corpus_dir, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

# SSL workaround
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Try direct download for WordNet
wordnet_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
wordnet_dir = os.path.join(corpus_dir, 'wordnet')

print("Directly downloading WordNet...")
try:
    # Download the zip file
    with urllib.request.urlopen(wordnet_url) as response:
        zip_data = response.read()
    
    # Extract the zip file
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
        zip_ref.extractall(corpus_dir)
    
    print("✓ WordNet downloaded and extracted successfully")
except Exception as e:
    print(f"Error downloading WordNet: {e}")
    print("Trying alternative method...")
    try:
        nltk.download('wordnet', download_dir=nltk_data_dir)
    except Exception as e2:
        print(f"Alternative download also failed: {e2}")

# Verify installation
print("\nVerification - checking if wordnet is properly installed:")
try:
    nltk.data.find('corpora/wordnet')
    print("✓ WordNet found and properly installed")
except LookupError as e:
    print(f"Error: {e}")
    print("WordNet was not installed correctly.")