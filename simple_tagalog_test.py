import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"Python version: {sys.version}")
print("Starting Tagalog sentiment test...")

# SSL certificate fix for nltk downloads
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Import sentiment analysis functions
try:
    print("Importing sentiment analysis functions...")
    from sentiment_analysis import (
        analyze_sentiment_vader, 
        enhanced_sentiment_analysis
    )
    print("Successfully imported sentiment analysis functions")
except Exception as e:
    print(f"Error importing sentiment_analysis: {e}")
    
# Import Tagalog sentiment functions
try:
    print("Importing Tagalog sentiment functions...")
    from tagalog_sentiment import (
        is_tagalog,
        tagalog_enhanced_sentiment_analysis
    )
    print("Successfully imported Tagalog functions")
except Exception as e:
    print(f"Error importing tagalog_sentiment: {e}")

# Create simple test dataset
def create_simple_test_data():
    print("Creating test data...")
    
    # Sample Tagalog texts
    texts = [
        "Ang ganda ng video mo!", 
        "Sobrang pangit naman nito",
        "Hindi ko sure kung okay to",
        "Magaling ka talaga! Idol kita",
        "Nakakabadtrip naman to eh",
        "Super nice naman ito, galing mo!"
    ]
    
    # Correct sentiment labels
    sentiments = [
        "Positive", 
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Positive"
    ]
    
    return pd.DataFrame({'text': texts, 'sentiment': sentiments})

# Function to extract just the sentiment label
def extract_sentiment_label(result):
    if not result or not isinstance(result, str):
        return "Neutral"
        
    if result.startswith("Positive"):
        return "Positive"
    elif result.startswith("Negative"):
        return "Negative"
    else:
        return "Neutral"

# Run the test
def run_test():
    print("Running test...")
    
    # Create test data
    test_data = create_simple_test_data()
    print(f"Created {len(test_data)} test examples")
    
    # Display the test data
    print("\nTest data:")
    for i, row in test_data.iterrows():
        print(f"{i+1}. '{row['text']}' - Expected: {row['sentiment']}")
    
    print("\nTesting language detection...")
    # Test language detection
    for i, text in enumerate(test_data['text']):
        is_tag = is_tagalog(text)
        print(f"{i+1}. '{text}' - Is Tagalog: {is_tag}")
    
    print("\nTesting English sentiment analysis...")
    # Test standard English sentiment
    english_results = []
    for text in test_data['text']:
        result = analyze_sentiment_vader(text)
        english_results.append(result)
        print(f"VADER: '{text}' -> {result}")
    
    # Extract just the sentiment label
    english_labels = [extract_sentiment_label(result) for result in english_results]
    
    print("\nTesting Tagalog sentiment analysis...")
    # Test Tagalog sentiment
    tagalog_results = []
    for text in test_data['text']:
        result = tagalog_enhanced_sentiment_analysis(text)
        tagalog_results.append(result)
        print(f"Tagalog: '{text}' -> {result}")
    
    # Extract just the sentiment label
    tagalog_labels = [extract_sentiment_label(result) for result in tagalog_results]
    
    print("\nResults comparison:")
    print("Text | Expected | English | Tagalog")
    print("-" * 50)
    for i, row in test_data.iterrows():
        print(f"{row['text'][:20]}... | {row['sentiment']} | {english_labels[i]} | {tagalog_labels[i]}")
    
    # Calculate accuracy
    english_accuracy = sum(english_labels[i] == test_data['sentiment'][i] for i in range(len(test_data))) / len(test_data)
    tagalog_accuracy = sum(tagalog_labels[i] == test_data['sentiment'][i] for i in range(len(test_data))) / len(test_data)
    
    print(f"\nEnglish sentiment accuracy: {english_accuracy:.2f}")
    print(f"Tagalog sentiment accuracy: {tagalog_accuracy:.2f}")
    
    # Create a simple bar chart
    try:
        plt.figure(figsize=(8, 5))
        plt.bar(['English', 'Tagalog'], [english_accuracy, tagalog_accuracy])
        plt.title('Sentiment Analysis Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the chart
        plt.savefig("tagalog_test_results.png")
        print("\nResults chart saved as 'tagalog_test_results.png'")
        
        # Show the chart
        plt.show()
    except Exception as e:
        print(f"Error creating chart: {e}")
    
    print("\nTest completed successfully.")

# Run the test
if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()