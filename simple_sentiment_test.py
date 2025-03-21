import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Import your functions - modify this path to match your file name
try:
    from sentiment_analysis import (
        analyze_sentiment_vader,
        train_mnb_model,
        combined_sentiment_analysis,
        enhanced_sentiment_analysis
    )
    print("Successfully imported sentiment analysis functions.")
except ImportError as e:
    print(f"Error importing sentiment analysis module: {e}")
    print("Please make sure your file is named 'sentiment_analysis.py' or update the import statement.")
    exit(1)

def create_test_data(num_samples=50):
    """Create simple test data for sentiment analysis."""
    print(f"Creating {num_samples} test examples...")
    
    # Positive examples
    positive_examples = [
        "This video is fire! 🔥 Love your content",
        "I've been waiting for this! Thanks for sharing 💕",
        "This looks amazing! Where did you get it?",
        "This is so helpful! Just what I needed today 🙏",
        "Absolutely love this! Been following you since day one 💯",
        "First time seeing someone do this right!",
        "Can't believe you made this work! So impressive",
        "What brand is this? I need it",
        "This made my day! Can't stop watching it on repeat",
        "Been struggling with this for weeks, your tips helped!",
        "Nailed it! Exactly how it should be done",
        "This changed my life! Thank you so much",
        "Love the way you explained this! So clear and helpful 👏",
        "This literally saved me hours of work thank you!",
        "You look gorgeous! Where did you get that outfit?",
        "Watching this on repeat! Can't get enough 🔥",
        "This is giving me life right now 😍",
        "Obsessed with this look! Tutorial please!",
        "You're so talented! I wish I could do this half as well",
    ]
    
    # Negative examples
    negative_examples = [
        "Worst tutorial I've ever seen 😠",
        "Why would anyone use this product?",
        "Mid content as usual 😴",
        "I don't get why people like this",
        "Straight up stole this idea from another creator",
        "The way you explained this is so confusing",
        "I tried this and it didn't work at all. Waste of time",
        "This trend is getting old tbh",
        "Did you seriously think this was good enough to post?",
        "Unfollowing after this disaster",
        "No hate but this isn't it chief",
        "Just another copied idea nothing special",
        "Stop trying to make this happen it's not going to happen",
        "Been doing this for years and this is still wrong",
        "This isn't even remotely accurate information",
    ]
    
    # Neutral examples
    neutral_examples = [
        "Not sure what this is about tbh",
        "Can someone explain what's happening here?",
        "Is this available in other colors?",
        "Cool but could be better",
        "Is there a tutorial on how to do this?",
        "Not my style but I can see why others like it",
        "Can you make this in tutorial form?",
        "Just checking the comments to see if it's worth watching",
        "Anyone else notice the mistake at 0:45?",
        "Hmm, not sure if this is worth the hype",
        "How long did this take you to make?",
        "Is anyone else having trouble with step 3?",
        "I'm confused about the part at 2:15",
        "Has anyone tested if this actually works?",
        "Nice video but the audio quality could be better",
        "What camera do you use? The quality is amazing",
        "Idk about this one... seems kinda sus",
    ]
    
    # Calculate how many samples to take from each category for a total of num_samples
    # Aiming for roughly equal distribution
    samples_per_category = num_samples // 3
    remainder = num_samples % 3
    
    # Distribute any remainder
    category_counts = [samples_per_category] * 3
    for i in range(remainder):
        category_counts[i] += 1
    
    # Sample with replacement to get enough examples
    positive_samples = np.random.choice(positive_examples, size=category_counts[0], replace=True)
    negative_samples = np.random.choice(negative_examples, size=category_counts[1], replace=True)
    neutral_samples = np.random.choice(neutral_examples, size=category_counts[2], replace=True)
    
    # Create the dataframe
    texts = np.concatenate([positive_samples, negative_samples, neutral_samples])
    sentiments = (['Positive'] * len(positive_samples) + 
                 ['Negative'] * len(negative_samples) + 
                 ['Neutral'] * len(neutral_samples))
    
    # Shuffle the data
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    
    test_df = pd.DataFrame({
        'text': texts[idx],
        'sentiment': np.array(sentiments)[idx]
    })
    
    print(f"Test data created with {len(test_df)} examples.")
    return test_df

def extract_sentiment_label(result):
    """Extract sentiment label from result string."""
    if not result or not isinstance(result, str):
        return "Neutral"
        
    if result.startswith("Positive"):
        return "Positive"
    elif result.startswith("Negative"):
        return "Negative"
    else:
        return "Neutral"

def test_vader(test_data):
    """Test VADER sentiment analysis."""
    print("Testing VADER sentiment analysis...")
    
    # Apply VADER
    vader_results = []
    for text in test_data['text']:
        try:
            result = analyze_sentiment_vader(text)
            vader_results.append(result)
        except Exception as e:
            print(f"Error analyzing text with VADER: {e}")
            vader_results.append("Neutral (0.00)")
    
    # Convert to Series
    vader_results = pd.Series(vader_results)
    
    # Extract sentiment labels
    vader_labels = vader_results.apply(extract_sentiment_label)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_data['sentiment'], vader_labels)
    
    print(f"VADER analysis completed with accuracy: {accuracy:.4f}")
    
    return {
        'predictions': vader_labels,
        'raw_results': vader_results,
        'accuracy': accuracy
    }

def test_mnb(test_data):
    """Test MNB sentiment analysis."""
    print("Testing MNB sentiment analysis...")
    
    try:
        # Apply MNB
        mnb_results = train_mnb_model(test_data['text'])
        
        # Extract sentiment labels
        mnb_labels = mnb_results.apply(extract_sentiment_label)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_data['sentiment'], mnb_labels)
        
        print(f"MNB analysis completed with accuracy: {accuracy:.4f}")
        
        return {
            'predictions': mnb_labels,
            'raw_results': mnb_results,
            'accuracy': accuracy
        }
    except Exception as e:
        print(f"Error during MNB analysis: {e}")
        return None

def quick_test():
    """Run simplified tests on sentiment analysis methods."""
    print("Starting quick sentiment analysis test...")
    
    # Create test data with 50 examples (modified from original 30)
    test_data = create_test_data(num_samples=50)
    
    # Show a few examples
    print("\nSample test data:")
    print(test_data.head(10))  # Just show first 10 to keep output manageable
    
    # Test VADER
    vader_results = test_vader(test_data)
    
    # Test MNB
    mnb_results = test_mnb(test_data)
    
    # Compare results
    results = {'VADER': vader_results['accuracy']}
    
    if mnb_results:
        results['MNB'] = mnb_results['accuracy']
    
    # Display results
    print("\nAccuracy Comparison:")
    for method, accuracy in results.items():
        print(f"{method}: {accuracy:.4f}")
    
    # Create simple bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values())
    plt.title('Sentiment Analysis Methods Accuracy (50 Examples)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    try:
        plt.savefig("sentiment_test_results.png")
        print("\nResults chart saved as 'sentiment_test_results.png'")
    except Exception as e:
        print(f"Could not save chart: {e}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display chart: {e}")
    
    print("\nTest with 50 examples completed successfully.")

if __name__ == "__main__":
    quick_test()