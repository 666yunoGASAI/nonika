import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Import your functions
print("starting here")
try:
    from sentiment_analysis import (
        analyze_sentiment_vader,
        train_mnb_model,
        combined_sentiment_analysis,
        enhanced_sentiment_analysis
    )
    from tagalog_sentiment import (
        is_tagalog,
        tagalog_enhanced_sentiment_analysis,
        get_tagalog_sentiment_breakdown
    )
    print("Successfully imported sentiment analysis functions.")
except ImportError as e:
    print(f"Error importing sentiment analysis modules: {e}")
    print("Please make sure your files are named correctly or update the import statements.")
    exit(1)

def create_tagalog_test_data(num_samples=50):
    """Create test data for Tagalog sentiment analysis."""
    print(f"Creating {num_samples} Tagalog test examples...")
    
    # Positive examples in Tagalog
    positive_examples = [
        "Ang ganda ng video mo! 🔥 Sobrang galing!",
        "Matagal ko nang hinihintay to! Salamat sa pag-share 💕",
        "Ang galing naman nito! Saan mo nabili yan?",
        "Nakakatulong to! Ito talaga yung kailangan ko ngayon 🙏",
        "Sobrang bet ko to! Sumusubaybay ako sayo simula pa nung una 💯",
        "First time kong makakita ng ganito na tama ang pagkakagawa!",
        "Di ko mapaniniwalan na nagawa mo to! Napaka-impressive",
        "Anong brand ito? Kailangan ko talaga to",
        "Nagpaligaya to ng araw ko! Di ako magsasawang panoorin paulit-ulit",
        "Ilang linggo na akong nahihirapan dito, nakatulong yung tips mo!",
        "Ang husay! Ganyan dapat ang pagkakagawa",
        "Nagbago ng buhay ko to! Maraming salamat",
        "Ang ganda ng pag-explain mo! Sobrang clear at helpful 👏",
        "Nakatipid ako ng ilang oras dahil dito, salamat!",
        "Ang ganda mo! Saan mo nabili yung outfit mo?",
        "Paulit-ulit kong pinapanood! Di ako nagsasawa 🔥",
        "Nakakabuhay ng diwa to 😍",
        "Obsessed ako sa look na to! Tutorial naman please!",
        "Ang galing mo! Sana kahit kalahati lang ng galing mo kaya ko din",
    ]
    
    # Negative examples in Tagalog
    negative_examples = [
        "Pinakamasama na tutorial na nakita ko 😠",
        "Bakit may gagamit pa ng produktong to?",
        "Pang-mid content na naman tulad ng dati 😴",
        "Di ko maintindihan kung bakit gusto to ng mga tao",
        "Halatang ninakaw lang tong idea na to sa ibang creator",
        "Ang labo ng pag-explain mo, nakakahilo",
        "Sinubukan ko 'to pero di naman gumagana. Sayang oras ko",
        "Nagsasawa na ko sa trend na to tbh",
        "Seryoso ba? Akala mo good enough to para i-post?",
        "Unfollow na ko after ng disaster na to",
        "No hate pero mali to",
        "Kopya lang 'to ng ibang idea, walang special",
        "Tama na, wag na pilitin 'to di naman magkakatotoo",
        "Ilang taon ko na 'tong ginagawa pero mali pa rin to",
        "Fake news to, di naman accurate ang information",
    ]
    
    # Neutral examples in Tagalog
    neutral_examples = [
        "Di ko sure kung ano to tbh",
        "Pwede bang may mag-explain kung ano nangyayari dito?",
        "Available ba to sa ibang kulay?",
        "Okay naman pero pwede pang pagandahin",
        "May tutorial ba kung paano gawin to?",
        "Hindi ko trip pero gets ko kung bakit gusto ng iba",
        "Pwede mo bang gawing tutorial format to?",
        "Chinecheck ko lang comments kung worth it panoorin",
        "May napansin ba kayong mali sa 0:45?",
        "Hmm, di ko sure kung worth it yung hype",
        "Gaano katagal mo ginawa to?",
        "May nahihirapan ba sa step 3?",
        "Nalilito ako sa part sa 2:15",
        "May nakatest na ba kung talagang gumagana to?",
        "Okay naman yung video pero pangit yung audio quality",
        "Anong camera gamit mo? Ang ganda ng quality",
        "Di ko sure dito... parang sus",
    ]
    
    # Mixed language (Taglish) examples
    taglish_examples = [
        "Super nice naman 'to! Love it talaga",
        "So helpful ng tutorial mo, salamat sa tips!",
        "Ang creative ng idea mo, I'm impressed!",
        "This is exactly what I need, sobrang helpful",
        "Hindi worth it, waste of time lang 'to",
        "So boring naman nito, nakakatulog ako",
        "Ang annoying ng boses mo, sorry pero totoo",
        "Disappointing 'to, wala namang substance",
        "Not sure if this works, try ko muna",
        "How much ito? Pahingi naman ng link",
        "Let me think about it muna, medyo pricey eh",
        "Any other colors available? Gusto ko sana yung blue",
    ]
    
    # Calculate samples from each category
    base_samples = num_samples // 4  # 4 categories including Taglish
    remainder = num_samples % 4
    
    # Distribute samples with any remainder going to the first categories
    category_counts = [base_samples] * 4
    for i in range(remainder):
        category_counts[i] += 1
    
    # Sample with replacement to get enough examples
    positive_samples = np.random.choice(positive_examples, size=category_counts[0], replace=True)
    negative_samples = np.random.choice(negative_examples, size=category_counts[1], replace=True)
    neutral_samples = np.random.choice(neutral_examples, size=category_counts[2], replace=True)
    taglish_samples = np.random.choice(taglish_examples, size=category_counts[3], replace=True)
    
    # Create labels for Taglish examples (half positive, half negative/neutral)
    taglish_mid = len(taglish_samples) // 2
    taglish_sentiments = ['Positive'] * taglish_mid + \
                         (['Negative'] * (taglish_mid // 2)) + \
                         (['Neutral'] * (len(taglish_samples) - taglish_mid - (taglish_mid // 2)))
    
    # Create the dataframe
    texts = np.concatenate([positive_samples, negative_samples, neutral_samples, taglish_samples])
    sentiments = (
        ['Positive'] * len(positive_samples) + 
        ['Negative'] * len(negative_samples) + 
        ['Neutral'] * len(neutral_samples) +
        taglish_sentiments
    )
    
    # Add language indicators
    languages = (
        ['Tagalog'] * len(positive_samples) + 
        ['Tagalog'] * len(negative_samples) + 
        ['Tagalog'] * len(neutral_samples) +
        ['Taglish'] * len(taglish_samples)
    )
    
    # Shuffle the data
    idx = np.arange(len(texts))
    np.random.shuffle(idx)
    
    test_df = pd.DataFrame({
        'text': texts[idx],
        'sentiment': np.array(sentiments)[idx],
        'language': np.array(languages)[idx]
    })
    
    print(f"Tagalog test data created with {len(test_df)} examples.")
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

def test_english_methods_on_tagalog(test_data):
    """Test standard English sentiment methods on Tagalog text."""
    print("Testing English sentiment methods on Tagalog text...")
    
    results = {}
    
    # Test VADER on Tagalog
    print("Testing VADER on Tagalog...")
    vader_results = []
    for text in test_data['text']:
        try:
            result = analyze_sentiment_vader(text)
            vader_results.append(result)
        except Exception as e:
            print(f"Error analyzing text with VADER: {e}")
            vader_results.append("Neutral (0.00)")
    
    vader_results = pd.Series(vader_results)
    vader_labels = vader_results.apply(extract_sentiment_label)
    vader_accuracy = accuracy_score(test_data['sentiment'], vader_labels)
    print(f"VADER on Tagalog accuracy: {vader_accuracy:.4f}")
    results['VADER'] = vader_accuracy
    
    # Test MNB on Tagalog
    try:
        print("Testing MNB on Tagalog...")
        mnb_results = train_mnb_model(test_data['text'])
        mnb_labels = mnb_results.apply(extract_sentiment_label)
        mnb_accuracy = accuracy_score(test_data['sentiment'], mnb_labels)
        print(f"MNB on Tagalog accuracy: {mnb_accuracy:.4f}")
        results['MNB'] = mnb_accuracy
    except Exception as e:
        print(f"Error during MNB analysis: {e}")
    
    # Test Combined on Tagalog
    try:
        print("Testing Combined approach on Tagalog...")
        combined_results = []
        for text in test_data['text']:
            result = combined_sentiment_analysis(text)
            combined_results.append(result)
        combined_results = pd.Series(combined_results)
        combined_labels = combined_results.apply(extract_sentiment_label)
        combined_accuracy = accuracy_score(test_data['sentiment'], combined_labels)
        print(f"Combined approach on Tagalog accuracy: {combined_accuracy:.4f}")
        results['Combined'] = combined_accuracy
    except Exception as e:
        print(f"Error during Combined analysis: {e}")
    
    # Test Enhanced on Tagalog
    try:
        print("Testing Enhanced sentiment on Tagalog...")
        enhanced_results = []
        for text in test_data['text']:
            result = enhanced_sentiment_analysis(text)
            enhanced_results.append(result)
        enhanced_results = pd.Series(enhanced_results)
        enhanced_labels = enhanced_results.apply(extract_sentiment_label)
        enhanced_accuracy = accuracy_score(test_data['sentiment'], enhanced_labels)
        print(f"Enhanced approach on Tagalog accuracy: {enhanced_accuracy:.4f}")
        results['Enhanced'] = enhanced_accuracy
    except Exception as e:
        print(f"Error during Enhanced analysis: {e}")
    
    return results

def test_tagalog_methods(test_data):
    """Test Tagalog-specific sentiment methods."""
    print("Testing Tagalog-specific sentiment methods...")
    
    # Test Tagalog-enhanced method
    tagalog_results = []
    for text in test_data['text']:
        try:
            result = tagalog_enhanced_sentiment_analysis(text)
            tagalog_results.append(result)
        except Exception as e:
            print(f"Error analyzing text with Tagalog method: {e}")
            tagalog_results.append("Neutral (0.00)")
    
    tagalog_results = pd.Series(tagalog_results)
    tagalog_labels = tagalog_results.apply(extract_sentiment_label)
    tagalog_accuracy = accuracy_score(test_data['sentiment'], tagalog_labels)
    print(f"Tagalog-enhanced method accuracy: {tagalog_accuracy:.4f}")
    
    # Check language detection accuracy
    detected_languages = test_data['text'].apply(is_tagalog)
    detection_accuracy = (detected_languages & (test_data['language'] == 'Tagalog')).mean()
    print(f"Language detection accuracy: {detection_accuracy:.4f}")
    
    return {
        'Tagalog-Enhanced': tagalog_accuracy,
        'Language Detection': detection_accuracy
    }

def compare_by_language_type(test_data, english_results, tagalog_results):
    """Compare results by language type (pure Tagalog vs Taglish)."""
    # Split the data by language
    pure_tagalog = test_data[test_data['language'] == 'Tagalog']
    taglish = test_data[test_data['language'] == 'Taglish']
    
    # Test on pure Tagalog
    print("\nAnalyzing pure Tagalog performance:")
    for method in list(english_results.keys()) + list(tagalog_results.keys()):
        # Skip language detection metric
        if method == 'Language Detection':
            continue
            
        # Get predictions for this method
        if method in english_results:
            # Standard English methods
            if method == 'VADER':
                preds = pure_tagalog['text'].apply(analyze_sentiment_vader)
            elif method == 'MNB':
                try:
                    preds = train_mnb_model(pure_tagalog['text'])
                except:
                    continue
            elif method == 'Combined':
                preds = pure_tagalog['text'].apply(combined_sentiment_analysis)
            elif method == 'Enhanced':
                preds = pure_tagalog['text'].apply(enhanced_sentiment_analysis)
        else:
            # Tagalog method
            preds = pure_tagalog['text'].apply(tagalog_enhanced_sentiment_analysis)
        
        # Calculate accuracy
        pred_labels = preds.apply(extract_sentiment_label)
        accuracy = accuracy_score(pure_tagalog['sentiment'], pred_labels)
        print(f"  {method} on pure Tagalog: {accuracy:.4f}")
    
    # Test on Taglish
    print("\nAnalyzing Taglish performance:")
    for method in list(english_results.keys()) + list(tagalog_results.keys()):
        # Skip language detection metric
        if method == 'Language Detection':
            continue
            
        # Get predictions for this method
        if method in english_results:
            # Standard English methods
            if method == 'VADER':
                preds = taglish['text'].apply(analyze_sentiment_vader)
            elif method == 'MNB':
                try:
                    preds = train_mnb_model(taglish['text'])
                except:
                    continue
            elif method == 'Combined':
                preds = taglish['text'].apply(combined_sentiment_analysis)
            elif method == 'Enhanced':
                preds = taglish['text'].apply(enhanced_sentiment_analysis)
        else:
            # Tagalog method
            preds = taglish['text'].apply(tagalog_enhanced_sentiment_analysis)
        
        # Calculate accuracy
        pred_labels = preds.apply(extract_sentiment_label)
        accuracy = accuracy_score(taglish['sentiment'], pred_labels)
        print(f"  {method} on Taglish: {accuracy:.4f}")

def tagalog_quick_test():
    """Run tests on Tagalog sentiment analysis methods."""
    print("Starting Tagalog sentiment analysis test...")
    
    # Create test data with Tagalog examples
    test_data = create_tagalog_test_data(num_samples=60)
    
    # Show a few examples
    print("\nSample Tagalog test data:")
    print(test_data.head(10))  # Just show first 10 to keep output manageable
    
    # Test English methods on Tagalog text
    english_results = test_english_methods_on_tagalog(test_data)
    
    # Test Tagalog-specific methods
    tagalog_results = test_tagalog_methods(test_data)
    
    # Compare methods by language type
    compare_by_language_type(test_data, english_results, tagalog_results)
    
    # Combine all results
    all_results = {**english_results, **tagalog_results}
    
    # Create bar chart comparing methods
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'blue', 'blue', 'blue', 'green', 'orange']
    plt.bar(all_results.keys(), all_results.values(), color=colors)
    plt.title('Sentiment Analysis Methods Accuracy on Tagalog Text')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add colors legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='English Methods'),
        Patch(facecolor='green', label='Tagalog Methods'),
        Patch(facecolor='orange', label='Language Detection')
    ]
    plt.legend(handles=legend_elements)
    
    # Save chart
    try:
        plt.savefig("tagalog_sentiment_test_results.png")
        print("\nResults chart saved as 'tagalog_sentiment_test_results.png'")
    except Exception as e:
        print(f"Could not save chart: {e}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display chart: {e}")
    
    print("\nTagalog sentiment analysis test completed successfully.")

if __name__ == "__main__":
    print("starting")
    tagalog_quick_test()