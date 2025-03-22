import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Import your sentiment analysis functions
from sentiment_analysis import (
    analyze_sentiment_vader,
    train_mnb_model,
    enhanced_sentiment_analysis,
    combined_sentiment_analysis,
    get_sentiment_breakdown
)

class SentimentTester:
    """Test framework for sentiment analysis models comparison."""
    
    def __init__(self, test_data_path=None):
        """Initialize the tester with dataset parameters."""
        self.test_data_path = test_data_path
        self.test_data = None
        self.results = {}
        
    def load_test_data(self, data_path=None):
        """Load test dataset (with ground truth labels)."""
        if data_path is not None:
            self.test_data_path = data_path
            
        if self.test_data_path.endswith('.csv'):
            self.test_data = pd.read_csv(self.test_data_path)
        elif self.test_data_path.endswith('.json'):
            self.test_data = pd.read_json(self.test_data_path)
        elif isinstance(self.test_data_path, pd.DataFrame):
            self.test_data = self.test_data_path
        else:
            raise ValueError("Unsupported data format. Please provide a CSV, JSON, or DataFrame.")
            
        # Validate test data has required columns
        required_columns = ['text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]
        if missing_columns:
            raise ValueError(f"Test data missing required columns: {missing_columns}")
            
        return self.test_data
    
    def create_test_data(self, num_samples=100, save_path=None):
        """
        Create test data for sentiment analysis if none is available.
        This generates fake data based on simple patterns.
        """
        # Positive examples with TikTok-specific vocabulary
        positive_examples = [
            "i love this", "this is amazing", "great video", "awesome content", "love it",
            "this is fire", "absolutely slaying", "this is so based",
            "vibes are immaculate", "period queen", "talented af"
        ]
        
        # Negative examples with TikTok-specific vocabulary
        negative_examples = [
            "i hate this", "this is terrible", "awful video", "bad content", "dislike it",
            "this is cringe", "major flop", "mid at best", "giving me the ick",
            "pure trash", "annoying af"
        ]
        
        # Neutral examples with TikTok-specific vocabulary
        neutral_examples = [
            "okay", "not sure", "maybe", "average", "alright", "idk about this",
            "just scrolling", "pov: me watching", "no thoughts"
        ]
        
        # Sample with replacement to get enough examples
        positive_samples = np.random.choice(positive_examples, size=num_samples//3, replace=True)
        negative_samples = np.random.choice(negative_examples, size=num_samples//3, replace=True)
        neutral_samples = np.random.choice(neutral_examples, size=num_samples//3, replace=True)
        
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
        
        # Save if requested
        if save_path:
            test_df.to_csv(save_path, index=False)
            
        self.test_data = test_df
        return test_df
    
    def _extract_sentiment_label(self, result):
        """Extract sentiment label from result string."""
        if not result or not isinstance(result, str):
            return "Neutral"
            
        if result.startswith("Positive"):
            return "Positive"
        elif result.startswith("Negative"):
            return "Negative"
        else:
            return "Neutral"
    
    def _extract_sentiment_score(self, result):
        """Extract sentiment score from result string."""
        if not result or not isinstance(result, str):
            return 0.0
            
        # Extract score value inside parentheses
        match = re.search(r'\(([-+]?\d+\.\d+)\)', result)
        if match:
            return float(match.group(1))
        else:
            return 0.0
    
    def test_vader(self):
        """Test VADER sentiment analysis."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data first.")
            
        # Apply VADER
        vader_results = self.test_data['text'].apply(analyze_sentiment_vader)
        
        # Extract sentiment labels
        vader_labels = vader_results.apply(self._extract_sentiment_label)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_data['sentiment'], vader_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.test_data['sentiment'], vader_labels, average='weighted'
        )
        
        # Store results
        self.results['vader'] = {
            'predictions': vader_labels,
            'raw_results': vader_results,
            'true_values': self.test_data['sentiment'],
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(self.test_data['sentiment'], vader_labels),
                'classification_report': classification_report(self.test_data['sentiment'], vader_labels)
            }
        }
        
        return self.results['vader']
    
    def test_mnb(self):
        """Test MNB sentiment analysis."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data first.")
            
        # Apply MNB
        mnb_results = train_mnb_model(self.test_data['text'])
        
        # Extract sentiment labels
        mnb_labels = mnb_results.apply(self._extract_sentiment_label)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_data['sentiment'], mnb_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.test_data['sentiment'], mnb_labels, average='weighted'
        )
        
        # Store results
        self.results['mnb'] = {
            'predictions': mnb_labels,
            'raw_results': mnb_results,
            'true_values': self.test_data['sentiment'],
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(self.test_data['sentiment'], mnb_labels),
                'classification_report': classification_report(self.test_data['sentiment'], mnb_labels)
            }
        }
        
        return self.results['mnb']
    
    def test_combined(self):
        """Test combined sentiment analysis."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data first.")
            
        # Apply combined approach
        combined_results = combined_sentiment_analysis(self.test_data['text'])
        
        # Extract sentiment labels
        combined_labels = combined_results.apply(self._extract_sentiment_label)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_data['sentiment'], combined_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.test_data['sentiment'], combined_labels, average='weighted'
        )
        
        # Store results
        self.results['combined'] = {
            'predictions': combined_labels,
            'raw_results': combined_results,
            'true_values': self.test_data['sentiment'],
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(self.test_data['sentiment'], combined_labels),
                'classification_report': classification_report(self.test_data['sentiment'], combined_labels)
            }
        }
        
        return self.results['combined']
    
    def test_enhanced(self):
        """Test enhanced sentiment analysis."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data first.")
            
        # Apply enhanced approach
        enhanced_results = enhanced_sentiment_analysis(self.test_data['text'])
        
        # Extract sentiment labels
        enhanced_labels = enhanced_results.apply(self._extract_sentiment_label)
        
        # Calculate metrics
        accuracy = accuracy_score(self.test_data['sentiment'], enhanced_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.test_data['sentiment'], enhanced_labels, average='weighted'
        )
        
        # Store results
        self.results['enhanced'] = {
            'predictions': enhanced_labels,
            'raw_results': enhanced_results,
            'true_values': self.test_data['sentiment'],
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(self.test_data['sentiment'], enhanced_labels),
                'classification_report': classification_report(self.test_data['sentiment'], enhanced_labels)
            }
        }
        
        return self.results['enhanced']
    
    def run_all_tests(self):
        """Run all sentiment analysis tests."""
        if self.test_data is None:
            raise ValueError("Test data not loaded. Call load_test_data first.")
            
        self.test_vader()
        self.test_mnb()
        self.test_combined()
        self.test_enhanced()
        
        return self.results
    
    def compare_methods(self):
        """Compare the performance of all tested methods."""
        if not self.results:
            raise ValueError("No test results available. Run tests first.")
            
        # Collect metrics from all methods
        methods = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        comparison = pd.DataFrame(index=methods, columns=metrics)
        
        for method in methods:
            for metric in metrics:
                comparison.loc[method, metric] = self.results[method]['metrics'][metric]
                
        return comparison
    
    def plot_comparison(self, save_path=None):
        """Visualize the comparison between methods."""
        comparison = self.compare_methods()
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        comparison.plot(kind='bar', figsize=(12, 6))
        plt.title('Sentiment Analysis Methods Comparison')
        plt.ylabel('Score')
        plt.xlabel('Method')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metric')
        
        if save_path:
            plt.savefig(save_path)
            
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrices
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        for i, (method, result) in enumerate(self.results.items()):
            cm = result['metrics']['confusion_matrix']
            if n_methods > 1:
                ax = axes[i]
            else:
                ax = axes
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{method.upper()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.', '_cm.'))
            
        plt.show()
        
        return comparison
    
    def analyze_misclassifications(self, method='all'):
        """Analyze misclassified examples for a specific method or all methods."""
        if not self.results:
            raise ValueError("No test results available. Run tests first.")
            
        if method != 'all' and method not in self.results:
            raise ValueError(f"Method {method} not found in results. Available methods: {list(self.results.keys())}")
            
        methods_to_analyze = list(self.results.keys()) if method == 'all' else [method]
        
        all_misclassifications = {}
        
        for m in methods_to_analyze:
            true_values = self.results[m]['true_values']
            predictions = self.results[m]['predictions']
            
            # Find misclassified examples
            misclassified_idx = true_values != predictions
            
            misclassified_examples = pd.DataFrame({
                'text': self.test_data.loc[misclassified_idx, 'text'],
                'true_sentiment': true_values[misclassified_idx],
                'predicted_sentiment': predictions[misclassified_idx],
                'raw_result': self.results[m]['raw_results'][misclassified_idx]
            })
            
            all_misclassifications[m] = misclassified_examples
            
        return all_misclassifications
    
    def error_analysis(self, save_path=None):
        """Perform detailed error analysis across methods."""
        misclassifications = self.analyze_misclassifications()
        
        # Calculate error rates by true class
        error_by_class = {}
        
        for method, misclass_df in misclassifications.items():
            # Count by true sentiment
            error_counts = misclass_df['true_sentiment'].value_counts()
            
            # Get total counts in test data
            total_counts = self.test_data['sentiment'].value_counts()
            
            # Calculate error rate
            error_rates = {}
            for sentiment in total_counts.index:
                if sentiment in error_counts:
                    error_rates[sentiment] = error_counts[sentiment] / total_counts[sentiment]
                else:
                    error_rates[sentiment] = 0.0
                    
            error_by_class[method] = error_rates
            
        # Convert to DataFrame for visualization
        error_df = pd.DataFrame(error_by_class)
        
        # Plot error rates
        plt.figure(figsize=(12, 6))
        error_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Error Rates by Sentiment Class')
        plt.ylabel('Error Rate')
        plt.xlabel('True Sentiment')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        
        if save_path:
            plt.savefig(save_path)
            
        plt.tight_layout()
        plt.show()
        
        return error_df, misclassifications


# Pytest functions for testing sentiment analysis methods

def test_vader_accuracy_threshold():
    """Test if VADER accuracy meets minimum threshold."""
    # Create synthetic test data if no real data is available
    tester = SentimentTester()
    tester.create_test_data(num_samples=100)
    vader_results = tester.test_vader()
    
    # Test that accuracy meets minimum threshold (adjust as needed)
    assert vader_results['metrics']['accuracy'] >= 0.6, "VADER accuracy below threshold"

def test_mnb_accuracy_threshold():
    """Test if MNB accuracy meets minimum threshold."""
    # Create synthetic test data if no real data is available
    tester = SentimentTester()
    tester.create_test_data(num_samples=100)
    mnb_results = tester.test_mnb()
    
    # Test that accuracy meets minimum threshold (adjust as needed)
    assert mnb_results['metrics']['accuracy'] >= 0.6, "MNB accuracy below threshold"

def test_combined_accuracy_threshold():
    """Test if combined approach accuracy meets minimum threshold."""
    # Create synthetic test data if no real data is available
    tester = SentimentTester()
    tester.create_test_data(num_samples=100)
    combined_results = tester.test_combined()
    
    # Test that accuracy meets minimum threshold (adjust as needed)
    assert combined_results['metrics']['accuracy'] >= 0.7, "Combined approach accuracy below threshold"

def test_enhanced_accuracy_threshold():
    """Test if enhanced approach accuracy meets minimum threshold."""
    # Create synthetic test data if no real data is available
    tester = SentimentTester()
    tester.create_test_data(num_samples=100)
    enhanced_results = tester.test_enhanced()
    
    # Test that accuracy meets minimum threshold (adjust as needed)
    assert enhanced_results['metrics']['accuracy'] >= 0.7, "Enhanced approach accuracy below threshold"

def test_comparative_performance():
    """Test if enhanced approach outperforms others."""
    # Create synthetic test data if no real data is available
    tester = SentimentTester()
    tester.create_test_data(num_samples=100)
    tester.run_all_tests()
    comparison = tester.compare_methods()
    
    # Test that enhanced or combined approaches outperform basic approaches
    best_method = comparison['accuracy'].idxmax()
    assert best_method in ['enhanced', 'combined'], "Advanced approaches not outperforming basic ones"

def test_specific_cases():
    """Test performance on specific challenging examples."""
    # Define some challenging examples
    challenging_examples = pd.DataFrame({
        'text': [
            "This is fire but the lighting is mid",  # Mixed sentiment
            "Not really feeling this tbh :/",        # Subtle negative
            "Idk if this is based or cringe",        # Ambiguous
            "yikes... but go off I guess",           # Mixed negative/neutral
            "sheesh 🔥 bussin fr no cap"             # Slang + emoji positive
        ],
        'sentiment': [
            "Positive",  # Overall still positive
            "Negative",  # Subtle negative
            "Neutral",   # Truly ambiguous
            "Negative",  # More negative than positive
            "Positive"   # Clearly positive with slang
        ]
    })
    
    # Create tester with these examples
    tester = SentimentTester(challenging_examples)
    
    # Run all tests
    tester.run_all_tests()
    
    # Analyze the results for specific examples
    misclassifications = tester.analyze_misclassifications()
    
    # Check if enhanced model performs better on these edge cases
    enhanced_misclassified = len(misclassifications.get('enhanced', pd.DataFrame()))
    vader_misclassified = len(misclassifications.get('vader', pd.DataFrame()))
    
    # Enhanced should handle these challenging cases better
    assert enhanced_misclassified <= vader_misclassified, "Enhanced model not handling challenging cases better"


# Main function to run the tests
def main():
    """Run the sentiment analysis tests and display results."""
    # Define test data path, or None to create synthetic data
    test_data_path = None  # "sentiment_test_data.csv"
    
    # Initialize tester
    tester = SentimentTester(test_data_path)
    
    # Load or create test data
    if test_data_path:
        tester.load_test_data()
    else:
        tester.create_test_data(num_samples=200, save_path="synthetic_test_data.csv")
        
    # Run all tests
    tester.run_all_tests()
    
    # Display comparison
    comparison = tester.compare_methods()
    print("\nPerformance Comparison:")
    print(comparison)
    
    # Plot results
    tester.plot_comparison(save_path="sentiment_comparison.png")
    
    # Error analysis
    error_rates, misclassifications = tester.error_analysis(save_path="error_analysis.png")
    
    print("\nError Rates by Class:")
    print(error_rates)
    
    # Display sample misclassifications for each method
    for method, misclass_df in misclassifications.items():
        print(f"\nSample {method.upper()} Misclassifications (top 5):")
        if not misclass_df.empty:
            print(misclass_df.head())
    
    # Run pytest tests
    print("\nRunning pytest tests...")
    pytest.main(["-xvs", __file__])

if __name__ == "__main__":
    main()

    # Add at the beginning of main()
print("Starting sentiment analysis tests...")
# Add before each major step
print("Creating test data...")
print("Running VADER tests...")
# etc.