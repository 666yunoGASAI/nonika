import streamlit as st
import pandas as pd
import re
import emoji
from apify_client import ApifyClient
from sentiment_analysis import (
    TrollDetector,
    analyze_sentiment_vader, 
    train_mnb_model, 
    combined_sentiment_analysis,
    enhanced_sentiment_analysis,
    get_sentiment_breakdown,
    analyze_for_trolling
)
# Import Tagalog sentiment functions
from tagalog_sentiment import (
    is_tagalog,
    get_tagalog_sentiment_breakdown
)
from text_processing import clean_text, tokenize_and_remove_stopwords, extract_hashtags
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import io
import csv
import chardet  # You may need to pip install chardet

# ADD THESE LINES HERE
import sys
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# END OF ADDED LINES

from market_trend_analysis import (
    calculate_market_trend_score,
    plot_market_prediction,
    predict_purchase_volume,
    generate_market_trend_report,
    add_market_trends_tab,
    get_troll_risk_level
)
import random

troll_detector = TrollDetector()


# Load API keys
load_dotenv()
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Initialize Apify Client
client = ApifyClient(APIFY_API_TOKEN)

# Streamlit App Configuration
st.set_page_config(
    page_title="TikTok Sentiment Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("TikTok Sentiment Analysis")
st.caption("Market Trend Classification Dashboard")

# Force clear Streamlit cache
if st.button("Clear Cache and Reload"):
    st.cache_data.clear()
    st.experimental_rerun()

# Functions for language-aware sentiment analysis
def add_language_settings():
    """Add language settings to the sidebar."""
    st.sidebar.header("Language Settings")
    language_mode = st.sidebar.radio(
        "Select language analysis mode:",
        ["Auto-detect", "English Only", "Tagalog Only", "Multilingual (English + Tagalog)"],
        index=0
    )
    
    st.sidebar.markdown("""
    **Language Mode Information:**
    - **Auto-detect**: Automatically identifies language and applies appropriate analysis
    - **English Only**: Optimized for English TikTok comments
    - **Tagalog Only**: Optimized for Filipino/Tagalog comments
    - **Multilingual**: Best for mixed language content or code-switching
    """)
    
    # Store language preference in session state
    st.session_state.language_mode = language_mode
    
    return language_mode

def analyze_sentiment_with_language_preference(text, language_mode=None):
    """
    Analyze sentiment with language mode preference.
    Returns ONLY sentiment score, no troll data.
    """
    if language_mode is None:
        language_mode = st.session_state.get('language_mode', "Auto-detect")
    
    if language_mode == "Auto-detect":
        if is_tagalog(text):
            return analyze_tagalog_sentiment_score(text)
        else:
            return analyze_english_sentiment_score(text)
    elif language_mode == "English Only":
        return analyze_english_sentiment_score(text)
    elif language_mode == "Tagalog Only":
        return analyze_tagalog_sentiment_score(text)
    else:  # Multilingual mode
        return analyze_tagalog_sentiment_score(text)

def analyze_comment_with_trolling(text, language_mode=None):
    """
    Analyzes comment for both sentiment and troll detection.
    Returns completely separate results.
    """
    # Get clean sentiment score (-1 to 1)
    sentiment_score = analyze_sentiment_score(text)
    
    # Get troll analysis separately
    troll_analysis = analyze_for_trolling(text)
    
    # Return completely separate results
    return {
        'sentiment_score': sentiment_score,  # Just the raw score
        'is_troll': troll_analysis['is_troll'],
        'troll_score': troll_analysis['troll_score']
    }

def get_sentiment_breakdown_with_language(text, language_mode=None):
    """
    Get sentiment breakdown with language preference.
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Sentiment breakdown dictionary
    """
    if language_mode is None:
        language_mode = st.session_state.get('language_mode', "Auto-detect")
    
    if language_mode == "Auto-detect":
        # Auto-detect language and apply appropriate breakdown
        if is_tagalog(text):
            return get_tagalog_sentiment_breakdown(text)
        else:
            return get_sentiment_breakdown(text)  # Your existing function
    
    elif language_mode == "English Only":
        # Force English breakdown
        return get_sentiment_breakdown(text)  # Your existing function
    
    elif language_mode == "Tagalog Only":
        # Force Tagalog breakdown
        return get_tagalog_sentiment_breakdown(text)
    
    else:  # Multilingual mode
        # Always use the tagalog breakdown which includes multilingual capabilities
        return get_tagalog_sentiment_breakdown(text)

# Function to fetch comments from TikTok
def fetch_tiktok_comments(video_link, max_comments=1000):
    """Fetches comments from a TikTok video using Apify."""
    run_input = {"postURLs": [video_link], "commentsPerPost": max_comments, "maxRepliesPerComment": 0}
    
    try:
        run = client.actor("BDec00yAmCm1QbMEI").call(run_input=run_input)
    except Exception as e:
        st.error(f"Error calling Apify actor: {e}")
        return None
    
    # Get items from dataset
    items = []
    try:
        items = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    
    # Create DataFrame
    if items:
        df = pd.DataFrame(items)
        # Select relevant columns if they exist
        columns = ['text']
        if 'likes' in df.columns:
            columns.append('likes')
        if 'username' in df.columns:
            columns.append('username')
        if 'created_at' in df.columns:
            columns.append('created_at')
            
        df = df[columns].rename(columns={'text': 'Comment'})
        return df
    return None

# Function to read files in multiple formats (XLSX and CSV)
def read_file_with_multiple_formats(uploaded_file):
    """
    Reads an uploaded file that could be either XLSX or CSV with various formats.
    
    Parameters:
    uploaded_file (UploadedFile): The file uploaded through Streamlit's file_uploader
    
    Returns:
    pandas.DataFrame: DataFrame containing the file data, or None if processing failed
    """
    if uploaded_file is None:
        return None
    
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Process based on file type
        if file_extension in ['xlsx', 'xls']:
            # For Excel files
            try:
                # Try standard pandas excel reader
                df = pd.read_excel(uploaded_file)
            except Exception as excel_error:
                st.warning(f"Standard Excel reader failed: {excel_error}. Trying alternative engines...")
                
                try:
                    # Try with openpyxl engine for .xlsx files
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception:
                    # Try with xlrd engine for .xls files
                    try:
                        df = pd.read_excel(uploaded_file, engine='xlrd')
                    except Exception as e:
                        st.error(f"Failed to read Excel file with all available engines: {e}")
                        return None
        
        elif file_extension == 'csv':
            # For CSV files - Improved approach for encoding issues
            
            # Read the file content as bytes to detect encoding
            file_content = uploaded_file.read()
            
            # Use chardet to detect encoding
            detection_result = chardet.detect(file_content)
            detected_encoding = detection_result['encoding']
            confidence = detection_result['confidence']
            
            st.info(f"Detected encoding: {detected_encoding} with confidence: {confidence:.2f}")
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Try common encodings in order of likelihood
            encodings_to_try = [
                detected_encoding,  # Try detected encoding first
                'utf-8',
                'latin1',
                'iso-8859-1',
                'cp1252',
                'utf-16',
                'utf-32',
                'utf-8-sig'  # UTF-8 with BOM
            ]
            
            # Remove None or invalid encodings
            encodings_to_try = [enc for enc in encodings_to_try if enc]
            
            # Try each encoding
            for encoding in encodings_to_try:
                try:
                    # Reset file pointer each time
                    uploaded_file.seek(0)
                    
                    # Try with comma delimiter
                    df = pd.read_csv(
                        uploaded_file, 
                        encoding=encoding, 
                        on_bad_lines='warn',  # More lenient with bad lines
                        low_memory=False      # Better for mixed data types
                    )
                    st.success(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    # If we get a unicode error, try next encoding
                    continue
                except Exception as e:
                    # For other exceptions, try different delimiters with this encoding
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file, 
                            sep=None,  # Try to auto-detect separator
                            engine='python', 
                            encoding=encoding
                        )
                        st.success(f"Successfully read CSV with {encoding} encoding and auto-detected delimiter")
                        break
                    except Exception:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(
                                uploaded_file, 
                                sep='\t', 
                                encoding=encoding
                            )
                            st.success(f"Successfully read CSV with {encoding} encoding and tab delimiter")
                            break
                        except Exception:
                            try:
                                uploaded_file.seek(0)
                                df = pd.read_csv(
                                    uploaded_file, 
                                    sep=';', 
                                    encoding=encoding
                                )
                                st.success(f"Successfully read CSV with {encoding} encoding and semicolon delimiter")
                                break
                            except Exception:
                                # Last resort: try to manually read the file
                                if encoding == encodings_to_try[-1]:
                                    try:
                                        uploaded_file.seek(0)
                                        # Try to read as binary and decode manually
                                        raw_content = uploaded_file.read()
                                        # Replace or ignore problematic bytes
                                        text_content = raw_content.decode(encoding, errors='replace')
                                        
                                        # Use StringIO to create a file-like object
                                        import io
                                        string_data = io.StringIO(text_content)
                                        
                                        # Try reading with all possible delimiters
                                        for delimiter in [',', '\t', ';']:
                                            try:
                                                string_data.seek(0)
                                                df = pd.read_csv(string_data, sep=delimiter)
                                                st.success(f"Successfully read CSV with manual decoding and {delimiter} delimiter")
                                                break
                                            except Exception:
                                                continue
                                        else:
                                            st.error(f"Failed to read CSV file with all available delimiters after manual decoding")
                                            return None
                                    except Exception as e:
                                        st.error(f"Failed to read CSV with all encodings: {e}")
                                        return None
            else:
                # If we've tried all encodings and still failed
                st.error("Failed to read CSV file with all available encodings")
                return None
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload an XLSX or CSV file.")
            return None
        
        # Check for comment column and rename if necessary
        if "Comment" not in df.columns:
            # Common text column names to look for
            text_column_keywords = ['text', 'comment', 'message', 'content', 'post']
            
            # Try to find a suitable column based on name
            potential_columns = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in text_column_keywords):
                    potential_columns.append(col)
            
            if potential_columns:
                # Use the first matching column
                df = df.rename(columns={potential_columns[0]: 'Comment'})
                st.info(f"Renamed column '{potential_columns[0]}' to 'Comment'.")
            else:
                # If no suitable column found by name, look for string columns with content
                for col in df.columns:
                    if df[col].dtype == 'object':  # object type usually means strings
                        # Check if column has meaningful content
                        sample = df[col].dropna().astype(str).str.len().mean()
                        if sample > 5:  # If average text length is reasonable
                            df = df.rename(columns={col: 'Comment'})
                            st.info(f"No explicit comment column found. Using '{col}' as the Comment column.")
                            break
                
                # If we still don't have a Comment column
                if "Comment" not in df.columns:
                    st.error("Could not identify a suitable text column to use as 'Comment'.")
                    return None
        
        # Ensure Comment column has string values
        df['Comment'] = df['Comment'].astype(str)
        
        # Drop rows with empty comments
        df = df[df['Comment'].str.strip() != '']
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Text Preprocessing Enhancements
def preprocess_text(text):
    """Cleans and processes text for better sentiment analysis."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    
    # Extract and save emojis before removing them
    emojis_found = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    
    # Convert emojis to text for sentiment analysis
    text_with_emoji_names = emoji.demojize(text, delimiters=(" ", " "))
    
    # Clean text for general analysis
    clean_version = clean_text(text_with_emoji_names)
    
    return {
        'cleaned_text': clean_version,
        'emojis': emojis_found,
        'demojized': text_with_emoji_names
    }

# Function to create a wordcloud
def create_wordcloud(text_series):
    """Create a WordCloud from a series of texts."""
    all_text = ' '.join(text_series.fillna(''))
    
    # Generate wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        contour_width=1
    ).generate(all_text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Function to analyze sentiment distribution
def plot_sentiment_distribution(df):
    """Create separate visualizations for sentiment and troll detection"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot sentiment distribution (using Enhanced Score)
    sentiment_types = df['Enhanced Score'].apply(
        lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    ).value_counts()
    
    sentiment_types.plot(kind='bar', ax=ax1, color=['green', 'gray', 'red'])
    ax1.set_title('Sentiment Distribution')
    ax1.set_ylabel('Count')
    
    # Plot troll distribution
    troll_counts = df['Is_Troll'].value_counts()
    colors = ['red' if x else 'green' for x in troll_counts.index]
    troll_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Troll Detection Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

# Function to create a sentiment heatmap
def create_sentiment_heatmap(df):
    """Create separate heatmaps for sentiment and troll detection."""
    # Sentiment comparison heatmap
    sentiment_methods = ['VADER Sentiment', 'MNB Sentiment', 'Combined Sentiment']
    sentiment_matrix = pd.DataFrame()
    
    # Get clean sentiment for Enhanced
    df['Clean_Sentiment'] = df['Enhanced Score'].apply(
        lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
    )
    
    for col in sentiment_methods:
        sentiment_matrix[col] = df[col].apply(lambda x: x.split(' ')[0])
    sentiment_matrix['Enhanced'] = df['Clean_Sentiment']  # Use Clean_Sentiment instead of Enhanced Sentiment
    
    # Calculate agreement matrix for sentiments only
    agreement_matrix = pd.DataFrame(index=['Positive', 'Neutral', 'Negative'], 
                                  columns=['Positive', 'Neutral', 'Negative'])
    agreement_matrix = agreement_matrix.fillna(0)
    
    # Count agreements using Clean_Sentiment
    for idx, row in sentiment_matrix.iterrows():
        enhanced = row['Enhanced']  # This is now from Clean_Sentiment
        for method in sentiment_methods:
            agreement_matrix.at[enhanced, row[method]] += 1
    
    # Create the heatmap
    fig = px.imshow(agreement_matrix, 
                    labels=dict(x="Other Methods", y="Enhanced Sentiment", color="Agreement Count"),
                    color_continuous_scale='Viridis')
    
    fig.update_layout(title="Sentiment Analysis Method Agreement")
    
    return fig

# Function to plot sentiment breakdown
def plot_sentiment_factors(comment, breakdown=None):
    """Create a plot showing the factors contributing to sentiment score."""
    if breakdown is None:
        breakdown = get_sentiment_breakdown_with_language(comment)
    
    # Create data for the plot
    if 'tagalog' in breakdown and breakdown['tagalog'] != 0:
        # This is a Tagalog or mixed language comment
        factors = ['VADER', 'ML Model', 'Emoji', 'Tagalog']
        values = [breakdown['vader'], breakdown['multilingual'], breakdown['emoji'], breakdown['tagalog']]
    else:
        # This is an English comment
        factors = ['VADER', 'ML Model', 'Emoji', 'Lexicon']
        values = [breakdown['vader'], breakdown['multilingual'] if 'multilingual' in breakdown else breakdown['ml'], 
                 breakdown['emoji'], breakdown['lexicon']]
    
    # Normalize values to -1 to 1 range
    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=factors,
        y=values,
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f"Sentiment Components for: '{comment[:50]}...'",
        xaxis_title="Analysis Component",
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis=dict(range=[-1, 1])
    )
    
    # Add a line for the final score
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=breakdown['final'],
        x1=3.5,
        y1=breakdown['final'],
        line=dict(
            color="blue",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=3.5,
        y=breakdown['final'],
        text=f"Final Score: {breakdown['final']:.2f}",
        showarrow=False,
        yshift=10
    )
    
    return fig

def plot_clean_sentiment_factors(comment, breakdown):
    """Create a plot showing ONLY sentiment factors (no troll data)."""
    # Create data for the plot
    factors = ['VADER', 'ML Model', 'Emoji', 'Lexicon']
    values = [
        breakdown['vader'],
        breakdown['multilingual'] if 'multilingual' in breakdown else breakdown['ml'],
        breakdown['emoji'],
        breakdown['lexicon']
    ]
    
    # Normalize values to -1 to 1 range
    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
    
    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=factors,
        y=values,
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f"Sentiment Components for: '{comment[:50]}...'",
        xaxis_title="Analysis Component",
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis=dict(range=[-1, 1])
    )
    
    # Add a line for the final sentiment score
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=breakdown['final'],
        x1=3.5,
        y1=breakdown['final'],
        line=dict(
            color="blue",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=3.5,
        y=breakdown['final'],
        text=f"Final Score: {breakdown['final']:.2f}",
        showarrow=False,
        yshift=10
    )
    
    return fig

def get_risk_color(risk_level):
    """Return color code for risk level."""
    return {
        'Low': '#4CAF50',
        'Medium': '#FFC107',
        'High': '#FF9800',
        'Critical': '#F44336'
    }.get(risk_level, '#808080')  # Default gray

# About page Tagalog information
TAGALOG_ABOUT_TEXT = """
### Filipino/Tagalog Language Support

This application now supports sentiment analysis for:
- English language comments
- Filipino/Tagalog language comments
- Code-switching (mix of English and Tagalog)
- Regional Filipino dialects (Bisaya, Ilokano, Bikolano, etc.)
- Social media slang and TikTok-specific expressions in Tagalog

The application now features troll detection capabilities:
- Identifies comments with troll-like behavior patterns
- Analyzes language patterns specific to trolling
- Detects excessive punctuation, ALL CAPS, and inflammatory language
- Provides a troll score for each comment
- Specialized detection for Filipino/Taglish troll comments

You can set your language preference in the sidebar under "Language Settings".
"""

# Create sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a section", ["Upload Data", "Fetch TikTok Comments", "Sentiment Explorer", "About"])

# Add language settings
language_mode = add_language_settings()

# About page
if page == "About":
    st.header("About TikTok Sentiment Analysis")
    st.markdown("""
    This application allows you to analyze the sentiment of TikTok comments to understand audience reactions and market trends.
    
    ### Features:
    - Upload Excel files containing TikTok comments
    - Fetch comments directly from TikTok videos using Apify
    - Analyze sentiment using multiple techniques:
      - VADER sentiment analysis (rule-based)
      - Machine learning-based classification (Multinomial Naive Bayes)
      - Enhanced analysis with TikTok-specific lexicon
      - Emoji sentiment analysis
      - Ensemble method combining all approaches
    - Visualize sentiment distribution
    - Generate word clouds from comments
    - Extract hashtags and analyze trends
    - Interactive sentiment comparison
    
    ### How to Use:
    1. Navigate to "Upload Data" to analyze your own data
    2. Or go to "Fetch TikTok Comments" to analyze comments from a TikTok video URL
    3. Use "Sentiment Explorer" to understand how sentiment analysis works
    4. Review the analysis and visualizations
    
    ### Technologies Used:
    - NLTK for natural language processing
    - scikit-learn for machine learning
    - VADER for rule-based sentiment analysis
    - Apify for data collection
    - Streamlit for the web interface
    - Plotly and Matplotlib for visualizations
    """)
    
    # Add Tagalog language support information
    st.markdown(TAGALOG_ABOUT_TEXT)

# Upload section
elif page == "Upload Data":
    st.header("Upload Your Data File")
    
    # Update file uploader to accept both xlsx and csv
    uploaded_file = st.file_uploader("Upload a file containing TikTok comments", type=["xlsx", "xls", "csv"])
    
    if uploaded_file:
        # Display a spinner while processing
        with st.spinner("Reading and processing file..."):
            # Process the uploaded file
            comments_df = read_file_with_multiple_formats(uploaded_file)
            
            if comments_df is not None:
                st.success(f"File uploaded and processed successfully. Found {len(comments_df)} comments.")
                
                # Continue with your existing processing pipeline
                with st.spinner("Processing comments..."):
                    # Process comments with existing functions
                    processed_data = comments_df['Comment'].apply(preprocess_text)
                    
                    # Add processed text columns
                    comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                    comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                    comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                    
                    # Extract hashtags
                    comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                    
                    # Apply sentiment analysis and troll detection SEPARATELY
                    troll_results = comments_df['Comment'].apply(
                        lambda text: analyze_comment_with_trolling(text, language_mode)
                    )

                    # Store results in COMPLETELY separate columns
                    comments_df['Enhanced Score'] = troll_results.apply(lambda x: x['sentiment_score'])
                    comments_df['Is_Troll'] = troll_results.apply(lambda x: x['is_troll'])
                    comments_df['Troll Score'] = troll_results.apply(lambda x: x['troll_score'])
                
                # Create display DataFrame with separate columns
                display_df = pd.DataFrame()
                display_df['Comment'] = comments_df['Comment']
                
                # Clean sentiment display - NO TROLL INFO
                display_df['Sentiment'] = comments_df['Enhanced Score'].apply(
                    lambda score: f"{'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'} ({score:.2f})"
                )
                
                # Completely separate troll column
                display_df['Troll Status'] = comments_df.apply(
                    lambda row: f"🚨 ({row['Troll Score']:.2f})" if row['Is_Troll'] else "",
                    axis=1
                )
                
                # Add this right after getting troll_results
                st.write("Sample troll_result:", troll_results.iloc[0] if len(troll_results) > 0 else "No results")

                # Add this right after setting Enhanced Sentiment
                st.write("Sample Enhanced Sentiment:", comments_df['Enhanced Score'].iloc[0] if len(comments_df) > 0 else "No data")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Data View", 
                    "Visualizations", 
                    "Sentiment Analysis",
                    "Troll Detection",  # New dedicated tab
                    "Statistics", 
                    "Market Trends"
                ])
                
                with tab1:
                    # Display data
                    st.subheader("Processed Comments")
                    # Show comment, clean sentiment, and separate troll status
                    st.dataframe(display_df[['Comment', 'Sentiment', 'Troll Status']])
                    
                    # For detailed analysis, show:
                    st.dataframe(comments_df[[
                        'Comment', 
                        'Enhanced Score',  # Raw sentiment score
                        'Is_Troll',       # Boolean troll flag
                        'Troll Score'     # Numerical troll score
                    ]])
                    
                    # Sentiment & Troll Detection Correction
                    st.subheader("Sentiment & Troll Detection Correction")
                    st.write("Select comments to correct their sentiment or troll status:")

                    # Let user select a comment
                    selected_comment_idx = st.selectbox(
                        "Select comment to relabel:", 
                        options=comments_df.index.tolist(),
                        format_func=lambda x: comments_df.loc[x, 'Comment'][:50] + "..."
                    )

                    # Show current status - calculate clean sentiment
                    current_score = comments_df.loc[selected_comment_idx, 'Enhanced Score']
                    current_sentiment = get_sentiment_type(current_score)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Current Sentiment Status:**")
                        st.write(f"Sentiment: {current_sentiment} ({current_score:.2f})")
                        
                        # Let user choose new sentiment
                        corrected_sentiment = st.radio(
                            "Correct sentiment:", 
                            options=["Positive", "Neutral", "Negative"]
                        )

                    with col2:
                        st.write("**Current Troll Status:**")
                        st.write(f"{'TROLL' if comments_df.loc[selected_comment_idx, 'Is_Troll'] else 'Not a Troll'} (Score: {comments_df.loc[selected_comment_idx, 'Troll Score']:.2f})")
                        
                        # Separate troll detection correction
                        is_troll_corrected = st.radio(
                            "Is this a troll comment?",
                            options=["No", "Yes"]
                        )

                    if st.button("Save Correction"):
                        # Update sentiment correction without mixing troll status
                        updated_df = update_sentiment_correction(comments_df, selected_comment_idx, corrected_sentiment)
                        st.success(f"Comment updated - Sentiment: {corrected_sentiment}, Troll Status: {'🚨' if updated_df['Is_Troll'].iloc[0] else 'Not a Troll'}")
                    
                    # Add language detection information
                    st.subheader("Language Information")
                    language_counts = comments_df['Comment'].apply(is_tagalog)
                    tagalog_count = language_counts.sum()
                    english_count = len(language_counts) - tagalog_count

                    col1, col2 = st.columns(2)
                    col1.metric("Detected Tagalog Comments", tagalog_count)
                    col2.metric("Detected English Comments", english_count)
                
                with tab2:
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Distribution")
                        # Plot sentiment distribution
                        fig = plot_sentiment_distribution(comments_df)
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Word Cloud")
                        fig = create_wordcloud(comments_df['Processed Comment'])
                        st.pyplot(fig)
                    
                    # Emoji analysis
                    st.subheader("Emoji Analysis")
                    all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                    if all_emojis:
                        emoji_counter = Counter(all_emojis)
                        top_emojis = emoji_counter.most_common(10)
                        
                        emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Count'])
                        
                        # Create horizontal bar chart for emojis
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(y=emoji_df['Emoji'], x=emoji_df['Count'], ax=ax, orient='h')
                        ax.set_title('Top 10 Emojis')
                        st.pyplot(fig)
                    else:
                        st.info("No emojis found in the comments.")
                
                with tab3:
                    # Sentiment Analysis Comparison - CLEAN VERSION
                    st.subheader("Sentiment Analysis Comparison")
                    
                    # Create a heatmap comparing ONLY sentiment methods
                    sentiment_heatmap = create_sentiment_heatmap(comments_df)
                    if sentiment_heatmap:
                        st.plotly_chart(sentiment_heatmap)
                    
                    # Select a comment to analyze in detail
                    st.subheader("Analyze Individual Comment")
                    selected_comment = st.selectbox(
                        "Select a comment to analyze:", 
                        comments_df['Comment'].tolist()
                    )
                    
                    if selected_comment:
                        # Display ONLY sentiment breakdown
                        breakdown = get_sentiment_breakdown_with_language(selected_comment, language_mode)
                        breakdown_fig = plot_clean_sentiment_factors(selected_comment, breakdown)
                        st.plotly_chart(breakdown_fig)
                        
                        # Show ONLY sentiment results
                        comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                        st.write("**Sentiment Analysis Results:**")
                        col1, col2 = st.columns(2)
                        col1.metric("VADER", comments_df.loc[comment_idx, 'VADER Sentiment'])
                        col1.metric("MNB", comments_df.loc[comment_idx, 'MNB Sentiment'])
                        col2.metric("Combined", comments_df.loc[comment_idx, 'Combined Sentiment'])
                        col2.metric("Enhanced", f"{comments_df.loc[comment_idx, 'Enhanced Score']:.2f}")
                        
                        # Add language detection info
                        is_tag = is_tagalog(selected_comment)
                        st.info(f"Detected language: {'Tagalog' if is_tag else 'English'}")
                
                with tab4:
                    # Troll Detection Analysis
                    st.subheader("Troll Detection Dashboard")
                    
                    # Overall Troll Metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        troll_count = len(comments_df[comments_df['Is_Troll']])
                        total_count = len(comments_df)
                        troll_percentage = (troll_count / total_count) * 100
                        
                        st.metric("Troll Comments", f"{troll_count} / {total_count}")
                        st.metric("Troll Percentage", f"{troll_percentage:.1f}%")
                        st.metric("Risk Level", get_troll_risk_level(troll_percentage))
                    
                    with col2:
                        # Troll Risk Distribution
                        risk_counts = comments_df['Troll Score'].apply(
                            lambda x: pd.cut([x], 
                                bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
                                labels=['Low', 'Medium', 'High', 'Critical'])[0]
                        ).value_counts()
                        
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title="Troll Risk Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Low': '#4CAF50',
                                'Medium': '#FFC107',
                                'High': '#FF9800',
                                'Critical': '#F44336'
                            }
                        )
                        st.plotly_chart(fig)
                    
                    # Individual Comment Troll Analysis
                    st.subheader("Analyze Comment for Troll Behavior")
                    selected_comment = st.selectbox(
                        "Select comment to analyze:", 
                        comments_df['Comment'].tolist(),
                        key="troll_analysis_select"  # Unique key to avoid conflicts
                    )
                    
                    if selected_comment:
                        comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                        troll_score = comments_df.loc[comment_idx, 'Troll Score']
                        is_troll = comments_df.loc[comment_idx, 'Is_Troll']
                        
                        # Display troll analysis results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Troll Status", "TROLL" if is_troll else "Not a Troll")
                            st.metric("Troll Score", f"{troll_score:.2f}")
                        
                        with col2:
                            risk_level = get_troll_risk_level(troll_score * 100)  # Convert score to percentage
                            st.markdown(
                                f"""
                                <div style="
                                    padding: 10px; 
                                    border-radius: 5px; 
                                    background-color: {get_risk_color(risk_level)}; 
                                    color: white; 
                                    text-align: center;
                                    margin: 10px 0;
                                ">
                                    Risk Level: {risk_level}
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                
                with tab5:
                    # Statistics
                    st.subheader("Comment Statistics")
                    
                    # Basic stats
                    stats = {
                        "Total Comments": len(comments_df),
                        "Average Length": int(comments_df['Comment'].apply(len).mean()),
                        "Emoji Usage": len(comments_df[comments_df['Emojis'] != '']),
                        
                        # Sentiment stats - calculated from score
                        "Positive": len(comments_df[comments_df['Enhanced Score'] > 0.05]),
                        "Negative": len(comments_df[comments_df['Enhanced Score'] < -0.05]),
                        "Neutral": len(comments_df[comments_df['Enhanced Score'] == 0]),
                        
                        # Troll stats
                        "Trolls": len(comments_df[comments_df['Is_Troll'] == True]),
                        "High Risk": len(comments_df[comments_df['Troll Score'] > 0.8]),
                        "Medium Risk": len(comments_df[(comments_df['Troll Score'] > 0.6) & (comments_df['Troll Score'] <= 0.8)]),
                        
                        # Language stats
                        "Tagalog": len(comments_df[comments_df['Comment'].apply(is_tagalog)]),
                        "English": len(comments_df[~comments_df['Comment'].apply(is_tagalog)])
                    }
                    
                    # Display in organized sections
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("General Stats")
                        st.metric("Total Comments", stats["Total Comments"])
                        st.metric("Average Length", stats["Average Length"])
                        st.metric("With Emojis", stats["Emoji Usage"])
                    
                    with col2:
                        st.subheader("Sentiment Distribution")
                        st.metric("Positive", stats["Positive"])
                        st.metric("Negative", stats["Negative"])
                        st.metric("Neutral", stats["Neutral"])
                    
                    with col3:
                        st.subheader("Troll Detection")
                        st.metric("Total Trolls", stats["Trolls"])
                        st.metric("High Risk", stats["High Risk"])
                        st.metric("Medium Risk", stats["Medium Risk"])
                    
                    # Hashtag analysis
                    st.subheader("Hashtag Analysis")
                    all_hashtags = [tag for tags in comments_df['Hashtags'] for tag in tags]
                    if all_hashtags:
                        hashtag_counter = Counter(all_hashtags)
                        top_hashtags = hashtag_counter.most_common(15)
                        
                        hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
                        
                        # Create horizontal bar chart for hashtags
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(y=hashtag_df['Hashtag'], x=hashtag_df['Count'], ax=ax, orient='h')
                        ax.set_title('Top 15 Hashtags')
                        st.pyplot(fig)
                    else:
                        st.info("No hashtags found in the comments.")
                with tab6:
                    # Market Trends Analysis with separated concerns
                    st.subheader("Market Analysis Dashboard")
                    
                    # Create two columns for main metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Metrics")
                        # Calculate metrics excluding trolls
                        valid_comments = comments_df[~comments_df['Is_Troll']]
                        valid_comments['Clean_Sentiment'] = valid_comments['Enhanced Score'].apply(
                            lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
                        )
                        market_score = calculate_market_trend_score(comments_df)
                        
                        st.metric("Market Sentiment Score", f"{market_score:.2f}")
                        st.metric("Valid Comments", f"{len(valid_comments)} / {len(comments_df)}")
                        
                        # Show sentiment distribution
                        sentiment_counts = valid_comments['Clean_Sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution (Excluding Trolls)",
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        st.subheader("Troll Analysis")
                        troll_count = len(comments_df[comments_df['Is_Troll']])
                        troll_percentage = (troll_count / len(comments_df)) * 100
                        
                        st.metric("Troll Percentage", f"{troll_percentage:.1f}%")
                        st.metric("Risk Level", get_troll_risk_level(troll_percentage))
                        
                        # Show troll risk distribution
                        risk_counts = comments_df['Troll Score'].apply(
                            lambda x: pd.cut([x], bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
                                           labels=['Low', 'Medium', 'High', 'Critical'])[0]
                        ).value_counts()
                        
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title="Troll Risk Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Low': '#4CAF50',
                                'Medium': '#FFC107',
                                'High': '#FF9800',
                                'Critical': '#F44336'
                            }
                        )
                        st.plotly_chart(fig)
                    
                    # Market prediction visualization
                    st.subheader("Market Prediction")
                    market_fig = plot_market_prediction(comments_df)
                    st.plotly_chart(market_fig)
                    
                    # Purchase volume prediction
                    purchase_volume = predict_purchase_volume(comments_df)
                    st.metric("Predicted Purchase Volume", f"{purchase_volume:,}")
                    
                    # Detailed market report
                    st.subheader("Market Analysis Report")
                    report = generate_market_trend_report(comments_df)
                    st.markdown(report)
                    
                    # Add download button for report
                    report_csv = pd.DataFrame({
                        'Metric': ['Market Score', 'Valid Comments', 'Troll Percentage', 'Purchase Volume'],
                        'Value': [
                            market_score,
                            len(valid_comments),
                            troll_percentage,
                            purchase_volume
                        ]
                    }).to_csv(index=False)
                    
                    st.download_button(
                        label="Download Market Analysis Report",
                        data=report_csv,
                        file_name="market_analysis_report.csv",
                        mime="text/csv"
                    )

# TikTok Comment Fetching
elif page == "Fetch TikTok Comments":
    st.header("Fetch TikTok Comments")
    
    # Input for TikTok video link
    video_link = st.text_input("Enter TikTok video link:")
    col1, col2 = st.columns(2)
    max_comments = col1.number_input("Maximum comments to fetch:", min_value=10, max_value=2000, value=500)
    analyze_button = col2.button("Fetch and Analyze")
    
    if analyze_button:
        if video_link:
            with st.spinner("Fetching comments, please wait..."):
                comments_df = fetch_tiktok_comments(video_link, max_comments=max_comments)
                
                if comments_df is not None and not comments_df.empty:
                    st.success(f"Fetched {len(comments_df)} comments!")
                    
                    # Process comments
                    with st.spinner("Processing comments..."):
                        # Process comments with existing functions
                        processed_data = comments_df['Comment'].apply(preprocess_text)
                        
                        # Add processed text columns
                        comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                        comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                        comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                        
                        # Extract hashtags
                        comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                        
                        # Apply sentiment analysis and troll detection SEPARATELY
                        troll_results = comments_df['Comment'].apply(
                            lambda text: analyze_comment_with_trolling(text, language_mode)
                        )

                        # Store results in COMPLETELY separate columns
                        comments_df['Enhanced Score'] = troll_results.apply(lambda x: x['sentiment_score'])
                        comments_df['Is_Troll'] = troll_results.apply(lambda x: x['is_troll'])
                        comments_df['Troll Score'] = troll_results.apply(lambda x: x['troll_score'])
                    
                    # Create display DataFrame with separate columns
                    display_df = pd.DataFrame()
                    display_df['Comment'] = comments_df['Comment']
                    
                    # Clean sentiment display - NO TROLL INFO
                    display_df['Sentiment'] = comments_df['Enhanced Score'].apply(
                        lambda score: f"{'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'} ({score:.2f})"
                    )
                    
                    # Completely separate troll column
                    display_df['Troll Status'] = comments_df.apply(
                        lambda row: f"🚨 ({row['Troll Score']:.2f})" if row['Is_Troll'] else "",
                        axis=1
                    )
                    
                    # Add this right after getting troll_results
                    st.write("Sample troll_result:", troll_results.iloc[0] if len(troll_results) > 0 else "No results")

                    # Add this right after setting Enhanced Sentiment
                    st.write("Sample Enhanced Sentiment:", comments_df['Enhanced Score'].iloc[0] if len(comments_df) > 0 else "No data")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "Data View", 
                        "Visualizations", 
                        "Sentiment Analysis",
                        "Troll Detection",  # New dedicated tab
                        "Statistics", 
                        "Market Trends"
                    ])
                    
                    with tab1:
                        # Display data
                        st.subheader("Processed Comments")
                        # Show comment, clean sentiment, and separate troll status
                        st.dataframe(display_df[['Comment', 'Sentiment', 'Troll Status']])
                        
                        # For detailed analysis, show:
                        st.dataframe(comments_df[[
                            'Comment', 
                            'Enhanced Score',  # Raw sentiment score
                            'Is_Troll',       # Boolean troll flag
                            'Troll Score'     # Numerical troll score
                        ]])
                        
                        # Sentiment & Troll Detection Correction
                        st.subheader("Sentiment & Troll Detection Correction")
                        st.write("Select comments to correct their sentiment or troll status:")

                        # Let user select a comment
                        selected_comment_idx = st.selectbox(
                            "Select comment to relabel:", 
                            options=comments_df.index.tolist(),
                            format_func=lambda x: comments_df.loc[x, 'Comment'][:50] + "..."
                        )

                        # Show current status - calculate clean sentiment
                        current_score = comments_df.loc[selected_comment_idx, 'Enhanced Score']
                        current_sentiment = get_sentiment_type(current_score)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Current Sentiment Status:**")
                            st.write(f"Sentiment: {current_sentiment} ({current_score:.2f})")
                            
                            # Let user choose new sentiment
                            corrected_sentiment = st.radio(
                                "Correct sentiment:", 
                                options=["Positive", "Neutral", "Negative"]
                            )

                        with col2:
                            st.write("**Current Troll Status:**")
                            st.write(f"{'TROLL' if comments_df.loc[selected_comment_idx, 'Is_Troll'] else 'Not a Troll'} (Score: {comments_df.loc[selected_comment_idx, 'Troll Score']:.2f})")
                            
                            # Separate troll detection correction
                            is_troll_corrected = st.radio(
                                "Is this a troll comment?",
                                options=["No", "Yes"]
                            )

                        if st.button("Save Correction"):
                            # Update sentiment correction without mixing troll status
                            updated_df = update_sentiment_correction(comments_df, selected_comment_idx, corrected_sentiment)
                            st.success(f"Comment updated - Sentiment: {corrected_sentiment}, Troll Status: {'🚨' if updated_df['Is_Troll'].iloc[0] else 'Not a Troll'}")
                        
                        # Add language detection information
                        st.subheader("Language Information")
                        language_counts = comments_df['Comment'].apply(is_tagalog)
                        tagalog_count = language_counts.sum()
                        english_count = len(language_counts) - tagalog_count

                        col1, col2 = st.columns(2)
                        col1.metric("Detected Tagalog Comments", tagalog_count)
                        col2.metric("Detected English Comments", english_count)
                    
                    with tab2:
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Distribution")
                            # Plot sentiment distribution
                            fig = plot_sentiment_distribution(comments_df)
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("Word Cloud")
                            fig = create_wordcloud(comments_df['Processed Comment'])
                            st.pyplot(fig)
                        
                        # Emoji analysis
                        st.subheader("Emoji Analysis")
                        all_emojis = ''.join(comments_df['Emojis'].fillna(''))
                        if all_emojis:
                            emoji_counter = Counter(all_emojis)
                            top_emojis = emoji_counter.most_common(10)
                            
                            emoji_df = pd.DataFrame(top_emojis, columns=['Emoji', 'Count'])
                            # Create horizontal bar chart for emojis
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(y=emoji_df['Emoji'], x=emoji_df['Count'], ax=ax, orient='h')
                            ax.set_title('Top 10 Emojis')
                            st.pyplot(fig)
                        else:
                            st.info("No emojis found in the comments.")
                        
                        # Calculate metrics excluding trolls
                        valid_comments = comments_df[~comments_df['Is_Troll']]
                        # Use Enhanced Score to determine sentiment, not Enhanced Sentiment
                        valid_comments['Clean_Sentiment'] = valid_comments['Enhanced Score'].apply(
                            lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
                        )
                    
                    with tab3:
                        # Sentiment Analysis Comparison - CLEAN VERSION
                        st.subheader("Sentiment Analysis Comparison")
                        
                        # Create a heatmap comparing ONLY sentiment methods
                        sentiment_heatmap = create_sentiment_heatmap(comments_df)
                        if sentiment_heatmap:
                            st.plotly_chart(sentiment_heatmap)
                        
                        # Select a comment to analyze in detail
                        st.subheader("Analyze Individual Comment")
                        selected_comment = st.selectbox(
                            "Select a comment to analyze:", 
                            comments_df['Comment'].tolist()
                        )
                        
                        if selected_comment:
                            # Display ONLY sentiment breakdown
                            breakdown = get_sentiment_breakdown_with_language(selected_comment, language_mode)
                            breakdown_fig = plot_clean_sentiment_factors(selected_comment, breakdown)
                            st.plotly_chart(breakdown_fig)
                            
                            # Show ONLY sentiment results
                            comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                            st.write("**Sentiment Analysis Results:**")
                            col1, col2 = st.columns(2)
                            col1.metric("VADER", comments_df.loc[comment_idx, 'VADER Sentiment'])
                            col1.metric("MNB", comments_df.loc[comment_idx, 'MNB Sentiment'])
                            col2.metric("Combined", comments_df.loc[comment_idx, 'Combined Sentiment'])
                            col2.metric("Enhanced", f"{comments_df.loc[comment_idx, 'Enhanced Score']:.2f}")
                            
                            # Add language detection info
                            is_tag = is_tagalog(selected_comment)
                            st.info(f"Detected language: {'Tagalog' if is_tag else 'English'}")
                    
                    with tab4:
                        # Troll Detection Analysis
                        st.subheader("Troll Detection Dashboard")
                        
                        # Overall Troll Metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            troll_count = len(comments_df[comments_df['Is_Troll']])
                            total_count = len(comments_df)
                            troll_percentage = (troll_count / total_count) * 100
                            
                            st.metric("Troll Comments", f"{troll_count} / {total_count}")
                            st.metric("Troll Percentage", f"{troll_percentage:.1f}%")
                            st.metric("Risk Level", get_troll_risk_level(troll_percentage))
                        
                        with col2:
                            # Troll Risk Distribution
                            risk_counts = comments_df['Troll Score'].apply(
                                lambda x: pd.cut([x], 
                                    bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
                                    labels=['Low', 'Medium', 'High', 'Critical'])[0]
                            ).value_counts()
                            
                            fig = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title="Troll Risk Distribution",
                                color=risk_counts.index,
                                color_discrete_map={
                                    'Low': '#4CAF50',
                                    'Medium': '#FFC107',
                                    'High': '#FF9800',
                                    'Critical': '#F44336'
                                }
                            )
                            st.plotly_chart(fig)
                        
                        # Individual Comment Troll Analysis
                        st.subheader("Analyze Comment for Troll Behavior")
                        selected_comment = st.selectbox(
                            "Select comment to analyze:", 
                            comments_df['Comment'].tolist(),
                            key="troll_analysis_select"  # Unique key to avoid conflicts
                        )
                        
                        if selected_comment:
                            comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                            troll_score = comments_df.loc[comment_idx, 'Troll Score']
                            is_troll = comments_df.loc[comment_idx, 'Is_Troll']
                            
                            # Display troll analysis results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Troll Status", "TROLL" if is_troll else "Not a Troll")
                                st.metric("Troll Score", f"{troll_score:.2f}")
                            
                            with col2:
                                risk_level = get_troll_risk_level(troll_score * 100)  # Convert score to percentage
                                st.markdown(
                                    f"""
                                    <div style="
                                        padding: 10px; 
                                        border-radius: 5px; 
                                        background-color: {get_risk_color(risk_level)}; 
                                        color: white; 
                                        text-align: center;
                                        margin: 10px 0;
                                    ">
                                        Risk Level: {risk_level}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    
                    with tab5:
                        # Statistics
                        st.subheader("Comment Statistics")
                        
                        # Basic stats
                        stats = {
                            "Total Comments": len(comments_df),
                            "Average Length": int(comments_df['Comment'].apply(len).mean()),
                            "Emoji Usage": len(comments_df[comments_df['Emojis'] != '']),
                            
                            # Sentiment stats - calculated from score
                            "Positive": len(comments_df[comments_df['Enhanced Score'] > 0.05]),
                            "Negative": len(comments_df[comments_df['Enhanced Score'] < -0.05]),
                            "Neutral": len(comments_df[comments_df['Enhanced Score'] == 0]),
                            
                            # Troll stats
                            "Trolls": len(comments_df[comments_df['Is_Troll'] == True]),
                            "High Risk": len(comments_df[comments_df['Troll Score'] > 0.8]),
                            "Medium Risk": len(comments_df[(comments_df['Troll Score'] > 0.6) & (comments_df['Troll Score'] <= 0.8)]),
                            
                            # Language stats
                            "Tagalog": len(comments_df[comments_df['Comment'].apply(is_tagalog)]),
                            "English": len(comments_df[~comments_df['Comment'].apply(is_tagalog)])
                        }
                        
                        # Display in organized sections
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("General Stats")
                            st.metric("Total Comments", stats["Total Comments"])
                            st.metric("Average Length", stats["Average Length"])
                            st.metric("With Emojis", stats["Emoji Usage"])
                        
                        with col2:
                            st.subheader("Sentiment Distribution")
                            st.metric("Positive", stats["Positive"])
                            st.metric("Negative", stats["Negative"])
                            st.metric("Neutral", stats["Neutral"])
                        
                        with col3:
                            st.subheader("Troll Detection")
                            st.metric("Total Trolls", stats["Trolls"])
                            st.metric("High Risk", stats["High Risk"])
                            st.metric("Medium Risk", stats["Medium Risk"])
                        
                        # Hashtag analysis
                        st.subheader("Hashtag Analysis")
                        all_hashtags = [tag for tags in comments_df['Hashtags'] for tag in tags]
                        if all_hashtags:
                            hashtag_counter = Counter(all_hashtags)
                            top_hashtags = hashtag_counter.most_common(15)
                            
                            hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
                            
                            # Create horizontal bar chart for hashtags
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(y=hashtag_df['Hashtag'], x=hashtag_df['Count'], ax=ax, orient='h')
                            ax.set_title('Top 15 Hashtags')
                            st.pyplot(fig)
                        else:
                            st.info("No hashtags found in the comments.")
                    with tab6:
                        # Market Trends Analysis with separated concerns
                        st.subheader("Market Analysis Dashboard")
                        
                        # Create two columns for main metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Metrics")
                            # Calculate metrics excluding trolls
                            valid_comments = comments_df[~comments_df['Is_Troll']]
                            valid_comments['Clean_Sentiment'] = valid_comments['Enhanced Score'].apply(
                                lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
                            )
                            market_score = calculate_market_trend_score(comments_df)
                            
                            st.metric("Market Sentiment Score", f"{market_score:.2f}")
                            st.metric("Valid Comments", f"{len(valid_comments)} / {len(comments_df)}")
                            
                            # Show sentiment distribution
                            sentiment_counts = valid_comments['Clean_Sentiment'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Sentiment Distribution (Excluding Trolls)",
                                color=sentiment_counts.index,
                                color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            st.subheader("Troll Analysis")
                            troll_count = len(comments_df[comments_df['Is_Troll']])
                            troll_percentage = (troll_count / len(comments_df)) * 100
                            
                            st.metric("Troll Percentage", f"{troll_percentage:.1f}%")
                            st.metric("Risk Level", get_troll_risk_level(troll_percentage))
                            
                            # Show troll risk distribution
                            risk_counts = comments_df['Troll Score'].apply(
                                lambda x: pd.cut([x], bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
                                               labels=['Low', 'Medium', 'High', 'Critical'])[0]
                            ).value_counts()
                            
                            fig = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title="Troll Risk Distribution",
                                color=risk_counts.index,
                                color_discrete_map={
                                    'Low': '#4CAF50',
                                    'Medium': '#FFC107',
                                    'High': '#FF9800',
                                    'Critical': '#F44336'
                                }
                            )
                            st.plotly_chart(fig)
                            
                            # Market prediction visualization
                            st.subheader("Market Prediction")
                            market_fig = plot_market_prediction(comments_df)
                            st.plotly_chart(market_fig)
                            
                            # Purchase volume prediction
                            purchase_volume = predict_purchase_volume(comments_df)
                            st.metric("Predicted Purchase Volume", f"{purchase_volume:,}")
                            
                            # Detailed market report
                            st.subheader("Market Analysis Report")
                            report = generate_market_trend_report(comments_df)
                            st.markdown(report)
                            
                            # Add download button for report
                            report_csv = pd.DataFrame({
                                'Metric': ['Market Score', 'Valid Comments', 'Troll Percentage', 'Purchase Volume'],
                                'Value': [
                                    market_score,
                                    len(valid_comments),
                                    troll_percentage,
                                    purchase_volume
                                ]
                            }).to_csv(index=False)
                            
                            st.download_button(
                                label="Download Market Analysis Report",
                                data=report_csv,
                                file_name="market_analysis_report.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("Failed to fetch comments. Please check the video link and try again.")
        else:
            st.warning("Please enter a TikTok video link.")

# Sentiment Explorer page
elif page == "Sentiment Explorer":
    st.header("Sentiment Analysis Explorer")
    
    # Input for testing
    test_comment = st.text_area("Enter a comment to analyze:", "This video is amazing! The tutorial was so helpful 🔥👍")
    
    if test_comment:
        # Create three columns for different aspects of analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Text Processing")
            # Process the text
            processed = preprocess_text(test_comment)
            
            st.write("**Original Text:**")
            st.write(test_comment)
            
            st.write("**Cleaned Text:**")
            st.write(processed['cleaned_text'])
            
            st.write("**Emojis Found:**")
            st.write(processed['emojis'] or "None")
            
            # Language detection
            is_tag = is_tagalog(test_comment)
            st.write("**Language Detection:**")
            st.write(f"{'Tagalog' if is_tag else 'English'}")
        
        with col2:
            st.subheader("Sentiment Analysis")
            
            # Get complete analysis
            analysis_result = analyze_comment_with_trolling(test_comment, language_mode)
            
            # Display sentiment information separately from troll info
            st.write("**Enhanced Sentiment Results:**")
            sentiment_type = get_sentiment_type(analysis_result['sentiment_score'])
            st.metric("Sentiment Type", sentiment_type)
            st.metric("Confidence Score", f"{analysis_result['sentiment_score']:.2f}")
            
            # Show breakdown
            st.write("**Sentiment Breakdown:**")
            breakdown = get_sentiment_breakdown_with_language(test_comment, language_mode)
            
            # Create sentiment components visualization
            sentiment_fig = plot_sentiment_factors(test_comment, breakdown)
            st.plotly_chart(sentiment_fig, use_container_width=True)
        
        with col3:
            st.subheader("Troll Detection")
            
            # Display troll detection results
            st.write("**Troll Analysis Results:**")
            
            # Show troll status with colored badge
            is_troll = analysis_result['is_troll']
            troll_score = analysis_result['troll_score']
            
            # Create color-coded risk level
            risk_level = pd.cut([troll_score], 
                bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical'])[0]
            
            risk_colors = {
                'Low': '#4CAF50',      # Green
                'Medium': '#FFC107',    # Amber
                'High': '#FF9800',      # Orange
                'Critical': '#F44336'   # Red
            }
            
            # Display troll metrics
            st.metric("Troll Status", "TROLL" if is_troll else "Not a Troll")
            st.metric("Troll Score", f"{troll_score:.2f}")
            
            # Display risk level with color
            st.markdown(
                f"""
                <div style="
                    padding: 10px; 
                    border-radius: 5px; 
                    background-color: {risk_colors[risk_level]}; 
                    color: white; 
                    text-align: center;
                    margin: 10px 0;
                ">
                    Risk Level: {risk_level}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add troll detection factors
            st.write("**Detection Factors:**")
            factors = [
                "ALL CAPS usage",
                "Excessive punctuation",
                "Inflammatory language",
                "Spam patterns",
                "Account behavior"
            ]
            
            # Show factor contribution (mock data - replace with actual analysis)
            for factor in factors:
                factor_score = min(max(troll_score + random.uniform(-0.2, 0.2), 0), 1)
                st.progress(factor_score)
                st.caption(f"{factor}: {factor_score:.2f}")
        
        # Add comparison section
        st.subheader("Analysis Methods Comparison")
        
        # Create comparison table with separate columns
        comparison_data = {
            'Method': ['VADER', 'MNB', 'Combined', 'Enhanced'],
            'Sentiment Score': [
                analyze_sentiment_vader(test_comment),
                train_mnb_model([test_comment])[0],
                combined_sentiment_analysis(test_comment),
                f"{analysis_result['sentiment_score']:.2f}"
            ],
            'Troll Detection': [
                'N/A',
                'N/A',
                'N/A',
                f"{'🚨' if analysis_result['is_troll'] else ''} ({analysis_result['troll_score']:.2f})"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Add explanation section
        st.subheader("Analysis Explanation")
        st.write("""
        The analysis above shows:
        1. **Text Processing**: How the raw text is cleaned and prepared for analysis
        2. **Sentiment Analysis**: The emotional tone of the comment
        3. **Troll Detection**: Whether the comment shows characteristics of trolling
        
        The systems work independently to ensure accurate classification of both sentiment and troll behavior.
        """)

# Add this test code to your app.py
st.subheader("Debugging Output")
st.write("Tagalog sentiment test:")
test_tag = tagalog_enhanced_sentiment_analysis("Ang ganda naman ng app na ito!")
st.write(f"Raw output: {test_tag}")

st.write("English sentiment test:")
test_eng = enhanced_sentiment_analysis("This app is really great!")
st.write(f"Raw output: {test_eng}")

# Add this DIRECTLY after importing the tagalog functions
# This completely overrides the original function
def tagalog_enhanced_sentiment_analysis(text):
    """
    A simple sentiment analysis function for Tagalog text.
    Returns a dictionary with sentiment score and sentiment label.
    """
    # Get base sentiment score
    score = analyze_tagalog_sentiment_score(text)
    
    # Convert to sentiment label
    sentiment = get_sentiment_type(score)
    
    # Return formatted string
    return f"{sentiment} ({score:.2f})"

# Override the imported function with our clean version
from tagalog_sentiment import get_tagalog_sentiment_breakdown

# Run the app
if __name__ == "__main__":
    # This ensures the app runs properly when executed directly
    pass

# When processing comments
def process_comments(comments_df):
    """Process comments with completely separate sentiment and troll detection"""
    # Create fresh DataFrame for display
    display_df = pd.DataFrame()
    display_df['Comment'] = comments_df['Comment']
    
    # Process each comment
    results = []
    for comment in comments_df['Comment']:
        analysis = analyze_comment_with_trolling(comment)
        results.append(analysis)
    
    # Add clean sentiment column
    display_df['Sentiment'] = [
        f"Positive ({r['sentiment_score']:.2f})" if r['sentiment_score'] > 0.05
        else f"Negative ({r['sentiment_score']:.2f})" if r['sentiment_score'] < -0.05
        else f"Neutral ({r['sentiment_score']:.2f})"
        for r in results
    ]
    
    # Add separate troll column
    display_df['Troll Alert'] = ['🚨' if r['is_troll'] else '' for r in results]
    
    # Store raw scores for calculations
    display_df['_sentiment_score'] = [r['sentiment_score'] for r in results]
    display_df['_troll_score'] = [r['troll_score'] for r in results]
    
    return display_df

def update_sentiment_correction(comments_df, selected_comment_idx, corrected_sentiment):
    """Update sentiment correction without mixing troll status"""
    # Map sentiment labels to scores
    sentiment_scores = {
        'Positive': 0.8,
        'Neutral': 0.0,
        'Negative': -0.8
    }
    
    # Update only the sentiment score
    comments_df.loc[selected_comment_idx, '_sentiment_score'] = sentiment_scores[corrected_sentiment]
    
    # Update the display sentiment
    score = sentiment_scores[corrected_sentiment]
    comments_df.loc[selected_comment_idx, 'Sentiment'] = f"{corrected_sentiment} ({score:.2f})"
    
    return comments_df

def calculate_market_metrics(comments_df):
    """Calculate market metrics using clean sentiment scores"""
    # Filter out troll comments
    valid_comments = comments_df[~comments_df['Is_Troll']]
    
    # Calculate sentiment from scores
    valid_comments['Clean_Sentiment'] = valid_comments['Enhanced Score'].apply(
        lambda score: 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
    )
    
    # Calculate metrics
    sentiment_counts = valid_comments['Clean_Sentiment'].value_counts()
    total_valid = len(valid_comments)
    
    metrics = {
        'positive_ratio': sentiment_counts.get('Positive', 0) / total_valid,
        'negative_ratio': sentiment_counts.get('Negative', 0) / total_valid,
        'troll_ratio': len(comments_df[comments_df['Is_Troll']]) / len(comments_df)
    }
    
    return metrics

def analyze_sentiment_score(text):
    """
    Analyze sentiment and return a clean score between -1 and 1.
    """
    # Use VADER for base sentiment
    vader_score = analyze_sentiment_vader(text)
    
    # Convert VADER text output to numeric score
    if 'Positive' in vader_score:
        base_score = 0.7
    elif 'Negative' in vader_score:
        base_score = -0.7
    else:
        base_score = 0.0
        
    # Adjust score based on emojis and other factors
    processed = preprocess_text(text)
    if processed['emojis']:
        # Simple emoji adjustment
        positive_emojis = len([e for e in processed['emojis'] if e in '😊😃😄😁👍❤️'])
        negative_emojis = len([e for e in processed['emojis'] if e in '😢😭😠😡👎'])
        emoji_adjustment = (positive_emojis - negative_emojis) * 0.1
        base_score += emoji_adjustment
    
    # Ensure score is between -1 and 1
    return max(min(base_score, 1.0), -1.0)

def get_sentiment_type(score):
    """
    Convert sentiment score to sentiment type.
    """
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_english_sentiment_score(text):
    """
    Analyze English text sentiment and return score between -1 and 1.
    """
    return analyze_sentiment_score(text)

def analyze_tagalog_sentiment_score(text):
    """
    Analyze Tagalog text sentiment and return score between -1 and 1.
    """
    # For now, use same analysis as English but could be enhanced for Tagalog
    return analyze_sentiment_score(text)
