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
    analyze_for_trolling,
    analyze_comment_with_trolling
)
# Import Tagalog sentiment functions
from tagalog_sentiment import (
    is_tagalog,
    tagalog_enhanced_sentiment_analysis,
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
# Import market trend analysis functions
from market_trend_analysis import (
    calculate_market_trend_score,
    plot_market_prediction,
    predict_purchase_volume,
    generate_market_trend_report,
    add_market_trends_tab
)

# Load API keys
load_dotenv()
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# Initialize Apify Client
client = ApifyClient(APIFY_API_TOKEN)

# Initialize troll detector
troll_detector = TrollDetector()

# Streamlit App Configuration
st.set_page_config(
    page_title="TikTok Sentiment Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("TikTok Sentiment Analysis")
st.caption("Market Trend Classification Dashboard")

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
    
    Args:
        text: Text to analyze
        language_mode: Language mode preference
        
    Returns:
        Sentiment analysis result
    """
    if language_mode is None:
        language_mode = st.session_state.get('language_mode', "Auto-detect")
    
    if language_mode == "Auto-detect":
        # Auto-detect language and apply appropriate analysis
        if is_tagalog(text):
            return tagalog_enhanced_sentiment_analysis(text)
        else:
            return enhanced_sentiment_analysis(text)  # Your existing function
    
    elif language_mode == "English Only":
        # Force English analysis regardless of language
        return enhanced_sentiment_analysis(text)  # Your existing function
    
    elif language_mode == "Tagalog Only":
        # Force Tagalog analysis regardless of language
        return tagalog_enhanced_sentiment_analysis(text)
    
    else:  # Multilingual mode
        # Always use the multilingual analyzer
        return tagalog_enhanced_sentiment_analysis(text)

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
def plot_sentiment_distribution(df, sentiment_column):
    """Create a bar chart of sentiment distribution."""
    # Extract sentiment categories without scores
    categories = df[sentiment_column].apply(lambda x: x.split(' ')[0])
    
    # Count occurrences
    counts = categories.value_counts()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        'Positive': 'green', 
        'Negative': 'red', 
        'Neutral': 'gray',
        'Troll': 'purple'  # Distinct color for trolls
    }
    
    # Create bar plot with custom colors
    bars = sns.barplot(x=counts.index, y=counts.values, 
                      palette=[colors.get(cat, 'blue') for cat in counts.index], 
                      ax=ax)
    
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Sentiment')
    
    # Add percentage labels
    total = counts.sum()
    for i, count in enumerate(counts):
        percentage = 100 * count / total
        ax.text(i, count + 5, f'{percentage:.1f}%', ha='center')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=cat) 
                      for cat, color in colors.items() if cat in counts.index]
    ax.legend(handles=legend_elements)
    
    return fig

# Create an interactive heatmap for sentiment comparison
def create_sentiment_heatmap(df):
    """Create a heatmap comparing different sentiment analysis methods."""
    # Make sure we have all the sentiment columns
    required_columns = ['VADER Sentiment', 'MNB Sentiment', 'Combined Sentiment', 'Enhanced Sentiment']
    if not all(col in df.columns for col in required_columns):
        return None
    
    # Extract just the sentiment label (not the score)
    sentiment_matrix = pd.DataFrame()
    for col in required_columns:
        sentiment_matrix[col] = df[col].apply(lambda x: x.split(' ')[0])
    
    # Calculate agreement matrix - now including Troll category
    agreement_matrix = pd.DataFrame(index=['Positive', 'Neutral', 'Negative', 'Troll'], 
                                  columns=['Positive', 'Neutral', 'Negative', 'Troll'])
    
    # Fill the matrix with zeros
    agreement_matrix = agreement_matrix.fillna(0)
    
    # Count agreements for the enhanced sentiment
    for idx, row in sentiment_matrix.iterrows():
        enhanced = row['Enhanced Sentiment']
        
        # Count VADER agreements
        vader = row['VADER Sentiment']
        agreement_matrix.at[enhanced, vader] += 1
        
        # Count MNB agreements
        mnb = row['MNB Sentiment']
        agreement_matrix.at[enhanced, mnb] += 1
        
        # Count Combined agreements
        combined = row['Combined Sentiment']
        agreement_matrix.at[enhanced, combined] += 1
    
    # Create the heatmap
    fig = px.imshow(agreement_matrix, 
                    labels=dict(x="Other Methods", y="Enhanced Sentiment", color="Agreement Count"),
                    x=agreement_matrix.columns,
                    y=agreement_matrix.index,
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

# About page Tagalog information
TAGALOG_ABOUT_TEXT = """
### Filipino/Tagalog Language Support

This application now supports sentiment analysis for:
- English language comments
- Filipino/Tagalog language comments
- Code-switching (mix of English and Tagalog)
- Regional Filipino dialects (Bisaya, Ilokano, Bikolano, etc.)
- Social media slang and TikTok-specific expressions in Tagalog

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
                with st.spinner("Analyzing comments..."):
                    # Process comments with your existing functions
                    processed_data = comments_df['Comment'].apply(preprocess_text)
                    
                    # Add processed text columns
                    comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                    comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                    comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                    
                    # Extract hashtags
                    comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                    
                    # Apply sentiment analysis
                    comments_df['VADER Sentiment'] = comments_df['Demojized'].apply(analyze_sentiment_vader)
                    
                    try:
                        comments_df['MNB Sentiment'] = train_mnb_model(comments_df['Processed Comment'])
                    except Exception as e:
                        st.warning(f"Error with MNB model: {e}. Using VADER only.")
                        comments_df['MNB Sentiment'] = "N/A"
                    
                    # Apply combined sentiment analysis
                    comments_df['Combined Sentiment'] = combined_sentiment_analysis(comments_df['Demojized'])
                    
                    # Apply enhanced sentiment analysis with language preference and troll detection
                    troll_results = comments_df['Comment'].apply(
                        lambda text: analyze_comment_with_trolling(text, language_mode)
                    )
                    comments_df['Enhanced Sentiment'] = troll_results.apply(lambda x: x['sentiment_text'])
                    comments_df['Is Troll'] = troll_results.apply(lambda x: x['is_troll'])
                    comments_df['Troll Score'] = troll_results.apply(lambda x: x['troll_score'])
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data View", "Visualizations", "Sentiment Analysis", "Statistics", "Market Trends"])
                
                with tab1:
                    # Display data
                    st.subheader("Processed Comments")
                    st.dataframe(comments_df[['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment', 'Enhanced Sentiment']])
                    
                    # Allow download of processed data
                    csv = comments_df.to_csv(index=False)
                    st.download_button(
                        label="Download processed data as CSV",
                        data=csv,
                        file_name="processed_tiktok_comments.csv",
                        mime="text/csv",
                    )
                    
                    # Sentiment Correction Feature
                    st.subheader("Sentiment Correction")
                    st.write("Select comments to manually correct their sentiment labels:")
                        
                    # Let user select a comment
                    selected_comment_idx = st.selectbox("Select comment to relabel:", 
                                                       options=comments_df.index.tolist(),
                                                       format_func=lambda x: comments_df.loc[x, 'Comment'][:50] + "...")

                    # Show current sentiment
                    current_sentiment = comments_df.loc[selected_comment_idx, 'Enhanced Sentiment']
                    st.write(f"Current sentiment: {current_sentiment}")

                    # Let user choose new sentiment
                    corrected_sentiment = st.radio("Correct sentiment:", 
                                                  options=["Positive", "Neutral", "Negative", "Troll"])

                    if st.button("Save Correction"):
                        # Save the corrected sentiment with a confidence of 1.0 (manual label)
                        comments_df.loc[selected_comment_idx, 'Enhanced Sentiment'] = f"{corrected_sentiment} (1.00)"
                        
                        # Save to a corrections file for future model training
                        correction_data = pd.DataFrame({
                            'Comment': [comments_df.loc[selected_comment_idx, 'Comment']],
                            'Corrected_Sentiment': [corrected_sentiment]
                        })
                        
                        # Append to CSV if it exists, create if it doesn't
                        try:
                            existing_corrections = pd.read_csv('sentiment_corrections.csv')
                            correction_data = pd.concat([existing_corrections, correction_data])
                        except:
                            pass
                        
                        correction_data.to_csv('sentiment_corrections.csv', index=False)
                        st.success(f"Comment sentiment corrected to {corrected_sentiment} and saved for future training.")
                    
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
                        fig = plot_sentiment_distribution(comments_df, 'Enhanced Sentiment')
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
                    # Sentiment Analysis Comparison
                    st.subheader("Sentiment Analysis Comparison")
                    
                    # Create a heatmap comparing sentiment methods
                    sentiment_heatmap = create_sentiment_heatmap(comments_df)
                    if sentiment_heatmap:
                        st.plotly_chart(sentiment_heatmap)
                    
                    # Select a comment to analyze in detail
                    st.subheader("Analyze Individual Comment")
                    selected_comment = st.selectbox("Select a comment to analyze:", comments_df['Comment'].tolist())
                    
                    if selected_comment:
                        # Display sentiment breakdown
                        breakdown = get_sentiment_breakdown_with_language(selected_comment, language_mode)
                        breakdown_fig = plot_sentiment_factors(selected_comment, breakdown)
                        st.plotly_chart(breakdown_fig)
                        
                        # Show the sentiment results from different methods
                        comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                        st.write("**Sentiment Analysis Results:**")
                        col1, col2 = st.columns(2)
                        col1.metric("VADER", comments_df.loc[comment_idx, 'VADER Sentiment'])
                        col1.metric("MNB", comments_df.loc[comment_idx, 'MNB Sentiment'])
                        col2.metric("Combined", comments_df.loc[comment_idx, 'Combined Sentiment'])
                        col2.metric("Enhanced", comments_df.loc[comment_idx, 'Enhanced Sentiment'])
                
                with tab4:
                    # Statistics
                    st.subheader("Comment Statistics")
                    
                    # Basic stats
                    stats = {
                        "Total Comments": len(comments_df),
                        "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                        "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                        "Positive Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Positive')]),
                        "Negative Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Negative')]),
                        "Neutral Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Neutral')]),
                        "Troll Comments": len(comments_df[comments_df['Is Troll'] == True]),
                        "Tagalog Comments": len(comments_df[comments_df['Comment'].apply(is_tagalog)])
                    }
                    
                    # Display stats in columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Comments", stats["Total Comments"])
                    col1.metric("Average Length", stats["Average Comment Length"])
                    col2.metric("Positive Comments", stats["Positive Comments"])
                    col2.metric("Negative Comments", stats["Negative Comments"])
                    col3.metric("Neutral Comments", stats["Neutral Comments"])
                    col3.metric("Troll Comments", stats["Troll Comments"])
                    col4.metric("Comments with Emojis", stats["Comments with Emojis"])
                    col4.metric("Tagalog Comments", stats["Tagalog Comments"])
                    
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
                with tab5:
                        # Add market trends analysis
                        add_market_trends_tab(comments_df)

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
                        processed_data = comments_df['Comment'].apply(preprocess_text)
                        
                        # Add processed text columns
                        comments_df['Processed Comment'] = processed_data.apply(lambda x: x['cleaned_text'])
                        comments_df['Emojis'] = processed_data.apply(lambda x: x['emojis'])
                        comments_df['Demojized'] = processed_data.apply(lambda x: x['demojized'])
                        
                        # Extract hashtags
                        comments_df['Hashtags'] = comments_df['Comment'].apply(extract_hashtags)
                        
                        # Apply sentiment analysis
                        with st.spinner("Performing sentiment analysis..."):
                            comments_df['VADER Sentiment'] = comments_df['Demojized'].apply(analyze_sentiment_vader)
                            
                            try:
                                comments_df['MNB Sentiment'] = train_mnb_model(comments_df['Processed Comment'])
                            except Exception as e:
                                st.warning(f"Error with MNB model: {e}. Using VADER only.")
                                comments_df['MNB Sentiment'] = "N/A"
                            
                            # Apply combined sentiment analysis
                            comments_df['Combined Sentiment'] = combined_sentiment_analysis(comments_df['Demojized'])
                            
                            # Apply enhanced sentiment analysis with language preference and troll detection
                            troll_results = comments_df['Comment'].apply(
                                lambda text: analyze_comment_with_trolling(text, language_mode)
                            )
                            comments_df['Enhanced Sentiment'] = troll_results.apply(lambda x: x['sentiment_text'])
                            comments_df['Is Troll'] = troll_results.apply(lambda x: x['is_troll'])
                            comments_df['Troll Score'] = troll_results.apply(lambda x: x['troll_score'])
                        
                        # Apply troll detection
                        troll_results = comments_df['Comment'].apply(
                            lambda text: analyze_for_trolling(text)
                        )
                        comments_df['Is Troll'] = troll_results.apply(lambda x: x['is_troll'])
                        comments_df['Troll Score'] = troll_results.apply(lambda x: x['troll_score'])
                    
                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data View", "Visualizations", "Sentiment Analysis", "Statistics", "Market Trends"])
                    
                    with tab1:
                        # Display data
                        st.subheader("Processed Comments")
                        st.dataframe(comments_df[['Comment', 'Processed Comment', 'VADER Sentiment', 'MNB Sentiment', 'Enhanced Sentiment']])
                        
                        # Allow download of processed data
                        csv = comments_df.to_csv(index=False)
                        st.download_button(
                            label="Download processed data as CSV",
                            data=csv,
                            file_name="processed_tiktok_comments.csv",
                            mime="text/csv",
                        )
                        
                        # Sentiment Correction Feature
                        st.subheader("Sentiment Correction")
                        st.write("Select comments to manually correct their sentiment labels:")
                            
                        # Let user select a comment
                        selected_comment_idx = st.selectbox("Select comment to relabel:", 
                                                           options=comments_df.index.tolist(),
                                                           format_func=lambda x: comments_df.loc[x, 'Comment'][:50] + "...")

                        # Show current sentiment
                        current_sentiment = comments_df.loc[selected_comment_idx, 'Enhanced Sentiment']
                        st.write(f"Current sentiment: {current_sentiment}")

                        # Let user choose new sentiment
                        corrected_sentiment = st.radio("Correct sentiment:", 
                                                      options=["Positive", "Neutral", "Negative", "Troll"])

                        if st.button("Save Correction"):
                            # Save the corrected sentiment with a confidence of 1.0 (manual label)
                            comments_df.loc[selected_comment_idx, 'Enhanced Sentiment'] = f"{corrected_sentiment} (1.00)"
                            
                            # Save to a corrections file for future model training
                            correction_data = pd.DataFrame({
                                'Comment': [comments_df.loc[selected_comment_idx, 'Comment']],
                                'Corrected_Sentiment': [corrected_sentiment]
                            })
                            
                            # Append to CSV if it exists, create if it doesn't
                            try:
                                existing_corrections = pd.read_csv('sentiment_corrections.csv')
                                correction_data = pd.concat([existing_corrections, correction_data])
                            except:
                                pass
                            
                            correction_data.to_csv('sentiment_corrections.csv', index=False)
                            st.success(f"Comment sentiment corrected to {corrected_sentiment} and saved for future training.")
                        
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
                            fig = plot_sentiment_distribution(comments_df, 'Enhanced Sentiment')
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
                        
                        # Troll visualization
                        st.subheader("Troll Comment Distribution")
                        troll_counts = comments_df['Is Troll'].value_counts()
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = ['red', 'green']
                        sns.barplot(x=['Troll', 'Not Troll'], 
                                   y=[troll_counts.get(True, 0), troll_counts.get(False, 0)], 
                                   palette=colors, ax=ax)
                        ax.set_title('Troll vs. Normal Comments')
                        ax.set_ylabel('Count')
                        total = len(comments_df)
                        troll_pct = 100 * troll_counts.get(True, 0) / total
                        normal_pct = 100 * troll_counts.get(False, 0) / total
                        ax.text(0, troll_counts.get(True, 0) + 5, f'{troll_pct:.1f}%', ha='center')
                        ax.text(1, troll_counts.get(False, 0) + 5, f'{normal_pct:.1f}%', ha='center')
                        st.pyplot(fig)
                    
                    with tab3:
                        # Sentiment Analysis Comparison
                        st.subheader("Sentiment Analysis Comparison")
                        
                        # Create a heatmap comparing sentiment methods
                        sentiment_heatmap = create_sentiment_heatmap(comments_df)
                        if sentiment_heatmap:
                            st.plotly_chart(sentiment_heatmap)
                        
                        # Select a comment to analyze in detail
                        st.subheader("Analyze Individual Comment")
                        selected_comment = st.selectbox("Select a comment to analyze:", comments_df['Comment'].tolist())
                        
                        if selected_comment:
                            # Display sentiment breakdown
                            breakdown = get_sentiment_breakdown_with_language(selected_comment, language_mode)
                            breakdown_fig = plot_sentiment_factors(selected_comment, breakdown)
                            st.plotly_chart(breakdown_fig)
                            
                            # Show the sentiment results from different methods
                            comment_idx = comments_df[comments_df['Comment'] == selected_comment].index[0]
                            st.write("**Sentiment Analysis Results:**")
                            col1, col2 = st.columns(2)
                            col1.metric("VADER", comments_df.loc[comment_idx, 'VADER Sentiment'])
                            col1.metric("MNB", comments_df.loc[comment_idx, 'MNB Sentiment'])
                            col2.metric("Combined", comments_df.loc[comment_idx, 'Combined Sentiment'])
                            col2.metric("Enhanced", comments_df.loc[comment_idx, 'Enhanced Sentiment'])
                            
                            # Add language detection info for the selected comment
                            is_tag = is_tagalog(selected_comment)
                            language = "Tagalog" if is_tag else "English"
                            st.info(f"Detected language: {language}")
                    
                    with tab4:
                        # Statistics
                        st.subheader("Comment Statistics")
                        
                        # Basic stats
                        stats = {
                            "Total Comments": len(comments_df),
                            "Average Comment Length": int(comments_df['Comment'].apply(len).mean()),
                            "Comments with Emojis": len(comments_df[comments_df['Emojis'] != '']),
                            "Positive Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Positive')]),
                            "Negative Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Negative')]),
                            "Neutral Comments": len(comments_df[comments_df['Enhanced Sentiment'].str.contains('Neutral')]),
                            "Troll Comments": len(comments_df[comments_df['Is Troll'] == True]),
                            "Tagalog Comments": len(comments_df[comments_df['Comment'].apply(is_tagalog)])
                        }
                        
                        # Display stats in columns
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Comments", stats["Total Comments"])
                        col1.metric("Average Length", stats["Average Comment Length"])
                        col2.metric("Positive Comments", stats["Positive Comments"])
                        col2.metric("Negative Comments", stats["Negative Comments"])
                        col3.metric("Neutral Comments", stats["Neutral Comments"])
                        col3.metric("Troll Comments", stats["Troll Comments"])
                        col4.metric("Comments with Emojis", stats["Comments with Emojis"])
                        col4.metric("Tagalog Comments", stats["Tagalog Comments"])
                        
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
                    with tab5:
                        # Add market trends analysis
                        add_market_trends_tab(comments_df)
                else:
                    st.error("Failed to fetch comments. Please check the video link and try again.")
        else:
            st.warning("Please enter a TikTok video link.")

# Sentiment Explorer page
elif page == "Sentiment Explorer":
    st.header("Sentiment Analysis Explorer")
    
    st.write("""
    This section allows you to explore how our sentiment analysis works with individual comments.
    Enter a sample comment to see how different sentiment analysis methods evaluate it.
    """)
    
    # Input for testing sentiment
    test_comment = st.text_area("Enter a comment to analyze:", "This video is amazing! The tutorial was so helpful 🔥👍")
    
    if test_comment:
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            # Process the comment
            processed = preprocess_text(test_comment)
            
            # Display the processed text
            st.subheader("Processed Text")
            st.write(f"**Original:** {test_comment}")
            st.write(f"**Cleaned:** {processed['cleaned_text']}")
            st.write(f"**Emojis Found:** {processed['emojis'] or 'None'}")
            st.write(f"**Demojized:** {processed['demojized']}")
            
            # Language detection
            is_tag = is_tagalog(test_comment)
            st.write(f"**Detected Language:** {'Tagalog' if is_tag else 'English'}")
            
            # Extract and display hashtags
            hashtags = extract_hashtags(test_comment)
            if hashtags:
                st.write(f"**Hashtags:** {', '.join(hashtags)}")
            
            # Perform sentiment analysis
            vader_sentiment = analyze_sentiment_vader(processed['demojized'])
            combined_sentiment = combined_sentiment_analysis(processed['demojized'])
            enhanced_sentiment = analyze_sentiment_with_language_preference(test_comment, language_mode)
            
            # Display sentiment results
            st.subheader("Sentiment Analysis")
            st.write(f"**VADER:** {vader_sentiment}")
            st.write(f"**Combined:** {combined_sentiment}")
            st.write(f"**Enhanced (with language detection):** {enhanced_sentiment}")
        
        with col2:
            # Display sentiment breakdown
            st.subheader("Sentiment Breakdown")
            breakdown = get_sentiment_breakdown_with_language(test_comment, language_mode)
            breakdown_fig = plot_sentiment_factors(test_comment, breakdown)
            st.plotly_chart(breakdown_fig)
            
            # Add explanation
            st.subheader("How It Works")
            st.write("""
            Our sentiment analysis combines multiple approaches:
            
            1. **VADER** - A rule-based sentiment analyzer specifically tuned for social media
            2. **ML Model** - A machine learning model trained on TikTok comments
            3. **Emoji Analysis** - Sentiment extraction from emojis
            4. **TikTok Lexicon** - Custom dictionary of TikTok-specific terms and slang
            """)
            
            # Add Tagalog-specific info if detected
            if is_tag:
                st.write("""
                5. **Tagalog Lexicon** - Custom dictionary for Tagalog words and expressions
                6. **Multilingual Analysis** - Special handling for code-switching (mixed languages)
                """)
            
            st.write("The final sentiment is a weighted combination of all methods, with language-specific optimizations.")

# Run the app
if __name__ == "__main__":
    # This ensures the app runs properly when executed directly
    pass
