        # market_trend_analysis.py
"""
Market trend analysis functions for TikTok Sentiment Analysis.
This module provides functions to analyze purchase intent and predict market trends
based on sentiment analysis of social media comments.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import linregress
from datetime import datetime

def detect_purchase_intent(text_series):
    """
    Enhanced analysis of text to detect purchase intent signals.
    Returns scores from 0-1 where higher scores = stronger buying signals.
    Expanded parameter detection for broader coverage of buying intent indicators.
    
    Args:
        text_series: Series or list of text strings to analyze
        
    Returns:
        List of scores from 0.0-1.0 where higher values indicate stronger purchase intent
    """
    import re
    
    # Enhanced keywords indicating purchase intent
    intent_phrases = [
        # Direct intent phrases
        r'(want|need|going|plan|intend) to (buy|get|purchase|order|acquire|own)',
        r'(will|gonna|about to) (buy|get|purchase|order|pick up|grab)',
        r'i\'m (buying|getting|purchasing|ordering)',
        r'adding to (cart|basket|wishlist|favorites)',
        r'take my money',
        r'shut up and take my money',
        r'where (can|do|could|should|would) (i|you|we|someone) (buy|get|purchase|find|order)',
        r'looking to (buy|purchase|get|acquire|invest in)',
        r'(buying|purchasing|getting|ordering) this',
        r'can\'t wait to (buy|get|have|own|try)',
        r'need this in my life',
        r'(ordering|buying|purchasing) (this|now|soon|today|tomorrow)',
        r'purchased',
        r'bought',
        r'(in|added to) my cart',
        r'on my wishlist',
        r'link to (buy|purchase|shop|order)',
        r'link in bio',
        r'where to (buy|shop|find|order)',
        
        # Additional social indicators
        r'must (have|buy|get|own)',
        r'everyone (needs|should get|should buy)',
        r'(obsessed|in love) with this',
        r'best purchase',
        r'worth (every penny|the money|the price)',
        r'(perfect|ideal|amazing) for (me|my)',
        r'i (need|want) one',
        r'(ordering|buying) (another|a second)',
        r'just (bought|ordered|purchased)',
        r'already (bought|ordered|purchased)',
        r'selling out (fast|quickly)',
        r'add to cart',
        r'buy now',
        r'take my money',
        r'drop the link',
        r'where did you (get|buy)',
        r'where can i (get|buy|find|order)',
        r'how much (is it|does it cost)',
        r'(great|good|amazing) (deal|price|value)',
        r'invest in this'
    ]
    
    # Build regex pattern for all phrases
    intent_pattern = '|'.join(intent_phrases)
    
    # Expanded question phrases indicating consideration
    question_phrases = [
        r'(is|are) (it|they|this|these) worth',
        r'should i (buy|get|invest in|spend money on)',
        r'(is|are) (it|they|this|these) (good|great|worth|quality|durable|reliable)',
        r'(how|what) (is|are) the (quality|price|cost|value|durability|performance)',
        r'does (it|this) (work|perform|last|hold up)',
        r'how (much|long|well|quickly)',
        r'(good|worth) (investment|purchase|buy)',
        r'anyone (tried|bought|used|have)',
        r'thoughts on',
        r'reviews on',
        r'recommend',
        r'comparable to',
        r'better than',
        r'thinking (about|of) (getting|buying)',
        r'would you recommend'
    ]
    question_pattern = '|'.join(question_phrases)
    
    # Social proof phrases that indirectly indicate purchase likelihood
    social_proof_phrases = [
        r'(everyone|everybody) (has|loves|wants|needs) (this|it|one)',
        r'(trending|viral|popular)',
        r'(sold out|selling fast)',
        r'(best seller|top rated)',
        r'influencer favorite',
        r'celebrity endorsed',
        r'(tiktok|instagram) made me buy it',
        r'(worth the hype|lives up to the hype)',
        r'(my friend|everyone i know) (has|got|loves) (this|it)'
    ]
    social_pattern = '|'.join(social_proof_phrases)
    
    # Positive review phrases (hint at validation for others to buy)
    review_phrases = [
        r'(highly|definitely|absolutely) recommend',
        r'(love|loving|loved) (this|it|mine)',
        r'(best|greatest|amazing) (purchase|investment|decision)',
        r'(changed|improved) my life',
        r'(excellent|fantastic|incredible|outstanding) (quality|value|product)',
        r'(five|5) stars?',
        r'would (buy|purchase|get) again',
        r'(satisfied|happy|impressed) with (my purchase|this|it)'
    ]
    review_pattern = '|'.join(review_phrases)
    
    results = []
    
    for text in text_series:
        if not isinstance(text, str):
            results.append(0.0)
            continue
            
        text = text.lower()
        
        # Check for direct purchase intent
        intent_matches = len(re.findall(intent_pattern, text))
        
        # Check for consideration questions
        question_matches = len(re.findall(question_pattern, text))
        
        # Check for social proof indicators
        social_matches = len(re.findall(social_pattern, text))
        
        # Check for positive review indicators
        review_matches = len(re.findall(review_pattern, text))
        
        # Check for specific actions
        actions = {
            'purchase_completed': any(word in text for word in ['bought', 'purchased', 'ordered', 'received', 'arrived', 'delivered', 'own', 'mine']),
            'future_intent': any(phrase in text for phrase in ['will buy', 'gonna buy', 'planning to', 'going to buy', 'about to order', 'adding to cart']),
            'desire': any(word in text for word in ['want', 'need', 'wish', 'hoping', 'must have', 'obsessed', 'love this', 'can\'t wait']),
            'urgency': any(phrase in text for phrase in ['selling out', 'limited stock', 'while supplies last', 'almost gone', 'back in stock', 'exclusive'])
        }
        
        # Calculate enhanced intent score
        intent_score = min(1.0, (
            0.8 * intent_matches +
            0.3 * question_matches +
            0.9 * actions['purchase_completed'] +
            0.7 * actions['future_intent'] +
            0.5 * actions['desire'] +
            0.4 * social_matches +
            0.3 * review_matches +
            0.6 * actions['urgency']
        ))
        
        results.append(intent_score)
    
    return results

def calculate_market_trend_score(df):
    """
    Calculate a market trend score that predicts buying behavior.
    Higher scores indicate stronger positive market trends.
    Enhanced with stronger correlation between positive sentiment and purchase intent.
    
    Args:
        df (pandas.DataFrame): DataFrame with Enhanced Sentiment column
        
    Returns:
        tuple: (trend_summary dict, updated DataFrame)
    """
    # Create a working copy
    data = df.copy()
    
    # 1. Extract sentiment scores
    data['sentiment_value'] = data['Enhanced Sentiment'].apply(
        lambda x: 1.0 if 'Positive' in x else (-1.0 if 'Negative' in x else 0.0)
    )
    
    # 2. Get sentiment magnitude (extract number in parentheses)
    def extract_score(text):
        match = re.search(r'\(([-+]?\d+\.\d+)\)', text)
        if match:
            return float(match.group(1))
        return 0.5  # default if no magnitude found
    
    data['sentiment_magnitude'] = data['Enhanced Sentiment'].apply(extract_score)
    
    # 3. Determine purchase intent
    data['purchase_intent'] = detect_purchase_intent(data['Comment'])
    
    # 4. Calculate sentiment density (percentage of positive comments)
    positive_ratio = (data['sentiment_value'] > 0).mean()
    
    # 5. Apply sentiment density multiplier to purchase intent
    # This creates a stronger relationship between positive sentiment and purchase intent
    sentiment_density_factor = 1.0 + (positive_ratio - 0.5) * 0.6  # Ranges from 0.7 to 1.3 typically
    data['adjusted_purchase_intent'] = data['purchase_intent'] * sentiment_density_factor
    data['adjusted_purchase_intent'] = data['adjusted_purchase_intent'].clip(0, 1)  # Ensure 0-1 range
    
    # 6. Calculate market trend score with enhanced weighting
    # More balanced approach with amplified interaction between sentiment and intent
    data['market_trend_score'] = (
        # Sentiment component (0.5 weight)
        0.5 * data['sentiment_value'] * data['sentiment_magnitude'] +
        # Purchase intent component (0.3 weight) 
        0.3 * data['adjusted_purchase_intent'] +
        # Interaction effect (0.2 weight) - rewards positive sentiment with intent
        0.2 * (data['sentiment_value'] > 0) * data['adjusted_purchase_intent']
    )
    
    # Scale to 0-100 range for easier interpretation
    min_score = data['market_trend_score'].min()
    max_score = data['market_trend_score'].max()
    
    if max_score > min_score:  # Avoid division by zero
        data['market_trend_score'] = 50 + 50 * (
            (data['market_trend_score'] - min_score) / (max_score - min_score) * 2 - 1
        )
    else:
        data['market_trend_score'] = 50  # Default to neutral if all scores are the same
    
    # Summarize the overall market trend
    overall_score = data['market_trend_score'].mean()
    
    # Calculate viral potential based on positive sentiment density and purchase intent
    viral_potential = (positive_ratio * 0.7 + data['adjusted_purchase_intent'].mean() * 0.3) * 100
    
    trend_summary = {
        'overall_score': overall_score,
        'positive_sentiment_ratio': positive_ratio,
        'purchase_intent_ratio': (data['adjusted_purchase_intent'] > 0.5).mean(),
        'sentiment_density_factor': sentiment_density_factor,
        'viral_potential': viral_potential,
        'trend_category': 'Strong Positive' if overall_score > 75 else
                          'Positive' if overall_score > 60 else
                          'Neutral' if overall_score > 40 else
                          'Negative' if overall_score > 25 else
                          'Strong Negative'
    }
    
    return trend_summary, data

def plot_market_prediction(data, trend_summary, save_path=None):
    """
    Visualize market trend prediction based on sentiment and intent.
    
    Args:
        data (pandas.DataFrame): DataFrame with market trend analysis
        trend_summary (dict): Summary statistics from calculate_market_trend_score
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create a figure with multiple components
    plt.figure(figsize=(12, 10))
    
    # 1. Gauge chart for overall market trend score
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    
    # Draw gauge
    gauge_colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9']
    bounds = [0, 25, 40, 60, 75, 100]
    norm = plt.Normalize(0, 100)
    
    # Draw gauge background
    for i in range(len(bounds)-1):
        ax1.add_patch(patches.Wedge(center=(0.5, 0), r=0.4, 
                                   theta1=180 - bounds[i+1] * 1.8, 
                                   theta2=180 - bounds[i] * 1.8,
                                   color=gauge_colors[i], alpha=0.8))
    
    # Fix for needle drawing - ensure we're working with a float
    overall_score = float(trend_summary['overall_score'])
    needle_angle = 180 - overall_score * 1.8
    needle_rad = np.radians(needle_angle)
    
    # Calculate needle endpoint
    x_end = 0.5 + 0.4 * np.cos(needle_rad)
    y_end = 0 + 0.4 * np.sin(needle_rad)
    
    # Draw needle as a line
    ax1.plot([0.5, x_end], [0, y_end], color='black', linewidth=2)
    
    # Add a small circle at the base of the needle
    ax1.add_patch(patches.Circle((0.5, 0), 0.02, color='black', zorder=10))
    
    # Add gauge labels
    for i, bound in enumerate(bounds):
        angle = 180 - bound * 1.8
        x = 0.5 + 0.45 * np.cos(np.radians(angle))
        y = 0 + 0.45 * np.sin(np.radians(angle))
        ax1.text(x, y, str(bound), ha='center', va='center', fontsize=10)
    
    # Add gauge categories
    categories = ['Strong\nNegative', 'Negative', 'Neutral', 'Positive', 'Strong\nPositive']
    for i, cat in enumerate(categories):
        angle = 180 - (bounds[i] + bounds[i+1]) * 0.9
        x = 0.5 + 0.3 * np.cos(np.radians(angle))
        y = 0 + 0.3 * np.sin(np.radians(angle))
        ax1.text(x, y, cat, ha='center', va='center', fontsize=9)
    
    # Add score in the center
    ax1.text(0.5, -0.15, f"Market Trend Score: {trend_summary['overall_score']:.1f}", 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(0.5, -0.25, f"Trend Category: {trend_summary['trend_category']}", 
             ha='center', va='center', fontsize=12)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axis('off')
    ax1.set_title('Market Trend Prediction', fontsize=16, pad=20)
    
    # 2. Sentiment distribution
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    sentiment_counts = (data['sentiment_value'] > 0).sum(), (data['sentiment_value'] == 0).sum(), (data['sentiment_value'] < 0).sum()
    ax2.pie(sentiment_counts, 
            labels=['Positive', 'Neutral', 'Negative'],
            colors=['#2ECC40', '#AAAAAA', '#FF4136'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Sentiment Distribution')
    
    # 3. Purchase intent distribution
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    intent_counts = (data['purchase_intent'] > 0.7).sum(), ((data['purchase_intent'] <= 0.7) & (data['purchase_intent'] > 0.3)).sum(), (data['purchase_intent'] <= 0.3).sum()
    ax3.pie(intent_counts, 
            labels=['High Intent', 'Medium Intent', 'Low Intent'],
            colors=['#0074D9', '#FFDC00', '#DDDDDD'],
            autopct='%1.1f%%',
            startangle=90)
    ax3.set_title('Purchase Intent Distribution')
    
    # 4. Sentiment vs. Intent scatter plot
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    scatter = ax4.scatter(data['sentiment_value'] * data['sentiment_magnitude'], 
                         data['purchase_intent'],
                         c=data['market_trend_score'], 
                         cmap='RdYlGn',
                         alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='Market Trend Score')
    ax4.set_xlabel('Sentiment (direction * magnitude)')
    ax4.set_ylabel('Purchase Intent')
    ax4.set_title('Sentiment vs Purchase Intent')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line if we have data points
    if len(data) > 1:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(
                data['sentiment_value'] * data['sentiment_magnitude'], 
                data['purchase_intent']
            )
            x_range = np.linspace(data['sentiment_value'].min(), data['sentiment_value'].max(), 100)
            ax4.plot(x_range, intercept + slope * x_range, 'r--', 
                    label=f'Correlation: {r_value:.2f}')
            ax4.legend()
        except Exception as e:
            print(f"Error calculating trend line: {e}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return plt.gcf()

def predict_purchase_volume(trend_summary, baseline_volume=1000):
    """
    Enhanced model to estimate potential purchase volume based on sentiment, intent and viral potential.
    
    Parameters:
    trend_summary (dict): Summary from calculate_market_trend_score
    baseline_volume (int): Baseline number of purchases for neutral sentiment
    
    Returns:
    dict: Dictionary with purchase predictions and growth projections
    """
    # Calculate score-based multiplier with enhanced sensitivity (0.4 to 2.5)
    # This turns the 0-100 score into a multiplier where:
    # - Score 50 (neutral) = 1.0x baseline
    # - Score 100 (max positive) = 2.5x baseline (increased from 2.0x)
    # - Score 0 (max negative) = 0.4x baseline (decreased from 0.5x)
    trend_multiplier = 1.0 + (trend_summary['overall_score'] - 50) / 40
    
    # Ensure multiplier stays within reasonable bounds
    trend_multiplier = max(0.4, min(2.5, trend_multiplier))
    
    # Consider sentiment density as a factor (if available)
    sentiment_boost = 1.0
    if 'sentiment_density_factor' in trend_summary:
        sentiment_boost = trend_summary['sentiment_density_factor']
    
    # Calculate viral potential boost
    viral_boost = 1.0
    if 'viral_potential' in trend_summary:
        # Viral potential transforms to a multiplier (1.0 to 1.5)
        viral_potential = trend_summary['viral_potential']
        viral_boost = 1.0 + (viral_potential - 50) / 100 if viral_potential > 50 else 1.0
    
    # Final enhanced multiplier with viral component
    enhanced_multiplier = trend_multiplier * sentiment_boost * viral_boost
    
    # Calculate predicted volume
    predicted_volume = baseline_volume * enhanced_multiplier
    
    # Calculate more sophisticated confidence interval
    # Higher confidence with more positive sentiment
    confidence_factor = 0.3 - (0.1 * trend_summary['positive_sentiment_ratio'])  # 0.2-0.3 range
    confidence_margin = confidence_factor * predicted_volume
    
    # Project growth over time (if trending positively)
    growth_projection = {}
    if trend_summary['overall_score'] > 50:
        # Calculate monthly growth rate based on viral potential
        monthly_growth_rate = 0.05  # Base growth
        if 'viral_potential' in trend_summary:
            # Viral products grow faster
            viral_growth = (trend_summary['viral_potential'] - 50) / 50 * 0.15
            monthly_growth_rate += max(0, viral_growth)
        
        # Project for 3 months
        for month in range(1, 4):
            growth_projection[f'month_{month}'] = predicted_volume * ((1 + monthly_growth_rate) ** month)
    else:
        # For negative trending products, project decline
        monthly_decline_rate = 0.05 * (50 - trend_summary['overall_score']) / 50
        for month in range(1, 4):
            growth_projection[f'month_{month}'] = predicted_volume * ((1 - monthly_decline_rate) ** month)
    
    # Create an enhanced prediction dictionary
    prediction = {
        'baseline_volume': baseline_volume,
        'trend_score': trend_summary['overall_score'],
        'trend_multiplier': trend_multiplier,
        'sentiment_boost': sentiment_boost,
        'viral_boost': viral_boost,
        'enhanced_multiplier': enhanced_multiplier,
        'predicted_volume': predicted_volume,
        'min_prediction': predicted_volume - confidence_margin,
        'max_prediction': predicted_volume + confidence_margin,
        'confidence_margin': confidence_margin,
        'confidence_factor': confidence_factor,
        'positive_sentiment_ratio': trend_summary['positive_sentiment_ratio'],
        'purchase_intent_ratio': trend_summary['purchase_intent_ratio'],
        'trend_category': trend_summary['trend_category'],
        'growth_projection': growth_projection
    }
    
    # Add viral potential if available
    if 'viral_potential' in trend_summary:
        prediction['viral_potential'] = trend_summary['viral_potential']
    
    return prediction

def generate_market_trend_report(data, product_name, prediction):
    """
    Generate a comprehensive market trend report for a product.
    
    Args:
        data (pandas.DataFrame): DataFrame with market trend analysis
        product_name (str): Name of the product
        prediction (dict): Dictionary from predict_purchase_volume
        
    Returns:
        str: Formatted report
    """
    # Format the report
    report = f"""
# Market Trend Analysis for {product_name}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary
Based on sentiment analysis of {len(data)} comments, {product_name} shows a **{prediction['trend_category']}** market trend with a score of **{prediction['trend_score']:.1f}** out of 100.

- **Positive Sentiment:** {prediction['positive_sentiment_ratio']*100:.1f}% of comments
- **Purchase Intent:** {prediction['purchase_intent_ratio']*100:.1f}% of comments show purchase intent
- **Predicted Sales Volume:** {prediction['predicted_volume']:.0f} units (±{prediction['confidence_margin']:.0f})

## Sentiment Breakdown
- Positive comments: {(data['sentiment_value'] > 0).sum()} ({(data['sentiment_value'] > 0).mean()*100:.1f}%)
- Neutral comments: {(data['sentiment_value'] == 0).sum()} ({(data['sentiment_value'] == 0).mean()*100:.1f}%)
- Negative comments: {(data['sentiment_value'] < 0).sum()} ({(data['sentiment_value'] < 0).mean()*100:.1f}%)

## Purchase Intent Analysis
- High intent (>70%): {(data['purchase_intent'] > 0.7).sum()} comments
- Medium intent (30-70%): {((data['purchase_intent'] <= 0.7) & (data['purchase_intent'] > 0.3)).sum()} comments
- Low intent (<30%): {(data['purchase_intent'] <= 0.3).sum()} comments

## Sales Prediction
With a baseline volume of {prediction['baseline_volume']} units:
- Expected market multiplier: {prediction['trend_multiplier']:.2f}x
- Predicted sales volume: {prediction['predicted_volume']:.0f} units
- Prediction range: {prediction['min_prediction']:.0f} to {prediction['max_prediction']:.0f} units

## Key Insights
"""
    
    # Add insights based on the data
    if prediction['trend_score'] > 75:
        report += "- Strong positive sentiment indicates excellent market reception\n"
        report += "- High purchase intent suggests strong sales potential\n"
        report += "- Consider increasing inventory to meet expected demand\n"
    elif prediction['trend_score'] > 60:
        report += "- Positive market reception with good purchase intent\n"
        report += "- Product is likely to perform well in the market\n"
        report += "- Regular inventory levels should be sufficient\n"
    elif prediction['trend_score'] > 40:
        report += "- Neutral market reception with mixed signals\n"
        report += "- Moderate sales performance expected\n"
        report += "- Consider product improvements to boost sentiment\n"
    else:
        report += "- Negative market reception with low purchase intent\n"
        report += "- Below average sales performance expected\n"
        report += "- Product improvements or repositioning recommended\n"
    
    # Add sample comments
    report += "\n## Sample Comments\n"
    
    # Add positive comments with high intent
    high_value_mask = (data['sentiment_value'] > 0) & (data['purchase_intent'] > 0.7)
    if high_value_mask.any():
        high_value_comments = data[high_value_mask].sort_values('market_trend_score', ascending=False)['Comment'].head(3)
        report += "\n### High-Value Positive Comments (with purchase intent)\n"
        for i, comment in enumerate(high_value_comments):
            report += f"{i+1}. \"{comment}\"\n"
    
    # Add negative comments
    negative_mask = data['sentiment_value'] < 0
    if negative_mask.any():
        negative_comments = data[negative_mask].sort_values('sentiment_value')['Comment'].head(3)
        report += "\n### Key Negative Comments\n"
        for i, comment in enumerate(negative_comments):
            report += f"{i+1}. \"{comment}\"\n"
    
    return report

# Add the Streamlit integration function
def add_market_trends_tab(comments_df):
    """
    Add a market trends tab to analyze purchase intent and market predictions.
    
    Args:
        comments_df (pandas.DataFrame): DataFrame with comment data
    """
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.header("Market Trend Analysis")
    
    # Get baseline purchase volume
    col1, col2 = st.columns(2)
    baseline_volume = col1.number_input("Baseline monthly sales volume:", min_value=100, value=1000)
    product_name = col2.text_input("Product name:", value="TikTok Product")
    
    # Calculate market trend scores
    with st.spinner("Calculating market trends..."):
        trend_summary, enhanced_df = calculate_market_trend_score(comments_df)
        prediction = predict_purchase_volume(trend_summary, baseline_volume)
    
    # Display key metrics
    st.subheader("Market Trend Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Trend Score", f"{trend_summary['overall_score']:.1f}/100", 
                trend_summary['trend_category'])
    col2.metric("Positive Sentiment", f"{trend_summary['positive_sentiment_ratio']*100:.1f}%")
    col3.metric("Purchase Intent", f"{trend_summary['purchase_intent_ratio']*100:.1f}%")
    col4.metric("Viral Potential", f"{trend_summary.get('viral_potential', 0):.1f}%")
    
    # Display sales prediction
    st.subheader("Sales Prediction")
    
    # Create a gauge chart for sales prediction
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction['predicted_volume'],
        title = {'text': "Predicted Sales Volume"},
        gauge = {
            'axis': {'range': [0, prediction['max_prediction'] * 1.2]},
            'bar': {'color': "#2ECC40"},
            'steps': [
                {'range': [0, prediction['min_prediction']], 'color': "#FFDC00"},
                {'range': [prediction['min_prediction'], prediction['max_prediction']], 'color': "#2ECC40"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': baseline_volume
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display prediction metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Volume", f"{baseline_volume:,} units")
    col1.metric("Enhanced Multiplier", f"{prediction.get('enhanced_multiplier', prediction['trend_multiplier']):.2f}x", 
               f"{(prediction.get('enhanced_multiplier', prediction['trend_multiplier'])-1)*100:.1f}%")
    
    col2.metric("Predicted Volume", f"{prediction['predicted_volume']:.0f} units")
    col2.metric("Confidence Range", f"±{prediction['confidence_margin']:.0f} units", 
                f"{prediction['min_prediction']:.0f} - {prediction['max_prediction']:.0f}")
    
    # Add growth projections if available
    if 'growth_projection' in prediction:
        growth = prediction['growth_projection']
        col3.metric("Forecasted Growth", "3-Month Projection")
        for month, value in growth.items():
            month_num = int(month.split('_')[1])
            change = (value - prediction['predicted_volume']) / prediction['predicted_volume'] * 100
            col3.metric(f"Month {month_num}", f"{value:.0f} units", f"{change:+.1f}%")
    
    # Purchase intent analysis
    st.subheader("Purchase Intent Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a histogram of purchase intent scores
        fig = px.histogram(enhanced_df, x='purchase_intent', 
                         nbins=20, 
                         title="Distribution of Purchase Intent Scores",
                         color_discrete_sequence=['#0074D9'])
        fig.update_layout(xaxis_title="Purchase Intent Score", yaxis_title="Number of Comments")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Intent vs sentiment heatmap
        intent_bins = [0, 0.3, 0.7, 1.0]
        intent_labels = ['Low', 'Medium', 'High']
        sentiment_bins = [-1.01, -0.3, 0.3, 1.01]  # Using -1.01 and 1.01 to include -1 and 1
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
        enhanced_df['intent_category'] = pd.cut(enhanced_df['purchase_intent'], bins=intent_bins, labels=intent_labels)
        enhanced_df['sentiment_category'] = pd.cut(enhanced_df['sentiment_value'], bins=sentiment_bins, labels=sentiment_labels)
        
        heat_data = pd.crosstab(enhanced_df['sentiment_category'], enhanced_df['intent_category'])
        
        # Create heatmap
        fig = px.imshow(heat_data, 
                       text_auto=True,
                       color_continuous_scale='Viridis',
                       title="Sentiment vs Purchase Intent")
        fig.update_layout(xaxis_title="Purchase Intent", yaxis_title="Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    
    # Add interaction analysis
    st.subheader("Sentiment and Intent Correlation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show correlation between intent and sentiment
        if 'adjusted_purchase_intent' in enhanced_df.columns:
            intent_plot_column = 'adjusted_purchase_intent'
            title_suffix = " (Adjusted by Sentiment Density)"
        else:
            intent_plot_column = 'purchase_intent'
            title_suffix = ""
            
        fig = px.scatter(enhanced_df, 
                       x='sentiment_value', y=intent_plot_column,
                       color='market_trend_score',
                       color_continuous_scale='RdYlGn',
                       title=f"Sentiment vs Purchase Intent{title_suffix}")
        fig.update_layout(xaxis_title="Sentiment Value", 
                        yaxis_title="Purchase Intent Score",
                        height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display viral potential indicators if available
        if 'viral_potential' in trend_summary:
            # Create viral potential gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = trend_summary['viral_potential'],
                title = {'text': "Viral Potential Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#9C27B0"},
                    'steps': [
                        {'range': [0, 30], 'color': "#E0E0E0"},
                        {'range': [30, 70], 'color': "#BA68C8"},
                        {'range': [70, 100], 'color': "#7B1FA2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            if trend_summary['viral_potential'] > 70:
                st.info("⭐ High viral potential detected! This product shows strong indicators for viral spread in social media.")
            elif trend_summary['viral_potential'] > 50:
                st.info("✅ Moderate viral potential. With targeted marketing, this product could gain momentum.")
    
    # Display market trend visualization 
    st.subheader("Market Trend Visualization")
    market_fig = plot_market_prediction(enhanced_df, trend_summary)
    st.pyplot(market_fig)
    
    # Top comments with purchase intent
    st.subheader("Top Comments with Purchase Intent")
    
    high_intent = enhanced_df[enhanced_df['purchase_intent'] > 0.7].sort_values('market_trend_score', ascending=False)
    if not high_intent.empty:
        for i, (_, row) in enumerate(high_intent.head(5).iterrows()):
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #0074D9;">
                <p style="margin: 0;"><strong>Comment {i+1}:</strong> {row['Comment']}</p>
                <p style="margin: 0; font-size: 0.8em;">Purchase Intent: {row['purchase_intent']:.2f} | Market Score: {row['market_trend_score']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No comments with high purchase intent detected.")
    
    # Show full report
    with st.expander("View Full Market Trend Report"):
        report = generate_market_trend_report(enhanced_df, product_name, prediction)
        st.markdown(report)
        
        # Allow download of the report
        report_bytes = report.encode()
        st.download_button(
            label="Download Market Report",
            data=report_bytes,
            file_name=f"{product_name.replace(' ', '_')}_market_report.md",
            mime="text/markdown",
        )
    
    # Allow download of the enhanced data
    csv = enhanced_df.to_csv(index=False)
    st.download_button(
        label="Download Market Analysis Data",
        data=csv,
        file_name="market_trend_analysis.csv",
        mime="text/csv",
    )
