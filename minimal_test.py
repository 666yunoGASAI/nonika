print("Starting minimal test...")

try:
    import pandas
    print("Pandas imported successfully")
    
    import numpy
    print("NumPy imported successfully")
    
    # Now try to import your modules - one line at a time
    import sentiment_analysis
    print("sentiment_analysis imported successfully")
    
    import tagalog_sentiment
    print("tagalog_sentiment imported successfully")
    
except Exception as e:
    print(f"Error: {e}")

print("Minimal test completed")