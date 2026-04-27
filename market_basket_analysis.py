"""
Market Basket Analysis using Apriori Algorithm
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)

def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the online retail CSV file."""
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    print(f"Original dataset shape: {df.shape}")
    
    # Parse InvoiceDate as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Remove rows with invalid dates first
    invalid_dates = df['InvoiceDate'].isna().sum()
    df = df.dropna(subset=['InvoiceDate'])
    print(f"Removed {invalid_dates} rows with invalid dates")
    
    # Remove rows with missing Description
    missing_desc = df['Description'].isna().sum()
    df = df.dropna(subset=['Description'])
    print(f"Removed {missing_desc} rows with missing descriptions")
    
    # Remove rows with negative or zero quantities
    invalid_qty = (df['Quantity'] <= 0).sum()
    df = df[df['Quantity'] > 0]
    print(f"Removed {invalid_qty} rows with invalid quantities")
    
    # Remove rows with missing CustomerID
    missing_customer = df['CustomerID'].isna().sum()
    df = df.dropna(subset=['CustomerID'])
    print(f"Removed {missing_customer} rows with missing CustomerID")
    
    # Create InvoiceNo by grouping transactions by CustomerID and date
    df['InvoiceNo'] = df.groupby(['CustomerID', df['InvoiceDate'].dt.date]).ngroup()
    print(f"Created {df['InvoiceNo'].nunique()} unique invoices")
    
    print(f"\nCleaned dataset shape: {df.shape}")
    return df

def construct_baskets(df: pd.DataFrame):
    """Group transactions by InvoiceNo and create basket lists."""
    baskets = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
    print(f"\nTotal number of baskets: {len(baskets)}")
    print(f"Average items per basket: {np.mean([len(b) for b in baskets]):.2f}")
    return baskets

def segment_by_time(df: pd.DataFrame):
    """Segment baskets into 4 time periods."""
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['TimePeriod'] = df['Hour'].apply(lambda h: 
        'Morning' if 6 <= h <= 11 else
        'Afternoon' if 12 <= h <= 17 else
        'Evening' if 18 <= h <= 23 else 'Night')
    
    time_baskets = {}
    for period in ['Morning', 'Afternoon', 'Evening', 'Night']:
        period_df = df[df['TimePeriod'] == period]
        period_baskets = period_df.groupby('InvoiceNo')['Description'].apply(list).tolist()
        time_baskets[period] = period_baskets
        print(f"{period}: {len(period_baskets)} baskets")
    
    return time_baskets

def library_one_hot_encode(baskets):
    """One-hot encode baskets using mlxtend."""
    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    return pd.DataFrame(te_array, columns=te.columns_)

def mine_frequent_itemsets(encoded_df: pd.DataFrame, min_support: float):
    """Apply Apriori algorithm."""
    return apriori(encoded_df, min_support=min_support, use_colnames=True)

def extract_rules(frequent_itemsets: pd.DataFrame, min_confidence: float = 0.6):
    """Extract association rules."""
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules[(rules['confidence'] >= 0.6) & (rules['lift'] > 1)].sort_values('confidence', ascending=False)

if __name__ == "__main__":
    # Load data
    df = load_data('Rida Zubair - online_retail - online_retail.csv')
    print("\nFirst few rows:")
    print(df.head())
    
    # Construct baskets
    baskets = construct_baskets(df)
    
    # Segment by time
    print("\nTime-based segmentation:")
    time_baskets = segment_by_time(df)
    
    # Apply Apriori to Morning baskets with higher support thresholds
    print("\n" + "="*60)
    print("Applying Apriori to Morning baskets")
    print("="*60)
    
    morning_baskets = time_baskets['Morning']
    if len(morning_baskets) > 0:
        morning_encoded = library_one_hot_encode(morning_baskets)
        print(f"Encoded shape: {morning_encoded.shape}")
        print(f"Total unique items: {morning_encoded.shape[1]}")
        
        # Use higher support thresholds for faster computation
        for support in [0.1, 0.15]:
            print(f"\nSupport threshold: {support}")
            try:
                freq_items = mine_frequent_itemsets(morning_encoded, support)
                print(f"Frequent itemsets found: {len(freq_items)}")
                
                if len(freq_items) > 0:
                    print("\nTop 10 frequent itemsets:")
                    for idx, row in freq_items.nlargest(10, 'support').iterrows():
                        items = list(row['itemsets'])
                        print(f"  {items[:3]}{'...' if len(items) > 3 else ''}: support={row['support']:.4f}")
                    
                    rules = extract_rules(freq_items, min_confidence=0.5)
                    print(f"\nRules found (confidence >= 0.5, lift > 1): {len(rules)}")
                    
                    if len(rules) > 0:
                        print("\nTop 5 rules:")
                        for idx, row in rules.head(5).iterrows():
                            print(f"  {list(row['antecedents'])} => {list(row['consequents'])}")
                            print(f"    Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}")
                else:
                    print("No frequent itemsets found. Try lowering the support threshold.")
            except Exception as e:
                print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)

