import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set_theme(style="whitegrid")

# Load data
data_path = "../data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
print("Loading data for EDA...")
rows = []
with open(data_path, 'r') as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# 1. Price Distribution (Why we use MinMaxScaler)
plt.figure(figsize=(10, 6))
# Clean prices
prices = df['sales_price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.extract(r'(\d+\.?\d*)').astype(float)
# Drop na and extreme outliers for plotting
prices_clean = prices[0].dropna()
prices_clean = prices_clean[prices_clean < 500] 
sns.histplot(prices_clean, bins=50, kde=True, color='skyblue')
plt.title('Distribution of Product Prices (under $500)', fontsize=14)
plt.xlabel('Price ($)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('price_dist.png')
plt.close()

# 2. Missing Values (Why we handle Weight as a sentinel)
plt.figure(figsize=(10, 6))
# Calculate missing percentages
missing = df.isnull().sum() / len(df) * 100
# Keep only features with > 0% missing
missing = missing[missing > 0].sort_values(ascending=False)
sns.barplot(x=missing.values, y=missing.index, palette='viridis')
plt.title('Percentage of Missing Values per Feature', fontsize=14)
plt.xlabel('% Missing', fontsize=12)
plt.tight_layout()
plt.savefig('missing_values.png')
plt.close()

# 3. Top Brands (Why we use OneHotEncoder with Top-N)
plt.figure(figsize=(10, 6))
top_brands = df['brand'].value_counts().head(15)
sns.barplot(x=top_brands.values, y=top_brands.index, palette='magma')
plt.title('Top 15 Most Common Brands', fontsize=14)
plt.xlabel('Product Count', fontsize=12)
plt.tight_layout()
plt.savefig('brand_dist.png')
plt.close()

