import pandas as pd
import re

# Load the original iSarcasm2022 dataset
original_df = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/isarcasm2022.csv')

# Load the new improved dataset
new_df = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/isarcasm_final_improved_20250617_204336.csv')

print(f"Original dataset size: {len(original_df)}")
print(f"New dataset size: {len(new_df)}")

# Create a single string of all new text content for faster searching
all_new_text = " ".join(new_df['text'].fillna('').str.lower())

print("\n" + "="*80)
print("CHECKING PHRASE PRESERVATION")
print("="*80)

found_count = 0
missing_count = 0
missing_phrases = []

# Check first 20 original phrases for testing
for idx, row in original_df.head(20).iterrows():
    original_tweet = row['tweet']
    
    # Skip if tweet is NaN
    if pd.isna(original_tweet):
        continue
        
    # Check if this exact phrase exists anywhere in the new dataset
    if original_tweet.lower() in all_new_text:
        found_count += 1
        print(f"✓ Found: \"{original_tweet[:60]}...\"" if len(original_tweet) > 60 else f"✓ Found: \"{original_tweet}\"")
    else:
        missing_count += 1
        missing_phrases.append({
            'original_idx': idx,
            'tweet': original_tweet,
            'sarcastic': row['sarcastic']
        })
        print(f"✗ Missing: \"{original_tweet[:60]}...\"" if len(original_tweet) > 60 else f"✗ Missing: \"{original_tweet}\"")

print(f"\nSUMMARY (first 20 entries):")
print(f"Original phrases found: {found_count}/{20} ({found_count/20*100:.1f}%)")
print(f"Original phrases missing: {missing_count}/{20} ({missing_count/20*100:.1f}%)")

if missing_phrases:
    print(f"\nMISSING PHRASES:")
    for i, missing in enumerate(missing_phrases):
        print(f"\n{i+1}. Original Index: {missing['original_idx']}")
        print(f"   Sarcastic: {missing['sarcastic']}")
        print(f"   Tweet: \"{missing['tweet']}\"")

# Show a few examples of how original phrases appear in new dataset
print(f"\n" + "="*80)
print("EXAMPLES OF HOW ORIGINAL PHRASES APPEAR IN NEW DATASET")
print("="*80)

for idx, row in original_df.head(5).iterrows():
    original_tweet = row['tweet']
    if pd.isna(original_tweet):
        continue
        
    # Find the new dataset entry containing this phrase
    for new_idx, new_row in new_df.iterrows():
        new_text = new_row['text']
        if pd.isna(new_text):
            continue
            
        if original_tweet.lower() in new_text.lower():
            print(f"\nOriginal #{idx}: \"{original_tweet}\"")
            print(f"New #{new_idx}: \"{new_text}\"")
            break