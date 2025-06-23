import pandas as pd
import re

# Load the original iSarcasm2022 dataset
original_df = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/isarcasm2022.csv')

# Load the new improved dataset
new_df = pd.read_csv('/shared_data0/cgoldberg/Concept_Inversion/Data/iSarcasm/isarcasm_final_improved_20250617_204336.csv')

print(f"Original dataset size: {len(original_df)}")
print(f"New dataset size: {len(new_df)}")

# Check if original phrases are preserved
print("\n" + "="*80)
print("CHECKING PHRASE PRESERVATION")
print("="*80)

found_count = 0
missing_count = 0
missing_phrases = []

for idx, row in original_df.iterrows():
    original_tweet = row['tweet']
    
    # Skip if tweet is NaN
    if pd.isna(original_tweet):
        continue
        
    # Check if this exact phrase exists anywhere in the new dataset
    # (either in sarcasm tags or as surrounding text)
    found = False
    
    for new_idx, new_row in new_df.iterrows():
        new_text = new_row['text']
        
        # Skip if text is NaN
        if pd.isna(new_text):
            continue
            
        # Check if original tweet is contained in new text
        if original_tweet.lower() in new_text.lower():
            found = True
            break
    
    if found:
        found_count += 1
    else:
        missing_count += 1
        missing_phrases.append({
            'original_idx': idx,
            'tweet': original_tweet,
            'sarcastic': row['sarcastic']
        })
        
print(f"\nSUMMARY:")
print(f"Original phrases found: {found_count}/{len(original_df)} ({found_count/len(original_df)*100:.1f}%)")
print(f"Original phrases missing: {missing_count}/{len(original_df)} ({missing_count/len(original_df)*100:.1f}%)")

if missing_phrases:
    print(f"\nFIRST 10 MISSING PHRASES:")
    for i, missing in enumerate(missing_phrases[:10]):
        print(f"\n{i+1}. Original Index: {missing['original_idx']}")
        print(f"   Sarcastic: {missing['sarcastic']}")
        print(f"   Tweet: \"{missing['tweet'][:100]}...\"" if len(missing['tweet']) > 100 else f"   Tweet: \"{missing['tweet']}\"")

# Also check the reverse - are there entries in new dataset that don't come from original?
print(f"\n" + "="*80)
print("CHECKING FOR NEW CONTENT")
print("="*80)

new_content_count = 0
for new_idx, new_row in new_df.iterrows():
    new_text = new_row['text']
    
    # Remove sarcasm tags to get the core content
    clean_text = re.sub(r'<SARCASM>.*?<SARCASM>', '', new_text).strip()
    
    # Check if any part of this comes from original
    found_original = False
    for orig_idx, orig_row in original_df.iterrows():
        orig_tweet = orig_row['tweet']
        orig_rephrase = orig_row['rephrase']
        
        if (orig_tweet.lower() in new_text.lower() or 
            orig_rephrase.lower() in clean_text.lower()):
            found_original = True
            break
    
    if not found_original:
        new_content_count += 1

print(f"Entries with completely new content: {new_content_count}/{len(new_df)} ({new_content_count/len(new_df)*100:.1f}%)")