import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Cleans email text by removing HTML, lowercasing, and fixing whitespace.
    """
    # Handle missing values (NaNs) by turning them into empty strings
    if not isinstance(text, str):
        return ""

    # 1. Strip HTML tags (looks for anything between < and >)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove excess whitespace (newlines, tabs, or multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    print("Loading raw data...")
    try:
        # Load the dataset you downloaded from Kaggle
        df = pd.read_csv('raw_emails.csv')
    except FileNotFoundError:
        print("Error: 'raw_emails.csv' not found.")
        print("Please download a dataset, rename it to 'raw_emails.csv', and put it in this folder.")
        return

    # --- IMPORTANT: RENAME COLUMNS TO MATCH YOUR TEAM'S SPEC ---
    # Datasets from the internet will have varying column names (e.g., 'Email Text', 'Is_Spam').
    # You MUST change the strings below to match whatever your downloaded CSV uses.
    
    # Example: df = df.rename(columns={'Message': 'text', 'Category': 'label'})
    # Assuming the Kaggle dataset uses 'Email Text' and 'Phishing Status':
    
    if 'text_combined' in df.columns:
        df = df.rename(columns={'text_combined': 'text'})
    # df = df.rename(columns={'Email Text': 'text', 'Phishing Status': 'label'}) 
    
    # For now, let's assume you've mapped them correctly or the CSV already uses 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"Error: Could not find 'text' and 'label' columns. Current columns are: {df.columns.tolist()}")
        print("Please update the script to rename your specific columns.")
        return

    print("Cleaning email text... (this might take a moment depending on file size)")
    df['text'] = df['text'].apply(clean_text)

    # Remove any emails that became completely empty after cleaning out HTML
    df = df[df['text'] != '']

    # Ensure labels are strictly integers (0 and 1)
    df['label'] = df['label'].astype(int)

    # Keep exactly the two columns your teammates require
    df = df[['text', 'label']]

    # Create the 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save to the exact path requested
    output_path = 'data/processed_emails.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSUCCESS! Saved processed data to {output_path}")

    # --- 4. REPORT CLASS BALANCE & TRAIN/TEST SPLIT ---
    print("\n" + "="*30)
    print("DATASET STATISTICS")
    print("="*30)
    
    total_emails = len(df)
    print(f"Total usable emails: {total_emails}")

    # value_counts(normalize=True) gives us the fraction, multiply by 100 for percentage
    balance = df['label'].value_counts(normalize=True) * 100
    
    # .get(1, 0) means "get the value for label 1, or default to 0 if it doesn't exist"
    print(f"Phishing (1):   {balance.get(1, 0):.2f}%")
    print(f"Legitimate (0): {balance.get(0, 0):.2f}%")

    # Generate the train/test split to prove the data is ready for Issue #3
    # stratify=df['label'] ensures the 80/20 split maintains the same class balance
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("\nSuggested Train/Test Split (80/20):")
    print(f"Training emails: {len(X_train)}")
    print(f"Testing emails:  {len(X_test)}")
    print("="*30)

if __name__ == "__main__":
    main()