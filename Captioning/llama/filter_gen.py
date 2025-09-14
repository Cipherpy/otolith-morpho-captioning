import pandas as pd
import re

# Read the Excel file
df = pd.read_csv('/home/reshma/Otolith/captioning/otolith/llama/masked_llama_test_results_progress.csv')

def extract_type_sagittal(text):
    if pd.isna(text):
        return text
    # Search for "Type:" (case-insensitive)
    match = re.search(r'(Type:\s*Sagittal.*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None  # Or you can return "" if you want empty string

# Create a copy of the original DataFrame
filtered_df = df.copy()

# Apply the function to 'generated caption' column
filtered_df['generated_caption'] = filtered_df['generated_caption'].apply(extract_type_sagittal)

# Keep only rows where extraction was successful (i.e., not None)
filtered_df = filtered_df[filtered_df['generated_caption'].notna()]

# Optionally, reset index
filtered_df = filtered_df.reset_index(drop=True)

# Save to a new Excel file if needed
filtered_df.to_csv('llama_masked_output_filtered.csv', index=False)

print(filtered_df.head())
