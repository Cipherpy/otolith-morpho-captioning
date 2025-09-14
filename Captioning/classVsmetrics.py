import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd

df1 = pd.read_csv('/home/reshma/Otolith/captioning/otolith/llama/grayscale_GPT_scored_captions_llama.csv')
df2 = pd.read_csv('/home/reshma/Otolith/captioning/otolith/llama/Grayscale_llama_otolith_paired_captions_scores.csv')

# Identify columns with the same name in both DataFrames (other than the key, here 'Image')
common_cols = [col for col in df1.columns if col in df2.columns and col != 'Image']

# For each common column, check if the data is identical
for col in common_cols:
    # Align the data for comparison (sorted by key column to ensure correct row mapping)
    temp1 = df1[['Image', col]].sort_values('Image').reset_index(drop=True)
    temp2 = df2[['Image', col]].sort_values('Image').reset_index(drop=True)
    # If data is identical for this column in both DataFrames
    if temp1[col].equals(temp2[col]):
        # Drop the column from df2 (will only keep from df1)
        df2 = df2.drop(columns=[col])
    else:
        # If not identical, rename df2's column to avoid collision
        df2 = df2.rename(columns={col: f"{col}_df2"})

# Now you can safely merge
df = pd.merge(df1, df2, on='Image', how='left')

print(df.info())
print(df.head())


# Metrics to include
metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR']

# Compute mean automatic metric per row
df['auto_avg'] = df[metrics].mean(axis=1)

# Normalize GPT score if needed
if df['gpt_score'].max() > 1.5:
    df['gpt_score_norm'] = df['gpt_score'] / df['gpt_score'].max()
else:
    df['gpt_score_norm'] = df['gpt_score']

# Group by class and take mean
df_grouped = df.groupby('actual_label')[['auto_avg', 'gpt_score_norm']].mean().reset_index()

# Sort by GPT score or class name (optional)
df_grouped = df_grouped.sort_values('gpt_score_norm', ascending=False)

# Plot grouped barplot
plt.figure(figsize=(16, 7))
x = range(len(df_grouped))
bar_width = 0.35

plt.bar([i - bar_width/2 for i in x], df_grouped['auto_avg'], width=bar_width, label='Auto Metric Avg')
plt.bar([i + bar_width/2 for i in x], df_grouped['gpt_score_norm'], width=bar_width, label='GPT Score (Norm)')

plt.xticks(x, df_grouped['actual_label'], rotation=75, ha='right', fontsize=13)
plt.ylabel('Score (Normalized)', fontsize=14)
plt.xlabel('Species', fontsize=14)
#plt.title('Automatic Metrics vs. Human (GPT) Scores by Class', fontsize=15, fontweight='bold')
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig("llama/plots/grayscale_llama_all_classes_groupedbar_auto_vs_gptscore.png", dpi=1500,bbox_inches='tight', transparent=True)
plt.show()
