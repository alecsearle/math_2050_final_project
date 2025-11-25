import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import statsmodels.api as sm

# Load and clean data
df = pd.read_csv("../2023_MCM_Problem_C_Data.csv")
df.columns = df.iloc[0]
df = df.drop(0).reset_index(drop=True)
df = df.iloc[:, 1:]

# Convert to numeric
numeric_cols = ['Number of  reported results', 'Number in hard mode',
                '1 try', '2 tries', '3 tries', '4 tries', 
                '5 tries', '6 tries', '7 or more tries (X)']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create summary statistics
summary = df[numeric_cols].describe().T
summary['Q1'] = df[numeric_cols].quantile(0.25)
summary['Q3'] = df[numeric_cols].quantile(0.75)
summary = summary[['mean', '50%', 'std', 'min', 'Q1', 'Q3', 'max']]
summary.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max']

# Clean names for report
summary.index = ['Reported Results', 'Hard Mode Players', '1 Try (%)', 
                 '2 Tries (%)', '3 Tries (%)', '4 Tries (%)', 
                 '5 Tries (%)', '6 Tries (%)', 'Failed (%)']

# Print formatted table
print("\n" + "="*90)
print("Summary Statistics")
print("="*90)
print(summary.round(2).to_string())
print("="*90)

# Save for report
summary.round(2).to_csv('wordle_summary_table.csv')
print("\n✓ Table saved to: wordle_summary_table.csv")


# ============================================================================
# DATA DESCRIPTION
# ============================================================================

# 1. Calculate avg_tries and success_rate
df['avg_tries'] = (1*df['1 try'] + 2*df['2 tries'] + 3*df['3 tries'] + 
                   4*df['4 tries'] + 5*df['5 tries'] + 6*df['6 tries'] + 
                   7*df['7 or more tries (X)']) / 100

df['success_rate'] = 100 - df['7 or more tries (X)']

print("\n" + "="*90)
print("Average Tries Statistics")
print("="*90)
print(f"Mean: {df['avg_tries'].mean():.2f} tries")
print(f"Std Dev: {df['avg_tries'].std():.2f}")
print(f"Range: {df['avg_tries'].min():.2f} to {df['avg_tries'].max():.2f}")
print(f"Success Rate: {df['success_rate'].mean():.1f}%")

# 2. Create word features
df['unique_letters'] = df['Word'].apply(lambda x: len(set(x)))
df['num_vowels'] = df['Word'].apply(lambda x: sum(1 for c in x if c in 'AEIOU'))
df['has_repeated'] = df['Word'].apply(lambda x: len(x) != len(set(x))).astype(int)

print("\n" + "="*90)
print("Word Feature Statistics")
print("="*90)
print(f"Words with 4 unique letters (repeated): {(df['unique_letters']==4).sum()}")
print(f"Words with 5 unique letters: {(df['unique_letters']==5).sum()}")
print(f"Average vowels per word: {df['num_vowels'].mean():.2f}")

# 3. Print key statistics for repeated vs unique letters
print("\n" + "="*90)
print("Difficulty: Repeated vs Unique Letters")
print("="*90)
repeated = df[df['unique_letters']==4]['avg_tries']
unique = df[df['unique_letters']==5]['avg_tries']
print(f"4 Unique Letters (Repeated): Mean = {repeated.mean():.2f}, SD = {repeated.std():.2f}")
print(f"5 Unique Letters:            Mean = {unique.mean():.2f}, SD = {unique.std():.2f}")
print(f"Difference: {repeated.mean() - unique.mean():.2f} tries")

# 4. Generate 4-panel visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel (a): Distribution of average tries
axes[0, 0].hist(df['avg_tries'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['avg_tries'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["avg_tries"].mean():.2f}')
axes[0, 0].set_xlabel('Average Number of Tries', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('(a) Distribution of Puzzle Difficulty', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Panel (b): Success rate distribution
axes[0, 1].hist(df['success_rate'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(df['success_rate'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["success_rate"].mean():.1f}%')
axes[0, 1].set_xlabel('Success Rate (%)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('(b) Distribution of Success Rates', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Panel (c): Average performance distribution
tries_cols = ['1 try', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries']
tries_dist = df[tries_cols].mean()
axes[1, 0].bar(range(1, 7), tries_dist, color='orange', alpha=0.7, edgecolor='black', width=0.6)
axes[1, 0].set_xlabel('Number of Tries', fontsize=11)
axes[1, 0].set_ylabel('Average Percentage (%)', fontsize=11)
axes[1, 0].set_title('(c) Average Player Performance Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(1, 7))
axes[1, 0].grid(axis='y', alpha=0.3)

# Panel (d): Difficulty by unique letters
data_4 = [df[df['unique_letters']==4]['avg_tries'].dropna(), 
          df[df['unique_letters']==5]['avg_tries'].dropna()]
bp = axes[1, 1].boxplot(data_4, labels=['4 Unique\n(Repeated)', '5 Unique'],
                        patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel('Average Tries', fontsize=11)
axes[1, 1].set_title('(d) Difficulty: Repeated vs Unique Letters', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_data_description.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved: figure1_data_description.png")
print("\n" + "="*90)
print("DATA DESCRIPTION CODE COMPLETE")
print("="*90)

# Save cleaned data for next steps
df.to_csv('wordle_cleaned.csv', index=False)
print("✓ Saved: wordle_cleaned.csv (for predictive modeling)")