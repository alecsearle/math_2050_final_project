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


# ============================================================================
# PREDICTIVE MODEL - LINEAR REGRESSION
# ============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("\n" + "="*90)
print("LINEAR REGRESSION MODEL - PREDICTING AVERAGE TRIES")
print("="*90)

# Define features and target
features = ['unique_letters', 'num_vowels', 'has_repeated']
X = df[features]
y = df['avg_tries']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

# Evaluate performance
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nModel Performance:")
print(f"  Training R²:   {train_r2:.4f} ({train_r2*100:.1f}% variance explained)")
print(f"  Testing R²:    {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"  Training RMSE: {train_rmse:.4f} tries")
print(f"  Testing RMSE:  {test_rmse:.4f} tries")

print("\nRegression Coefficients:")
print(f"  {'Feature':<20} {'Coefficient':>12} {'Effect'}")
print("  " + "-"*50)
for feature, coef in zip(features, lr_model.coef_):
    effect = "increases" if coef > 0 else "decreases"
    print(f"  {feature:<20} {coef:>12.4f}  {effect} difficulty")
print(f"  {'Intercept':<20} {lr_model.intercept_:>12.4f}")

# Interpretation
print("\nKey Findings:")
if lr_model.coef_[features.index('has_repeated')] > 0:
    print(f"  • Repeated letters increase difficulty by {lr_model.coef_[features.index('has_repeated')]:.3f} tries")
if lr_model.coef_[features.index('unique_letters')] != 0:
    print(f"  • Each additional unique letter changes difficulty by {lr_model.coef_[features.index('unique_letters')]:.3f} tries")
if lr_model.coef_[features.index('num_vowels')] != 0:
    print(f"  • Each additional vowel changes difficulty by {lr_model.coef_[features.index('num_vowels')]:.3f} tries")

# Create visualization: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Average Tries', fontsize=12)
axes[0].set_ylabel('Predicted Average Tries', fontsize=12)
axes[0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Add text box with metrics
textstr = f'R² = {test_r2:.3f}\nRMSE = {test_rmse:.3f}'
axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Feature Importance (Coefficients)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=True)

colors = ['red' if x < 0 else 'green' for x in coef_df['Coefficient']]
axes[1].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_xlabel('Coefficient Value', fontsize=12)
axes[1].set_title('Feature Importance (Regression Coefficients)', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved: figure2_regression_results.png")

# Predict example words
print("\n" + "="*90)
print("EXAMPLE PREDICTIONS FOR FUTURE WORDS")
print("="*90)

def predict_word(word):
    """Predict difficulty for a given word"""
    features_dict = {
        'unique_letters': len(set(word)),
        'num_vowels': sum(1 for c in word if c in 'AEIOU'),
        'has_repeated': int(len(word) != len(set(word)))
    }
    X_new = pd.DataFrame([features_dict])
    predicted_tries = lr_model.predict(X_new)[0]
    return predicted_tries, features_dict

test_words = ['SLATE', 'CRANE', 'AUDIO', 'EERIE', 'QUEUE']
print(f"\n{'Word':<10} {'Predicted Tries':<18} {'Unique':<8} {'Vowels':<8} {'Repeated'}")
print("-" * 65)
for word in test_words:
    pred, feat = predict_word(word)
    print(f"{word:<10} {pred:<18.2f} {feat['unique_letters']:<8} {feat['num_vowels']:<8} {'Yes' if feat['has_repeated'] else 'No'}")

print("\n" + "="*90)
print("LINEAR REGRESSION COMPLETE")
print("="*90)



































