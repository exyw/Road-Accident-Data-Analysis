import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# --- Define stop words ---
stop_words = set([
    'the', 'and', 'to', 'of', 'in', 'a', 'on', 'at', 'with', 'for', 'from',
    'by', 'an', 'as', 'is', 'was', 'were', 'this', 'that', 'or', 'no', 'not',
    'are', 'be', 'into', 'out', 'over', 'under', 'during', 'while', 'had', 'has'
])

# --- Load datasets ---
accident_data = pd.read_csv('datasets/accident.csv')
vehicle_data = pd.read_csv('datasets/vehicle.csv')
atmospheric_data = pd.read_csv('datasets/atmospheric_cond.csv')
road_data = pd.read_csv('datasets/road_surface_cond.csv')

# --- Merge datasets ---
merged = accident_data.merge(vehicle_data[['ACCIDENT_NO', 'TARE_WEIGHT', 'VEHICLE_YEAR_MANUF', 'VEHICLE_TYPE']], on='ACCIDENT_NO')
merged = merged.merge(atmospheric_data[['ACCIDENT_NO', 'ATMOSPH_COND']], on='ACCIDENT_NO', how='left')
merged = merged.merge(road_data[['ACCIDENT_NO', 'SURFACE_COND']], on='ACCIDENT_NO', how='left')

# --- Report missing values after merging ---
total_rows = len(merged)
missing_surface = (merged['SURFACE_COND'] == 9).sum()
missing_atmosphere = (merged['ATMOSPH_COND'] == 9).sum()
percent_missing_surface = (missing_surface / total_rows) * 100
percent_missing_atmosphere = (missing_atmosphere / total_rows) * 100
print(f"Missing SURFACE_COND: {missing_surface} rows ({percent_missing_surface:.2f}%)")
print(f"Missing ATMOSPH_COND: {missing_atmosphere} rows ({percent_missing_atmosphere:.2f}%)")

# --- Feature engineering: Vehicle age ---
merged['ACCIDENT_DATE'] = pd.to_datetime(merged['ACCIDENT_DATE'])
merged['VEHICLE_AGE'] = merged['ACCIDENT_DATE'].dt.year - merged['VEHICLE_YEAR_MANUF']

# --- Filter implausible vehicle ages and weights ---
merged = merged[(merged['VEHICLE_AGE'] >= 0) & (merged['VEHICLE_AGE'] <= 100)]
merged = merged[(merged['TARE_WEIGHT'] > 0) & (merged['TARE_WEIGHT'] <= 50000)]

# --- Define severity classification ---
def compute_severity(row):
    total = row['NO_PERSONS']
    if total == 0:
        return 'Unknown'
    
    # Weighted proportional severity score
    severity_score = (
        3 * row['NO_PERSONS_KILLED'] +
        2 * row['NO_PERSONS_INJ_2'] +
        1 * row['NO_PERSONS_INJ_3']
    ) / total

    # Categorize based on score
    if severity_score >= 2.0:
        return 'Fatal'
    elif severity_score >= 1.0:
        return 'Serious'
    elif severity_score >= 0.5:
        return 'Moderate'
    else:
        return 'Minor'
    
merged['SEVERITY'] = merged.apply(compute_severity, axis=1)

# --- Map severity to numeric values ---
severity_mapping = {
    'Minor': 1,
    'Moderate': 2,
    'Serious': 3, 
    'Fatal': 4
}
merged['severity_numeric'] = merged['SEVERITY'].map(severity_mapping)
order =["Minor", "Moderate", "Serious", "Fatal"]
merged['SEVERITY'] = pd.Categorical(merged['SEVERITY'], categories=order, ordered=True)

# --- Classify vehicle types ---
def classify_vehicle_group(code):
    if code in [1, 2, 3]:
        return 'Passenger Car'
    elif code in [4, 5, 71]:
        return 'Light Commercial'
    elif code in [6, 60, 61, 62, 63, 7, 72]:
        return 'Heavy Vehicle'
    elif code in [8, 9]:
        return 'Bus'
    elif code in [10, 11, 12, 20]:
        return 'Motorcycle'
    elif code == 13:
        return 'Bicycle'
    elif code in [14, 15, 16]:
        return 'Other Transport'
    else:
        return 'Unknown/Other'
    
merged['VEHICLE_TYPE_GROUP'] = merged['VEHICLE_TYPE'].apply(classify_vehicle_group)
merged = merged[merged['VEHICLE_TYPE_GROUP'] != 'Unknown/Other']

# --- Calculate proportions of severity by vehicle group ---
proportions = (
    merged
    .groupby(['VEHICLE_TYPE_GROUP', 'SEVERITY'], observed=True)
    .size()
    .groupby(level=0)
    .transform(lambda x: x / x.sum())  # Normalize within each VEHICLE_TYPE_GROUP
    .reset_index(name='Proportion')    # Reset index and assign column
)

# --- Extract driver behavior flags ---
merged['DCA_DESC'] = merged['DCA_DESC'].fillna("")
def extract_behavior_flags(desc):
    desc = str(desc).lower()
    return {
        'loss_of_control': int('control' in desc or 'lost control' in desc),
        'rear_end': int('rear' in desc and 'end' in desc),
        'intersection_related': int('intersection' in desc),
        'lane_change': int('lane' in desc),
        'overtaking': int('overtaking' in desc or 'overtake' in desc)
    }

behavior_df = merged['DCA_DESC'].apply(extract_behavior_flags).apply(pd.Series)
merged = pd.concat([merged.reset_index(drop=True), behavior_df.reset_index(drop=True)], axis=1)

# --- Impute missing values for correlation analysis ---
merged_corr = merged.copy()
merged_corr['SURFACE_COND'] = merged_corr['SURFACE_COND'].fillna(merged_corr['SURFACE_COND'].mode()[0])
merged_corr['ATMOSPH_COND'] = merged_corr['ATMOSPH_COND'].fillna(merged_corr['ATMOSPH_COND'].mode()[0])

merged_corr['poor_visibility'] = merged_corr['ATMOSPH_COND'].isin([2, 3, 4, 5, 6, 7]).astype(int)
merged_corr['slippery_surface'] = merged_corr['SURFACE_COND'].isin([2, 3, 4, 5]).astype(int)

# --- Visualisations and Correlation Analysis ---
# Figure 1: Distribution of Severity Classes
accidents_type_count = merged['SEVERITY'].value_counts()
figure_1, ax1 = plt.subplots(figsize=(10, 10))
accidents_type_count.plot(kind='pie', autopct='%1.1f%%', ax=ax1, title="Accident type Categorised")
ax1.set_ylabel('')
plt.tight_layout()
plt.show()

# Figure 2: Tare Weight vs Vehicle Age by Accident Severity
# Singular scatterplot
sns.scatterplot(
    data=merged,
    x='TARE_WEIGHT',
    y='VEHICLE_AGE',
    hue='SEVERITY',
    palette='Set2'
)

plt.title("Tare Weight vs Vehicle Age by Accident Severity")
plt.xlabel("Tare Weight (kg)")
plt.ylabel("Vehicle Age (years)")
plt.legend(title="Severity")
plt.show()

# Figure 3: Scatter plot matrix - separating severity classes
sns.relplot(
    data=merged, 
    x="TARE_WEIGHT", 
    y="VEHICLE_AGE", 
    col="SEVERITY", 
    col_order=order, 
    col_wrap=2, 
    kind="scatter", 
    height=4, 
    aspect=1, 
    alpha=0.6
    )
plt.subplots_adjust(top = 0.9)
plt.suptitle("Tare Weight vs Vehicle Age by Accident Severity")
plt.show()

# Figure 4: Proportional Accident Severity by Vehicle Type Group
# Plot
plt.figure(figsize=(16, 8))
sns.barplot(
    data=proportions, 
    x='VEHICLE_TYPE_GROUP', 
    y='Proportion', 
    hue='SEVERITY', 
    hue_order= order, 
    palette='magma'
)

plt.title('Proportional Accident Severity by Vehicle Type Group')
plt.xlabel('Vehicle Type Group')
plt.ylabel('Proportion of Accidents')
#plt.xticks(rotation=0, ha='center')
plt.legend(title='Severity', bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.show()

# --- Data used in Figure 4 (Proportional Accident Severity by Vehicle Type Group) ---
proportion_data = (
    merged
    .groupby(['VEHICLE_TYPE_GROUP', 'SEVERITY'], observed=True)
    .size()
    .reset_index(name='Count')
)

# Calculate proportions within each VEHICLE_TYPE_GROUP
proportion_data['Proportion'] = (
    proportion_data
    .groupby('VEHICLE_TYPE_GROUP')['Count']
    .transform(lambda x: x / x.sum())
)

# Display the data used for Figure 4
print("\n=== Proportional Data by Vehicle Type Group and Severity ===")
print(proportion_data.sort_values(by=['VEHICLE_TYPE_GROUP', 'SEVERITY']))


# Figure 5: Vehicle Age vs. Accident Severity by Vehicle Typle Group 
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=merged, 
    x='SEVERITY',
    y='VEHICLE_AGE', 
    hue='VEHICLE_TYPE_GROUP', 
    palette='coolwarm'
)

plt.title('Vehicle Age vs. Accident Severity by Vehicle Typle Group')
plt.xlabel('Accident Severity')
plt.ylabel('Vehicle Age (Years)')
plt.legend(title='Vehicle Type Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Display the boxplot statistics used for Figure 5
boxplot_data = (
    merged
    .groupby(['SEVERITY', 'VEHICLE_TYPE_GROUP'])['VEHICLE_AGE']
    .describe()
    .reset_index()
)
print("\n=== Vehicle Age Stats by Severity and Vehicle Type Group ===")
print(boxplot_data[['SEVERITY', 'VEHICLE_TYPE_GROUP', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# Figure 6: Correlation Heat Maps
# --- Compute correlation matrix ---
correlation_df = merged_corr[['severity_numeric', 'TARE_WEIGHT', 'VEHICLE_AGE', 'poor_visibility', 'slippery_surface']].corr()
clean_names = {'severity_numeric': 'Severity Score', 'TARE_WEIGHT': 'Tare Weight(kg)', 'VEHICLE_AGE': 'Vehicle Age(years)', 
               'slippery_surface': 'Slippery Surface', 'poor_visibility': 'Poor Visibility'}
# Plot
heatmap = sns.heatmap(correlation_df, annot=True, cmap='coolwarm')

# Re-label axis for heatmap
heatmap.set_xticklabels([clean_names.get(t.get_text(), t.get_text()) for t in heatmap.get_xticklabels()], rotation=45, ha='right')
heatmap.set_yticklabels([clean_names.get(t.get_text(), t.get_text()) for t in heatmap.get_yticklabels()], rotation=0)

plt.title("Correlation Matrix: Severity vs. Contextual and Vehicle Factors")
plt.tight_layout()
plt.savefig("severity_correlation_matrix.png")
plt.show()


# Figure 7: Most frequent words in Accident DCA_DESC Wordpies 
# --- Preprocess text for word frequencies ---
def preprocess_text(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # remove punctuation/numbers
    words = text.split()
    return [word for word in words if word not in stop_words]

# --- Compute most common DCA_DESC words per vehicle group ---
grouped_data = merged.groupby('VEHICLE_TYPE_GROUP')
word_frequencies = {}

for group_name, group_df in grouped_data:
    all_words = []
    for desc in group_df['DCA_DESC']:
        all_words.extend(preprocess_text(desc))
    word_counts = Counter(all_words)
    word_frequencies[group_name] = word_counts.most_common(10)

# Plot 4 vehicle types side-by-side in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

vehicle_groups = ['Motorcycle', 'Passenger Car', 'Light Commercial', 'Bus']

for i, group_name in enumerate(vehicle_groups):
    words, counts = zip(*word_frequencies.get(group_name, [('None', 1)]))
    axes[i].pie(counts, labels=words, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Most Frequent DCA Words: {group_name}')

plt.tight_layout()
plt.savefig('driver_behavior_four_vehicle_types.png')
plt.show()

# --- Supervised Learning Model Pipeline ---
# --- Encode severity class labels ---
severity_map = {'Minor': 0, 'Moderate': 1, 'Serious': 2, 'Fatal': 3}
merged['severity_class'] = merged['SEVERITY'].str.strip().str.title().map(severity_map)
merged = merged.dropna(subset=['severity_class'])

# --- Select features and target ---
features = ['TARE_WEIGHT', 'VEHICLE_AGE', 'SURFACE_COND', 'ATMOSPH_COND', 'VEHICLE_TYPE_GROUP'] + list(behavior_df.columns)
X = merged[features]
y = merged['severity_class']

# --- Define column types ---
num_cols = ['TARE_WEIGHT', 'VEHICLE_AGE'] + list(behavior_df.columns)  # Include behavior flags as numeric
cat_cols = ['SURFACE_COND', 'ATMOSPH_COND', 'VEHICLE_TYPE_GROUP']

# --- Define preprocessing pipelines ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


# --- Define Supervised Learning models ---
logistic_model = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced'))
])

tree_model = Pipeline([
    ('preprocess', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced'))
])

rf_model = Pipeline([
    ('preprocess', preprocessor),  # Use the same ColumnTransformer
    ('clf', RandomForestClassifier(n_estimators=100, 
                                   max_depth=None, 
                                   class_weight='balanced',
                                   random_state=42))
])

knn_model = Pipeline([
    ('preprocess', preprocessor),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])

# --- Train and evaluate models ---
models = {
    'Logistic Regression': logistic_model,
    'Decision Tree': tree_model,
    'Random Forest': rf_model,
    'K-Nearest Neighbors': knn_model
}

labels = ['Minor', 'Moderate', 'Serious', 'Fatal']
model_preds = {}

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model_preds[name] = preds  # store for plotting later
    print(classification_report(y_test, preds, target_names=labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

# --- Plot confusion matrices ---
fig, axes = plt.subplots(2, 2, figsize=(18, 5))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar=False)
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

fig.suptitle('Confusion Matrices for Severity Classification Models', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('confusion_matrices_all_models.png')
plt.show()