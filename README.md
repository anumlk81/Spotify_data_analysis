# Spotify Data Analysis & Popularity Prediction

A comprehensive machine learning project that analyzes Spotify music data, performs dimensionality reduction with PCA, and predicts track popularity using XGBoost classification with cross-validation.

## Project Overview

This project analyzes a Spotify dataset containing 1,717 tracks and builds a predictive model to classify songs as "Popular" or "Unpopular" based on audio and artist features. The workflow includes:

1. **Data Exploration & Cleaning**: Handle missing values, duplicates, and data inconsistencies
2. **Dimensionality Reduction (PCA)**: Reduce feature space from 5 dimensions to 4 principal components
3. **Model Development**: Train an XGBoost classifier with class imbalance handling
4. **Cross-Validation**: Evaluate using Stratified K-Fold and Stratified Group K-Fold validation
5. **Performance Analysis**: Generate comprehensive metrics and feature importance analysis

## Dataset

- **Source**: [Spotify Global Music Dataset 2009-2025](https://www.kaggle.com/datasets/wardabilal/spotify-global-music-dataset-20092025) (Kaggle)
- **Total Records**: 1,717 tracks
- **Key Features**: Track popularity, artist popularity, artist followers, album metrics, track duration, and explicit content flag

## Technologies & Libraries

- **Python 3.x**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computation
- **Scikit-Learn**: Preprocessing, PCA, cross-validation, metrics
- **XGBoost**: Gradient boosting classification with class imbalance handling
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive analysis environment

## Project Structure

```
├── datasets/
│   ├── spotify_data.csv              # Raw Spotify dataset
│   └── pca_cleaned_data.csv          # Processed data with PCA components
├── scripts/
│   ├── visualization.ipynb           # EDA and data cleaning analysis
│   ├── pca.ipynb                     # PCA dimensionality reduction
│   ├── prediction.py                 # XGBoost model and evaluation
│   └── cleaded_data.csv              # Cleaned intermediate data
├── visualization_output/             # Generated plots and visualizations
├── model_performance_metrics.txt     # Detailed model evaluation report
└── README.md                         # This file
```

## Workflow Details

### 1. Data Cleaning & Exploration
**File**: [scripts/visualization.ipynb](scripts/visualization.ipynb)

- Load raw Spotify dataset and explore shape, data types, and summary statistics
- Identify and visualize missing values (log scale visualization)
- Handle missing values: Drop rows with missing artist names (MCAR), fill album names with "Unknown"
- Remove exact duplicate rows
- Validate data consistency for downstream analysis

### 2. PCA - Dimensionality Reduction
**File**: [scripts/pca.ipynb](scripts/pca.ipynb)

**Objective**: Reduce feature dimensionality while maintaining 85% of variance

**Features Used for PCA**:
- `track_popularity`
- `artist_popularity`
- `artist_followers`
- `album_total_tracks`
- `track_duration_min`

**Process**:
1. Standardize features using `StandardScaler` (critical due to different scales: popularity 0-100 vs followers in millions)
2. Apply PCA with 0.85 variance threshold
3. Generate 4 principal components: PC1, PC2, PC3, PC4
4. Analyze component loadings to understand what each component represents
5. Create visualization of PC1 vs PC2 colored by explicit content
6. Concatenate PCA components with original data and save as `pca_cleaned_data.csv`

### 3. XGBoost Popularity Prediction
**File**: [scripts/prediction.py](scripts/prediction.py)

**Classification Task**: Binary classification of track popularity
- **Positive Class (Popular)**: Tracks with popularity ≥ 70 (475 samples, 27.6%)
- **Negative Class (Unpopular)**: Tracks with popularity < 70 (1,242 samples, 72.4%)

**Model Features**:
- `track_number`, `artist_popularity`, `artist_followers`, `track_duration_min`
- `PC1`, `PC2`, `PC3`, `PC4` (PCA components)

**XGBoost Configuration**:
- **Class Imbalance Handling**: `scale_pos_weight = 2.61` (weights minority class appropriately)
- **Hyperparameters**:
  - n_estimators: 100 boosting rounds
  - max_depth: 6 (tree depth)
  - learning_rate: 0.1
  - subsample: 0.8 (80% of samples per iteration)
  - colsample_bytree: 0.8 (80% of features per tree)
  - eval_metric: logloss (better for imbalanced classification)

**Train-Test Split**: 80-20 with stratification to maintain class distribution

### 4. Cross-Validation Strategy
**File**: [scripts/prediction.py](scripts/prediction.py)

Two K-Fold validation approaches are compared:

| CV Method | Mean F1 Score | Std Dev | Purpose |
|-----------|---------------|---------|---------|
| **Stratified K-Fold** | 0.9495 | ± 0.0054 | Standard approach maintaining class balance |
| **Stratified Group K-Fold** | 0.9286 | ± 0.0089 | Prevents artist-based leakage (groups by artist) |

**Why Stratified?**: Essential for imbalanced datasets to ensure both folds have representative class distributions

## Model Performance

### Overall Metrics
```
Accuracy:  0.9720  (97.2% of predictions correct)
Precision: 0.9313  (93.1% of positive predictions are correct)
Recall:    0.9705  (97.1% of actual popular tracks are identified)
F1 Score:  0.9505  (Balanced precision-recall metric)
ROC-AUC:   0.9974  (Excellent ranking of popular vs unpopular)
```

### Classification Report
```
              Precision  Recall  F1-Score  Support
   Unpopular      0.99     0.97      0.98     1,242
     Popular      0.93     0.97      0.95       475
   
    Accuracy                         0.97     1,717
   Macro Avg      0.96     0.97      0.97     1,717
Weighted Avg      0.97     0.97      0.97     1,717
```

### Feature Importance Ranking
The top features driving the model's predictions:

| Feature | Importance Score |
|---------|------------------|
| PC1 (PCA Component 1) | 0.261 |
| PC2 (PCA Component 2) | 0.225 |
| Artist Popularity | 0.156 |
| PC4 (PCA Component 4) | 0.141 |
| Artist Followers | 0.078 |
| Track Number | 0.070 |
| Track Duration (min) | 0.035 |
| PC3 (PCA Component 3) | 0.033 |

## Key Findings

1. **PCA Effectiveness**: 4 principal components capture 85% of variance, compressing 5 features effectively
2. **Model Performance**: XGBoost achieves 97.2% accuracy with excellent recall (97.1%), catching almost all popular tracks
3. **Feature Priority**: PCA components (PC1, PC2) are the strongest predictors, followed by artist-related features
4. **Class Imbalance Handling**: The `scale_pos_weight` parameter successfully balances the 72.4% / 27.6% class distribution
5. **Generalization**: Both cross-validation methods show strong, consistent F1 scores (0.93-0.95), indicating robust model generalization

## Output Files

| File | Description |
|------|-------------|
| `pca_cleaned_data.csv` | Dataset with PCA components added (used for model training) |
| `model_performance_metrics.txt` | Complete model evaluation report with all metrics and cross-validation results |
| `spotify_xgboost_model.pkl` | Trained and saved XGBoost model for future predictions |
| `visualization_output/` | Generated plots (confusion matrices, ROC curves, feature importance charts, CV comparison) |


## Conclusion

This project demonstrates end-to-end machine learning workflows including data cleaning, feature engineering (PCA), model training, and rigorous evaluation with cross-validation. The XGBoost model achieves exceptional performance on the Spotify popularity prediction task, with PC1 and PC2 (from PCA dimensionality reduction) emerging as the most important features for determining track popularity.