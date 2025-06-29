# Machine Learning Libraries

Topics:

## 1. Data Preparation and Feature Engineering

### 1. Data Cleaning

- Handling Missing Values:
  - Imputation (mean/median/mode, interpolation).
  - Removing rows/columns.
- Outlier Detection and Treatment:
  - Z-score, IQR methods.
  - Clipping or transformation.
- Data Type Conversion:
  - Converting categorical to numerical.
  - Ensuring correct data types (dates, floats).

### 2. Data Transformation

- Scaling:
  - Standardization (z-score).
  - Min-max normalization.
  - Robust scaling (less sensitive to outliers).
- Encoding Categorical Variables:
  - One-hot encoding.
  - Label encoding.
  - Target/mean encoding (with caution).
- Date/Time Feature Extraction:
  - Extracting year, month, day, weekday, etc.
- Text Processing:
  - Tokenization, stemming, lemmatization.
  - Vectorization (TF-IDF, embeddings).

### 3. Feature Engineering

- Creating New Features:
  - Ratios, differences, interactions between variables.
- Binning:
  - Converting continuous variables into categories.
- Polynomial Features:
  - Generating higher-order terms.
- Aggregation:
  - Group-by aggregates (mean, sum).

### 4. Feature Selection

- Filter Methods:
  - Correlation analysis.
  - Chi-squared test.
- Wrapper Methods:
  - Recursive Feature Elimination (RFE).
- Embedded Methods:
  - Regularization (LASSO, Ridge).
  - Tree-based feature importance.

### 5. Dimensionality Reduction

- Principal Component Analysis (PCA).
- t-SNE, UMAP for visualization.
- Truncated SVD for sparse data.

## 2. Core Libraries

### 1. Scikit-learn

Widely used for classical ML algorithms (regression, classification, clustering).
Consistent API:

- .fit(X, y) – train the model.
- .predict(X) – make predictions.
- .transform(X) – preprocess data.

Common algorithms:

- Linear/Logistic Regression.
- Decision Trees, Random Forests.
- SVM, KNN.
- Clustering: K-Means, DBSCAN.

Use when:

- You are working on classical ML problems:
  - Linear/Logistic Regression.
  - Tree-based models.
  - Clustering or dimensionality reduction.
- Your data is tabular and fits in memory.
- You want to quickly prototype models with a clean, consistent API.
- You need pipelines for preprocessing and modeling.

Example use cases:

- Predicting customer churn using Random Forest.
- Clustering customers with K-Means.
- Running logistic regression on a small dataset.

Not ideal for:

- Large-scale distributed training.
- Deep learning/neural networks.

### 2. XGBoost / LightGBM / CatBoost

Use case: Gradient boosting decision trees (state-of-the-art for tabular data). Highly optimized gradient boosting.

Features:

- Regularization (lambda, alpha).
- Early stopping.
- Handling missing values.

API:

- Native (xgb.train) or scikit-learn compatible (XGBClassifier).

Use when:

- You have tabular data with complex interactions.
- You need state-of-the-art performance in regression or classification.
- You want fast training and good handling of missing values.

### 4. TensorFlow, Pytorch

Use case: Deep learning (neural networks).

TensorFlow:

- Core library for defining computation graphs.
- Tensors: Multidimensional arrays.
- Session (TF1) vs Eager Execution (TF2).
- GPU support.

Use when:

- You need deep learning, such as:
- Image classification (CNNs).
- Natural Language Processing (RNNs, Transformers).
- Time series forecasting with neural nets.
- You want production-ready deployment pipelines (TensorFlow Serving, TFLite).

Pytorch:

- Tensor operations: Similar to NumPy but can use GPU.
- Dynamic computation graph: Built on-the-fly during execution.

Use when:

- You are doing research or experimentation with deep learning, especially when:
- You need dynamic computation graphs (define-by-run).
- You want Pythonic, imperative code that’s easier to debug.
- You want excellent GPU acceleration with fine-grained control.
- You are developing models where you expect frequent architectural changes.
