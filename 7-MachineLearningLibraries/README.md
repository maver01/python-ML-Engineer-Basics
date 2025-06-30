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

[link](https://scikit-learn.org/stable/modules/preprocessing.html)

#### Imputers:

- `impute.SimpleImputer`: Use to inpute missing data. Can define strategies like mean or use simple fillna with specified value.
- `impute.KNNImputer`: Use mean k-Nearest Neighbours to fill missing data. Two samples are close if the features that neither is missing are close.
- `impute.IterativeImputer`: Multivariate imputer that estimates each feature from all the others. Use round-robin.

#### Scalers:

- `preprocessing.StandardScaler`: Use when the data follows a Gaussian distribution or when you want to compare how many standard deviations a value is from the mean.
- `preprocessing.normalize`: Use when the data does not follow a Gaussian distribution. It is particularly useful when the features are on different scales and you want to ensure that no single feature dominates the model. Example: text embeddings, when direction is more important than magnitude.
- `preprocessing.MinMaxScaler`: Use when need robustness to very small standard deviations of features and preserving zero entries in sparse data. Example: pixel values: [0, 128, 255] --> [0, 0.388, 1].
- `preprocessing.RobustScaler`: Use when there are a lot of outliers. It maintains information relative to the ones close to each other (not outliers).
- `preprocessing.Binarizer`: Tranform features into binary.

#### Encoders:

- `preprocessing.OneHotEncoder`: Transform categorical features into binary features.
- `preprocessing.OrdinalEncoder`: Accounts for infrequent features, and combines them into the same category.

#### Non-Linearity:

- `preprocessing.PolynomialFeatures`: Creates new columns with polynomial features: X1, X2 --> X1, X2, X1², X2², X1X2. Can be used to add non-linearity without using non-linear models.

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

- .transform(X) – preprocess data.
- .fit(X, y) – train the model.
- .predict(X) – make predictions.

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

## 3. Classic ML, what model to choose:

### 1. Classification with Tabular Data

- Random Forest Classifier – fast, robust, less tuning
- Gradient Boosting Classifier (e.g., HistGradientBoostingClassifier, XGBoost, LightGBM) – often highest accuracy
- Logistic Regression – strong baseline
- MLPClassifier (Neural Net) – if data is large and complex

| Metric                   | What it measures                                         | Notes                                                |
| ------------------------ | -------------------------------------------------------- | ---------------------------------------------------- |
| **Accuracy**             | Proportion of correct predictions                        | Simple, but can be misleading for imbalanced classes |
| **Precision**            | Correct positive predictions / total predicted positives | How many predicted positives are correct             |
| **Recall (Sensitivity)** | Correct positive predictions / total actual positives    | How many actual positives are detected               |
| **F1 Score**             | Harmonic mean of precision & recall                      | Balances precision and recall                        |
| **Confusion Matrix**     | Counts of TP, FP, TN, FN                                 | Detailed error analysis                              |
| **ROC AUC**              | Area under ROC curve (TPR vs FPR)                        | Measures model’s ability to distinguish classes      |

### 2. Classification with Text data

- Naive Bayes – fast, very strong baseline for text
- Logistic Regression – often better than Naive Bayes
- Gradient Boosting / Random Forest – good but slower to train
- Linear SVM – performs well on sparse text features

### 3. Regression

- Gradient Boosting Regressor – generally top performer
- Random Forest Regressor – fast, less sensitive to tuning
- Ridge / Lasso Regression – if you suspect linear relationships
- MLPRegressor – if you have many features and nonlinearities

| Metric                             | What it measures                                               | Notes                                 |
| ---------------------------------- | -------------------------------------------------------------- | ------------------------------------- |
| **Mean Squared Error (MSE)**       | Average squared difference between predicted and actual values | Penalizes larger errors more          |
| **Root Mean Squared Error (RMSE)** | Square root of MSE                                             | Same units as target, interpretable   |
| **Mean Absolute Error (MAE)**      | Average absolute difference                                    | Less sensitive to outliers than MSE   |
| **R-squared (R²)**                 | Square of correlation between predicted and real               | Ranges from 0 to 1 (higher is better) |
| **Adjusted R-squared**             | R² adjusted for number of predictors                           | Useful for multiple regression        |

### 4. Clustering (unsupervised)

- KMeans – fast, scalable, default clustering method
- DBSCAN – if clusters are irregularly shaped
- Agglomerative Clustering – if you need hierarchy

| Metric                                  | What it measures                                                  | Notes                                          |
| --------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| **Silhouette Score**                    | How well points fit their own cluster vs others                   | Ranges from -1 to +1 (higher better)           |
| **Calinski-Harabasz Index**             | Ratio of between-cluster dispersion and within-cluster dispersion | Higher score means better clusters             |
| **Davies-Bouldin Index**                | Average similarity between each cluster and its most similar one  | Lower score means better clusters              |
| **Adjusted Rand Index (ARI)**           | Similarity between clustering and ground truth labels             | Requires true labels                           |
| **Normalized Mutual Information (NMI)** | Mutual information normalized between 0 and 1                     | Measures shared information, needs true labels |
