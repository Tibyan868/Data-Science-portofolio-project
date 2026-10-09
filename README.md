# Data-Science-portofolio-project
# Hi, I'm Tibyan

**Data Scientist & Data Analyst** — I turn raw, messy data into clean datasets, clear charts, and machine learning models that answer a business question.

My work follows the same path each time: understand the problem, clean the data, explore it visually, build and compare models, and report the result honestly against a baseline.

- GitHub: [github.com/Tibyan868](https://github.com/Tibyan868)

---

## Projects at a glance

| # | Project | Type | Question it answers | Best result |
|---|---------|------|---------------------|-------------|
| 1 | [Loan Approval Prediction](https://github.com/Tibyan868/loan_approval) | Classification | Should this loan application be approved? | **90% accuracy** (Decision Tree) |
| 2 | [Customer Churn Prediction](https://github.com/Tibyan868/customer-churn) | Classification | Which customers are about to leave? | **80.6% accuracy** (AdaBoost) · **86% recall** (Logistic Regression) |
| 3 | [Customer Segmentation](https://github.com/Tibyan868/segmentation) | Clustering | What kinds of customers do we have? | **4 segments**, silhouette score **0.38** |
| 4 | [Titanic Survival Prediction](https://github.com/Tibyan868/tree_pre_pruning) | Classification | Who survived, and why? | **82.7% accuracy** (pre-pruned Decision Tree) |

---

## 1. Loan Approval Prediction

**Repository:** [loan_approval](https://github.com/Tibyan868/loan_approval)

**Problem.** Reviewing loan applications by hand is slow and inconsistent. Two similar applicants can get different answers.

**Goal.** Predict whether an application will be approved from the applicant's financial and personal profile, so that screening is faster and more consistent.

**Data.** 1,000 applications, 20 columns: income, co-applicant income, credit score, debt-to-income ratio, savings, collateral value, loan amount and term, employment, education, and more. About 30% of applications were approved.

**What I did**

- Found 5% missing values in every column and filled them: mean for numeric columns, most frequent value for categorical ones.
- Explored the data with pie charts, bar charts, histograms split by approval, box plots, and a correlation heatmap.
- Encoded categories (label encoding for ordered ones, one-hot encoding for the rest) and standardised the features.
- Trained and compared four models on an 80/20 split.

**Results**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **Decision Tree** | **90.0%** | 0.83 | 0.85 | 0.84 |
| Logistic Regression | 86.5% | 0.78 | 0.77 | 0.78 |
| Gaussian Naive Bayes | 86.5% | 0.80 | 0.74 | 0.77 |
| K-Nearest Neighbours (k=7) | 74.5% | 0.61 | 0.46 | 0.52 |

Always predicting "not approved" would score 69.5%, so the best model adds about 20 points over the baseline. The scaler and model are saved together with `pickle` for reuse.

---

## 2. Customer Churn Prediction

**Repository:** [customer-churn](https://github.com/Tibyan868/customer-churn)

**Problem.** A telecom company loses about one in four customers. Winning a new customer costs more than keeping an existing one, but the company cannot tell who is about to leave.

**Goal.** Flag customers who are likely to churn early enough for the retention team to act.

**Data.** 7,043 customers, 9 columns: tenure, phone service, contract type, paperless billing, payment method, monthly charges, total charges, and churn. 26.5% of customers churned.

**What I did**

- Normalised column names and checked for nulls and duplicates.
- Found that `TotalCharges` was stored as text with 11 blank entries, converted it to numeric, and filled the blanks with the mean.
- Visualised churn by contract type, plus pair plots and a correlation heatmap. Month-to-month contracts stand out as the highest-risk group.
- Encoded categorical columns, added a tenure × contract interaction feature, and standardised the features.
- Compared Logistic Regression, Random Forest, AdaBoost, K-Nearest Neighbours, and a Decision Tree tuned with `GridSearchCV` (5-fold, scored on F1).

**Results**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| **AdaBoost** | **80.6%** | 0.67 | 0.52 | 0.59 |
| Decision Tree (tuned) | 78.5% | 0.62 | 0.47 | 0.54 |
| Random Forest | 77.9% | 0.60 | 0.50 | 0.55 |
| K-Nearest Neighbours | 77.6% | 0.59 | 0.52 | 0.55 |
| **Logistic Regression (class-weighted)** | 74.0% | 0.50 | **0.86** | **0.63** |

Always predicting "stays" would score 73.5%, so accuracy alone is a weak measure here. For churn, missing a leaving customer is the expensive mistake, which makes the class-weighted Logistic Regression the most useful model: it catches 86% of churners. A Streamlit front end (`app.py`) for single-customer predictions is in progress.

---

## 3. Customer Segmentation

**Repository:** [segmentation](https://github.com/Tibyan868/segmentation)

**Problem.** A retailer sends the same marketing to every customer, although customers differ widely in income, household, and spending.

**Goal.** Group customers into a small number of meaningful segments so that marketing can be targeted.

**Data.** 2,240 customers, 22 columns: demographics, household, spending across six product categories, purchase channels, and campaign response.

**What I did**

- Filled 24 missing income values with the median.
- Built new features: total spending, total children, customer tenure in days, a simplified education level, and a living-with (alone / partner) flag.
- Removed 4 extreme outliers in age and income.
- One-hot encoded, standardised, and reduced the data to 3 principal components with PCA.
- Chose the number of clusters with the elbow method and the silhouette score, then fitted K-Means and Agglomerative clustering (Ward linkage) and profiled each cluster.

**Results.** This is unsupervised learning, so there is no accuracy score. The elbow method pointed to 4 clusters, with a silhouette score of 0.38.

| Segment | Customers | Avg. income | Avg. total spend | Campaign response |
|---------|-----------|-------------|------------------|-------------------|
| Budget families (with partner, children at home) | 905 | 39,700 | 222 | 8% |
| Affluent couples (few children) | 534 | 72,800 | 1,237 | 17% |
| Budget single-adult households (children at home) | 444 | 37,000 | 166 | 14% |
| Affluent singles (few children) | 353 | 70,700 | 1,190 | 32% |

The two affluent segments spend about six times more than the budget segments, and affluent singles respond to campaigns four times as often as budget families. That makes them the first group to target.

---

## 4. Titanic Survival Prediction

**Repository:** [tree_pre_pruning](https://github.com/Tibyan868/tree_pre_pruning)

**Problem.** A decision tree left to grow freely memorises its training data and performs worse on new passengers.

**Goal.** Predict passenger survival, and show how pre-pruning (limiting tree depth and split size) improves accuracy on unseen data.

**Data.** 891 passengers, 12 columns. Missing values in Age (177), Cabin (687), and Embarked (2).

**What I did**

- Imputed missing values: mean for numeric columns, most frequent value for categorical ones.
- Explored survival by sex, class, and port. Women survived at 74% against 19% for men.
- Encoded categorical features and dropped free-text columns (name, ticket, cabin).
- Trained an unpruned tree as a baseline, then tested `max_depth` from 2 to 9 and `min_samples_split` from 5 to 35.
- Plotted the final tree with `plot_tree` to explain its decisions.

**Results**

| Model | Accuracy |
|-------|----------|
| Unpruned Decision Tree (baseline) | 74.9% |
| **Pre-pruned Decision Tree** (`max_depth=7`, `min_samples_split=5`) | **82.7%** |

Pre-pruning added about 8 points of accuracy while producing a smaller, more readable tree.

---

## Skills

### Python

- **Data handling:** pandas, NumPy
- **Machine learning (scikit-learn):** Logistic Regression, K-Nearest Neighbours, Naive Bayes, Decision Trees, Random Forest, AdaBoost, K-Means, Agglomerative clustering, PCA
- **Model evaluation and tuning:** train/test split, accuracy, precision, recall, F1, confusion matrix, silhouette score, `GridSearchCV`
- **Delivery:** saving models with `pickle`, Streamlit, Jupyter Notebook

### SQL

- Querying, filtering, joining, and aggregating data for analysis and reporting

### Data cleaning

- Handling missing values (mean, median, and most-frequent imputation)
- Fixing data types (numbers stored as text, date parsing)
- Checking duplicates and normalising column names
- Detecting and removing outliers
- Consolidating messy categories into clean groups
- Encoding (label and one-hot) and feature scaling
- Feature engineering (totals, tenure, interaction features)

### Data visualisation

- **Libraries:** Matplotlib, Seaborn
- **Charts:** pie and bar charts, histograms split by class, box plots, pair plots, correlation heatmaps, 3D PCA scatter plots, elbow and silhouette curves, decision tree diagrams
- **Purpose:** every chart is tied to a question, such as which contract type churns most or which features separate approved from rejected loans

---

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SQL · Jupyter Notebook · Streamlit · Git & GitHub · VS Code
