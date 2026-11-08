# Credit Card Customer Churn Capstone

A data science capstone that explores customer attrition in the BankChurners credit-card dataset. The notebook walks through data loading, SQL analysis, exploratory data analysis, cleaning, clustering, classification, and threshold selection for a retention campaign.

## Project question

Which credit-card customers are more likely to leave, and how could a bank use those signals to prioritize retention outreach?

## What the notebook covers

1. Loads the Kaggle dataset and stores it in SQLite for SQL-based questions.
2. Explores churn patterns by income, product count, and customer activity.
3. Cleans the data, removes target-leakage columns, and prepares features.
4. Uses K-Means clustering and PCA to describe customer segments.
5. Compares Logistic Regression, Random Forest, and Gradient Boosting classifiers.
6. Compares class weighting, SMOTE, and decision-threshold tuning for the imbalanced target.
7. Summarizes the model’s limits and practical next steps.

## Main result

The notebook reports a Gradient Boosting model with a 0.12 decision threshold, selected using out-of-fold training predictions under an illustrative cost assumption of $300 per missed churner and $30 per false alarm. On the held-out test set, it reports F1 of 0.81 and ROC AUC of 0.99. The notebook explains why the lower threshold favors catching more churners and discusses the uncertainty and limitations of these estimates.

These results are specific to one stratified train/test split and the dataset. The assumed costs are illustrative; they should be replaced with business-specific values before using the threshold operationally.

## Getting started

The notebook downloads the dataset using `kagglehub`:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn kagglehub
```

Open `Capstone Project.ipynb` in Jupyter or Google Colab and run the cells from top to bottom. An internet connection is required for the dataset download. The notebook creates a local `bank_churn.db` SQLite database while running.

## Dataset

BankChurners (Credit Card Customers), published on Kaggle by `sakshigoyal7`: [dataset page](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Notes

- Embedded chart images are omitted from the committed notebook to keep it compact; running the notebook regenerates the plots. Text outputs are retained.
- Explanations throughout the notebook cover data-cleaning rationale, modeling choices, evaluation metrics, cost assumptions, and limitations.
- This is an educational capstone, not a production churn system.
