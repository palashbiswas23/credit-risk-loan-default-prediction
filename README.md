# Credit Risk & Loan Default Prediction

## Project Overview
This project analyzes loan portfolio risk and predicts the likelihood of loan default using historical LendingClub loan data.

It combines SQL-based data analysis, Python machine learning, and an interactive Power BI dashboard to explore borrower characteristics, identify risk patterns, and evaluate model performance.

## Business Objectives
- Analyze loan portfolio performance and default patterns.
- Identify loan grades and borrower segments associated with higher default rates.
- Examine the relationship between borrower characteristics and loan defaults.
- Build a machine learning model to classify potential loan defaults.
- Present portfolio insights through an interactive Power BI dashboard.

## Tools & Technologies
- **SQL:** Data querying and analysis using MySQL Workbench
- **Python:** Data preparation and machine learning
- **Pandas:** Data manipulation
- **Scikit-learn:** Model training and evaluation
- **Power BI:** Interactive dashboards and visual analytics
- **Git & GitHub:** Project version control and documentation

## Dataset
The project uses a sample of LendingClub loan data.

- Portfolio analysis: 44,006 resolved loan records
- Machine learning evaluation: 8,802 test records
- Target variable: `default_flag`
  - `0` = Fully Paid
  - `1` = Charged Off / Default

Unresolved loan statuses were excluded from the model's labeled training data.

**Note:** Portfolio analysis and model evaluation use separate datasets and should be interpreted independently.

## Machine Learning
A Logistic Regression model was developed to predict loan default.

### Test Results

| Metric | Result |
|---|---:|
| Accuracy | 66.44% |
| Default Precision | 33.56% |
| Default Recall | 64.89% |
| Default F1 Score | 44.24% |
| ROC-AUC | 72.65% |

The model identifies a meaningful proportion of actual defaults, but its precision indicates that many loans predicted as defaults did not actually default.

These results are an initial baseline, not a production-ready credit decision system.

## Power BI Dashboard
The interactive dashboard contains five pages:

1. **Executive Overview** — portfolio KPIs and loan distribution.
2. **Portfolio Risk Analysis** — default patterns by grade, purpose, interest rate, home ownership, and employment length.
3. **Borrower & Vintage Analysis** — FICO scores, annual income, debt-to-income ratio, loan term, and verification status.
4. **ML Model Performance** — accuracy, precision, recall, F1 score, and confusion matrix.
5. **Credit Risk Decision Lab** — interactive filters for exploring loan risk segments.

## Key Learning Outcomes
- Data cleaning and preparation
- SQL-based portfolio analysis
- Classification model development
- Model evaluation using precision, recall, F1 score, and ROC-AUC
- Interactive dashboard design
- Translating analytical results into business insights

## Limitations
- The model is an initial baseline and requires further validation.
- Model predictions should not be used as the sole basis for lending decisions.
- Data quality and class imbalance can affect model performance.
- Sensitive borrower information must not be published.
- Results describe this dataset and may not generalize to other lending portfolios.

## Author
MBA (Finance) | Aspiring Financial Analyst