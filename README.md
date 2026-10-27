# EduPulse AI — Student Performance & Academic Risk Intelligence

A complete beginner-friendly, portfolio-ready data science project. It uses synthetic student data to demonstrate exploratory analysis, machine learning, model evaluation, and an interactive Streamlit dashboard.

## Features

- Interactive overview with KPIs and charts
- Random Forest regression to estimate a final score
- Random Forest classification to estimate pass/support likelihood
- Filterable dataset explorer and CSV download
- Model evaluation (MAE, R², accuracy, precision, recall, F1)
- Synthetic dataset generator for reproducibility
- No API key, paid service, database, or real student data required

## Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit

## 1. Install Python

Use Python 3.11 or 3.12 if possible. During installation on Windows, select **Add Python to PATH**.

## 2. Open the project in VS Code

Extract the ZIP, open the `my-project` folder in VS Code, then open a terminal in that folder.

## 3. Create a virtual environment

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:
```powershell
.venv\Scripts\activate.bat
```
Or use Command Prompt and run `.venv\Scripts\activate.bat`.

If `py -3.11` is unavailable, use `python -m venv .venv`.

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Start the app

```bash
streamlit run app.py
```

Your browser should open automatically. If it does not, copy the local address shown in the terminal (usually `http://localhost:8501`) into your browser.

## Dataset

`data/student_performance.csv` contains 1,200 synthetic records. To regenerate it:
```bash
python generate_data.py
```

The data is artificial and must not be presented as real student records.

## Project structure

```text
my-project/
├── app.py
├── generate_data.py
├── requirements.txt
├── README.md
├── data/
│   └── student_performance.csv
└── models/
    └── .gitkeep
```

## Machine-learning workflow

1. Load the dataset with pandas.
2. Split predictors and targets.
3. Impute missing values and one-hot encode categorical fields.
4. Train a Random Forest regressor and classifier.
5. Evaluate on a held-out test split.
6. Display predictions and visualizations in Streamlit.

## Important limitations and responsible use

- This is an educational demonstration trained on synthetic data.
- The displayed metrics do not establish real-world accuracy.
- Do not use predictions as the sole basis for grading, admissions, discipline, or other consequential decisions.
- A real deployment requires informed consent, data minimization, representative data, privacy/security controls, bias and validity testing, and human oversight.

## Suggested future improvements

- Add SHAP explanations for individual predictions.
- Compare Random Forest with Linear Regression and Gradient Boosting.
- Add cross-validation and hyperparameter tuning.
- Build a model card and fairness evaluation report.
- Deploy the dashboard using Streamlit Community Cloud.
