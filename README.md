# 🚗 Road Accident Data Analysis – Victoria

This project performs a comprehensive analysis of road crash severity using the [Victoria Road Crash Dataset](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data).  
It investigates how **vehicle characteristics**, **environmental conditions**, and **driver behaviours** influence crash outcomes across Victoria, Australia.

The workflow covers **data preprocessing**, **feature engineering**, **exploratory visualisation**, and **multi-model classification** using modern machine learning techniques.

All logic is implemented in a single Python file: `road_accident_analysis.py`.

---

## Key Features
- **Data Cleaning & Preprocessing:** Handles missing values, merges multiple CSV datasets, and standardises categorical features.  
- **Exploratory Data Analysis (EDA):** Visualises crash severity trends across factors such as lighting, weather, and vehicle type.  
- **Feature Engineering:** Extracts and transforms features to improve model interpretability and accuracy.  
- **Machine Learning Models:** Trains and compares Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbour classifiers.  
- **Evaluation Metrics:** Uses confusion matrices, ROC curves, and accuracy/recall/F1 metrics for model assessment.  

---

## Project Structure
road-accident-data-analysis/
│
├── datasets.zip
├── road_accident_analysis.py
└── README.md

---

## How to Run

1. Ensure all required datasets are located in the `datasets/` subfolder.  
2. Install dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
3. Run python road_accident_analysis.py

## Required Files

The following CSV files (from the [Victoria Road Crash Dataset](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data)) must be present in the `datasets/` folder:

- `accident.csv`
- `vehicle.csv`
- `atmospheric_cond.csv`
- `road_surface_cond.csv`

> ⚠️ **Note:** File names must match exactly (case-sensitive).

---

## Outputs

- Key visualisations (severity by weather, lighting, vehicle type, etc.)
- Model performance plots (ROC curves, confusion matrices)
- Printed summaries of model accuracy and feature importance

Visual outputs are displayed and/or saved in the same directory.

---

## Dataset Source

Victorian Government Data Directory –  
[**Victoria Road Crash Data**](https://discover.data.vic.gov.au/dataset/victoria-road-crash-data)

---

## Technologies Used

- **Python 3**
- **pandas**, **matplotlib**, **seaborn**
- **scikit-learn**
- **re**
