# Calorie Burn Prediction

This project was developed for a Kaggle Playground Series competition. The objective was to predict the number of calories burned during physical activity using physiological features. The dataset is synthetic and intended for competition use only. The competition closed on **May 31, 2025**, and the data is licensed under **Apache 2.0**.

---

## Data
The data included in this repository is the **original dataset** from which the competition data was synthetically derived.  
This dataset is licensed under **CC0: Public Domain**.

---


## Project Summary

I explored multiple machine learning models — including **Random Forest**, **XGBoost**, and **MLPRegressor** — and ultimately built an **ensemble model** using `VotingRegressor` that outperformed all individual models.

Project stages:
- **Exploratory Data Analysis** (see `Supplementary Notebooks/Calories_Burned_Exploration.ipynb`)
- **Feature Engineering** (BMI, relative intensity, and interaction terms)
- **Outlier removal**
- **Hyperparameter Tuning** using `GridSearchCV`
- **Model Validation** and **Comparison**
- **Final Ensemble Model Training and Prediction**

The final ensemble model submission achieved an **RMSLE score of 0.05955**, placing me in the top 40% in the competition.

---

## Notebooks
- `Notebooks/Calories_Burned_Ensemble.ipynb` (**Main notebook** – final model and submission)
- `Notebooks/Supplementary Work/Calories_Burned_XGBoost.ipynb` (Fine-tuned XGBoost model - secondary model)
- `Notebooks/Supplementary Work/Calories_Burned_Exploration.ipynb` (Initial EDA and early modelling)

## Key Lessons
- **Parallelization** - Learned how to speed up modeling (especially tuning) by parallelizing workloads
- **Memory Management** - Learned to choose appropriate sample sizes when using multiple cores to avoid excessive RAM usage
- **Hyperparameter Tuning** - Learned techniques for hyperparameter tuning to improve model performance

## Requirements

The following packages need to be installed:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost
