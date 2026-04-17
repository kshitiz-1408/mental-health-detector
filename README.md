# Mental Health Status Prediction in R

This project builds a complete machine learning workflow in R to predict a binary mental health status label such as `At_Risk` vs `Not_At_Risk` from a CSV dataset with demographic, lifestyle, and work-related features.

## Project Structure

- `mental_health_prediction.R` - main end-to-end training and evaluation script
- `data/mental_health_dataset.csv` - input dataset CSV
- `outputs/eda/` - EDA plots such as class balance, histograms, correlation heatmap, ROC curves, and feature importance
- `outputs/models/` - trained models, preprocessing objects, metrics, and confusion matrices
- `outputs/predictions/` - sample test predictions

## Expected Dataset

The script expects a CSV file with a target column named `mental_health_status` and feature columns such as:

- `age`
- `gender`
- `sleep_hours`
- `work_hours`
- `social_interaction`
- `exercise_hours`
- `screen_time_hours`
- `diet_quality`
- `smoking_status`
- `alcohol_use`
- any other numeric or categorical predictors relevant to mental health

The target should be binary. The script accepts values like `0/1`, `Yes/No`, `At_Risk/Not_At_Risk`, or similar labels and standardizes them internally.

## What the Script Does

1. Loads the CSV dataset.
2. Standardizes column names.
3. Standardizes the target column into a binary factor.
4. Splits the dataset into training and testing sets.
5. Handles missing values using training-set statistics only.
6. Encodes categorical variables with one-hot encoding.
7. Scales numeric features with center and scale preprocessing.
8. Runs exploratory data analysis with `ggplot2`.
9. Trains two models:
   - Logistic Regression
   - Random Forest
10. Evaluates models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - AUC / ROC
11. Compares model performance and selects the best model.
12. Produces feature importance for the Random Forest model.
13. Saves sample test predictions and model artifacts.

## Optional Enhancements Included

- 5-fold cross-validation through `caret`
- ROC curve plots
- Feature importance plot
- Saved model objects for reuse

## How to Run

1. Place your CSV file at `data/mental_health_dataset.csv`.
2. Open R or RStudio in the project folder.
3. Run:

```r
source("mental_health_prediction.R")
```

If required packages are missing, the script installs them automatically.

## Output Files

After execution, check:

- `outputs/models/model_metrics.csv`
- `outputs/models/logistic_confusion_matrix.csv`
- `outputs/models/random_forest_confusion_matrix.csv`
- `outputs/models/random_forest_feature_importance.csv`
- `outputs/predictions/test_predictions.csv`
- `outputs/eda/*.png`

## Notes

- Logistic regression is implemented as a binary classifier.
- If your dataset uses a different target column name, rename it to `mental_health_status` before running the script.
- If your source labels are not binary, convert them to a binary risk label before using this workflow.
