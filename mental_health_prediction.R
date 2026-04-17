# Mental Health Status Prediction Project
#
# This script loads a CSV dataset, cleans the data, performs EDA,
# trains Logistic Regression and Random Forest models, evaluates them,
# and saves predictions and plots to disk.

set.seed(42)

# -----------------------------
# Package setup
# -----------------------------
required_packages <- c(
  "tidyverse",
  "caret",
  "randomForest",
  "pROC"
)

missing_packages <- required_packages[!vapply(
  required_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]

if (length(missing_packages) > 0) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  install.packages(missing_packages)
}

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

# -----------------------------
# Paths and configuration
# -----------------------------
project_root <- getwd()
data_path <- file.path(project_root, "data", "mental_health_dataset.csv")
output_eda_dir <- file.path(project_root, "outputs", "eda")
output_model_dir <- file.path(project_root, "outputs", "models")
output_prediction_dir <- file.path(project_root, "outputs", "predictions")

if (!file.exists(data_path)) {
  stop(
    "Dataset not found at: ", data_path, "\n",
    "Place your CSV file there and make sure it includes a binary target column named 'mental_health_status'."
  )
}

# -----------------------------
# Helper functions
# -----------------------------
standardize_column_names <- function(data_frame) {
  names(data_frame) <- make.names(names(data_frame), unique = TRUE)
  data_frame
}

mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    return(NA)
  }
  names(sort(table(x), decreasing = TRUE))[1]
}

impute_training_test <- function(train_df, test_df, target_column) {
  predictor_columns <- setdiff(names(train_df), target_column)
  numeric_columns <- predictor_columns[vapply(train_df[predictor_columns], is.numeric, logical(1))]
  categorical_columns <- setdiff(predictor_columns, numeric_columns)

  impute_values <- list()

  for (column in numeric_columns) {
    med <- median(train_df[[column]], na.rm = TRUE)
    if (is.nan(med)) {
      med <- 0
    }
    train_df[[column]][is.na(train_df[[column]])] <- med
    test_df[[column]][is.na(test_df[[column]])] <- med
    impute_values[[column]] <- med
  }

  for (column in categorical_columns) {
    train_df[[column]] <- as.character(train_df[[column]])
    test_df[[column]] <- as.character(test_df[[column]])

    train_mode <- mode_value(train_df[[column]])
    if (is.na(train_mode)) {
      train_mode <- "Unknown"
    }

    train_df[[column]][is.na(train_df[[column]]) | train_df[[column]] == ""] <- train_mode
    test_df[[column]][!is.na(test_df[[column]]) & !(test_df[[column]] %in% unique(train_df[[column]]))] <- train_mode
    test_df[[column]][is.na(test_df[[column]]) | test_df[[column]] == ""] <- train_mode

    train_levels <- sort(unique(train_df[[column]]))
    train_df[[column]] <- factor(train_df[[column]], levels = train_levels)
    test_df[[column]] <- factor(test_df[[column]], levels = train_levels)

    impute_values[[column]] <- train_mode
  }

  list(train = train_df, test = test_df, impute_values = impute_values)
}

standardize_target <- function(target_vector) {
  if (is.numeric(target_vector) || is.integer(target_vector)) {
    unique_values <- sort(unique(target_vector[!is.na(target_vector)]))
    if (identical(unique_values, c(0, 1))) {
      return(factor(ifelse(target_vector == 1, "At_Risk", "Not_At_Risk"), levels = c("At_Risk", "Not_At_Risk")))
    }
    if (identical(unique_values, c(1, 2))) {
      return(factor(ifelse(target_vector == 2, "At_Risk", "Not_At_Risk"), levels = c("At_Risk", "Not_At_Risk")))
    }
    stop("Numeric target must use 0/1 or 1/2 encoding.")
  }

  target_vector <- trimws(as.character(target_vector))
  normalized_lower <- stringr::str_to_lower(target_vector)
  normalized_lower <- stringr::str_replace_all(normalized_lower, "[^a-z0-9]+", "_")
  normalized_lower <- stringr::str_replace_all(normalized_lower, "^_|_$", "")
  positive_labels <- c("1", "yes", "true", "at_risk", "risk", "depressed", "anxious", "stressed", "high", "positive")
  negative_labels <- c("0", "no", "false", "not_at_risk", "healthy", "normal", "low", "negative", "none")

  normalized <- ifelse(
    normalized_lower %in% positive_labels, "At_Risk",
    ifelse(normalized_lower %in% negative_labels, "Not_At_Risk", NA_character_)
  )

  if (anyNA(normalized)) {
    unknown_values <- sort(unique(target_vector[is.na(normalized)]))
    stop(
      "Target column contains labels that are not recognized as binary classes: ",
      paste(unknown_values, collapse = ", "),
      "\nPlease recode the target to a binary label before running the script."
    )
  }

  factor(normalized, levels = c("At_Risk", "Not_At_Risk"))
}

ensure_directory <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

plot_and_save <- function(plot_object, file_path, width = 8, height = 6) {
  ggsave(file_path, plot = plot_object, width = width, height = height, dpi = 300)
}

evaluate_predictions <- function(actual, predicted_class, predicted_probability, positive_class = "At_Risk") {
  confusion <- caret::confusionMatrix(predicted_class, actual, positive = positive_class)
  positive_rows <- confusion$table

  tp <- positive_rows[positive_class, positive_class]
  fp <- positive_rows[positive_class, setdiff(colnames(positive_rows), positive_class)]
  fn <- positive_rows[setdiff(rownames(positive_rows), positive_class), positive_class]

  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1_score <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) NA_real_ else 2 * precision * recall / (precision + recall)
  accuracy <- as.numeric(confusion$overall["Accuracy"])

  roc_object <- pROC::roc(
    response = actual,
    predictor = predicted_probability,
    levels = c("Not_At_Risk", "At_Risk"),
    direction = "<",
    quiet = TRUE
  )

  list(
    confusion = confusion,
    metrics = tibble(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1_score = f1_score,
      auc = as.numeric(pROC::auc(roc_object))
    ),
    roc = roc_object
  )
}

save_roc_plot <- function(roc_object, file_path, model_name) {
  roc_df <- tibble(
    fpr = 1 - roc_object$specificities,
    tpr = roc_object$sensitivities
  )

  roc_plot <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(color = "#2C7FB8", linewidth = 1.1) +
    geom_abline(linetype = "dashed", color = "gray50") +
    coord_equal() +
    labs(
      title = paste(model_name, "ROC Curve"),
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    theme_minimal(base_size = 12)

  plot_and_save(roc_plot, file_path, width = 7, height = 6)
}

# -----------------------------
# Load and clean data
# -----------------------------
dataset <- readr::read_csv(data_path, show_col_types = FALSE)
dataset <- standardize_column_names(dataset)

if (!"mental_health_status" %in% names(dataset)) {
  stop("The dataset must contain a target column named 'mental_health_status'.")
}

dataset$mental_health_status <- standardize_target(dataset$mental_health_status)
dataset <- dataset %>% drop_na(mental_health_status)

# Split before preprocessing to avoid data leakage.
train_index <- caret::createDataPartition(dataset$mental_health_status, p = 0.8, list = FALSE)
train_raw <- dataset[train_index, ]
test_raw <- dataset[-train_index, ]

# Remove ID-like columns if they exist, because they do not help the model.
id_candidates <- names(train_raw)[str_detect(names(train_raw), "(^id$|id$|_id$)")]
if (length(id_candidates) > 0) {
  train_raw <- train_raw %>% select(-all_of(id_candidates))
  test_raw <- test_raw %>% select(-all_of(id_candidates))
}

# Impute missing values using training data only.
imputed <- impute_training_test(train_raw, test_raw, "mental_health_status")
train_clean <- imputed$train
test_clean <- imputed$test

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
ensure_directory(output_eda_dir)

class_plot <- ggplot(train_clean, aes(x = mental_health_status, fill = mental_health_status)) +
  geom_bar(alpha = 0.9) +
  scale_fill_manual(values = c("At_Risk" = "#D95F02", "Not_At_Risk" = "#1B9E77")) +
  labs(
    title = "Mental Health Status Distribution",
    x = "Status",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
plot_and_save(class_plot, file.path(output_eda_dir, "class_distribution.png"), width = 7, height = 5)

numeric_columns <- names(train_clean)[vapply(train_clean, is.numeric, logical(1))]
predictor_numeric_columns <- setdiff(numeric_columns, "mental_health_status")

if (length(predictor_numeric_columns) > 0) {
  age_like_column <- intersect(c("age", "Age"), names(train_clean))
  if (length(age_like_column) > 0) {
    age_plot <- ggplot(train_clean, aes(x = .data[[age_like_column[1]]], fill = mental_health_status)) +
      geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
      facet_wrap(~ mental_health_status) +
      labs(
        title = "Age Distribution by Mental Health Status",
        x = "Age",
        y = "Count"
      ) +
      theme_minimal(base_size = 12)
    plot_and_save(age_plot, file.path(output_eda_dir, "age_distribution.png"), width = 8, height = 5)
  }

  sleep_like_column <- intersect(c("sleep_hours", "Sleep.Hours", "sleep_hours_per_day"), names(train_clean))
  if (length(sleep_like_column) > 0) {
    sleep_plot <- ggplot(train_clean, aes(x = mental_health_status, y = .data[[sleep_like_column[1]]], fill = mental_health_status)) +
      geom_boxplot(alpha = 0.85) +
      scale_fill_manual(values = c("At_Risk" = "#D95F02", "Not_At_Risk" = "#1B9E77")) +
      labs(
        title = "Sleep Hours by Mental Health Status",
        x = "Status",
        y = "Sleep Hours"
      ) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "none")
    plot_and_save(sleep_plot, file.path(output_eda_dir, "sleep_hours_by_status.png"), width = 7, height = 5)
  }

  correlation_data <- train_clean %>%
    select(all_of(predictor_numeric_columns)) %>%
    cor(use = "pairwise.complete.obs") %>%
    as.data.frame() %>%
    rownames_to_column("variable1") %>%
    pivot_longer(-variable1, names_to = "variable2", values_to = "correlation")

  correlation_plot <- ggplot(correlation_data, aes(x = variable1, y = variable2, fill = correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0, limits = c(-1, 1)) +
    labs(
      title = "Correlation Heatmap of Numeric Features",
      x = NULL,
      y = NULL,
      fill = "Corr"
    ) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  plot_and_save(correlation_plot, file.path(output_eda_dir, "numeric_correlation_heatmap.png"), width = 8, height = 7)
}

# -----------------------------
# Feature engineering and preprocessing
# -----------------------------
predictor_columns <- setdiff(names(train_clean), "mental_health_status")
train_predictors <- train_clean %>% select(all_of(predictor_columns))
test_predictors <- test_clean %>% select(all_of(predictor_columns))

# One-hot encode categorical variables using the training set structure.
dummy_encoder <- caret::dummyVars(~ ., data = train_predictors, fullRank = TRUE)
train_matrix <- predict(dummy_encoder, newdata = train_predictors) %>% as.data.frame()
test_matrix <- predict(dummy_encoder, newdata = test_predictors) %>% as.data.frame()

# Scale numeric features so model coefficients and distances are comparable.
preprocessor <- caret::preProcess(train_matrix, method = c("center", "scale"))
train_scaled <- predict(preprocessor, train_matrix)
test_scaled <- predict(preprocessor, test_matrix)

train_model_data <- bind_cols(train_scaled, mental_health_status = train_clean$mental_health_status)
test_model_data <- bind_cols(test_scaled, mental_health_status = test_clean$mental_health_status)

# -----------------------------
# Cross-validation setup
# -----------------------------
cv_control <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = caret::twoClassSummary,
  savePredictions = "final"
)

# -----------------------------
# Train models
# -----------------------------
logistic_model <- caret::train(
  mental_health_status ~ .,
  data = train_model_data,
  method = "glm",
  family = binomial,
  trControl = cv_control,
  metric = "ROC"
)

random_forest_model <- caret::train(
  mental_health_status ~ .,
  data = train_model_data,
  method = "rf",
  trControl = cv_control,
  metric = "ROC",
  importance = TRUE,
  tuneLength = 5
)

# -----------------------------
# Evaluate on test data
# -----------------------------
logistic_probabilities <- predict(logistic_model, newdata = test_model_data, type = "prob")[, "At_Risk"]
logistic_predictions <- predict(logistic_model, newdata = test_model_data)
logistic_results <- evaluate_predictions(
  actual = test_model_data$mental_health_status,
  predicted_class = logistic_predictions,
  predicted_probability = logistic_probabilities
)

rf_probabilities <- predict(random_forest_model, newdata = test_model_data, type = "prob")[, "At_Risk"]
rf_predictions <- predict(random_forest_model, newdata = test_model_data)
rf_results <- evaluate_predictions(
  actual = test_model_data$mental_health_status,
  predicted_class = rf_predictions,
  predicted_probability = rf_probabilities
)

metrics_table <- bind_rows(
  logistic_results$metrics %>% mutate(model = "Logistic Regression"),
  rf_results$metrics %>% mutate(model = "Random Forest")
) %>%
  select(model, everything()) %>%
  arrange(desc(f1_score), desc(auc), desc(accuracy))

best_model_name <- metrics_table$model[1]

# -----------------------------
# Save metrics and reports
# -----------------------------
ensure_directory(output_model_dir)
ensure_directory(output_prediction_dir)

readr::write_csv(metrics_table, file.path(output_model_dir, "model_metrics.csv"))

logistic_confusion_df <- as.data.frame.matrix(logistic_results$confusion$table)
rf_confusion_df <- as.data.frame.matrix(rf_results$confusion$table)

readr::write_csv(tibble::rownames_to_column(logistic_confusion_df, var = "actual"), file.path(output_model_dir, "logistic_confusion_matrix.csv"))
readr::write_csv(tibble::rownames_to_column(rf_confusion_df, var = "actual"), file.path(output_model_dir, "random_forest_confusion_matrix.csv"))

saveRDS(logistic_model, file.path(output_model_dir, "logistic_model.rds"))
saveRDS(random_forest_model, file.path(output_model_dir, "random_forest_model.rds"))
saveRDS(preprocessor, file.path(output_model_dir, "preprocessor.rds"))
saveRDS(dummy_encoder, file.path(output_model_dir, "dummy_encoder.rds"))

# -----------------------------
# Feature importance
# -----------------------------
rf_importance <- caret::varImp(random_forest_model, scale = TRUE)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

readr::write_csv(rf_importance, file.path(output_model_dir, "random_forest_feature_importance.csv"))

feature_plot <- rf_importance %>%
  slice_head(n = min(15, n())) %>%
  ggplot(aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col(fill = "#2C7FB8") +
  coord_flip() +
  labs(
    title = "Top Random Forest Feature Importance",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal(base_size = 12)
plot_and_save(feature_plot, file.path(output_eda_dir, "random_forest_feature_importance.png"), width = 8, height = 7)

# -----------------------------
# ROC curves
# -----------------------------
save_roc_plot(logistic_results$roc, file.path(output_eda_dir, "logistic_roc.png"), "Logistic Regression")
save_roc_plot(rf_results$roc, file.path(output_eda_dir, "random_forest_roc.png"), "Random Forest")

# -----------------------------
# Sample predictions
# -----------------------------
sample_predictions <- tibble(
  actual = test_model_data$mental_health_status,
  logistic_predicted_class = logistic_predictions,
  logistic_predicted_probability = round(logistic_probabilities, 4),
  rf_predicted_class = rf_predictions,
  rf_predicted_probability = round(rf_probabilities, 4)
)

readr::write_csv(sample_predictions, file.path(output_prediction_dir, "test_predictions.csv"))

# Print a small sample so the user can inspect predictions quickly.
cat("\nModel comparison table:\n")
print(metrics_table)

cat("\nBest model selected:", best_model_name, "\n")
cat("\nSample test predictions:\n")
print(head(sample_predictions, 10))

cat("\nFiles saved to:\n")
cat("- EDA plots: ", output_eda_dir, "\n", sep = "")
cat("- Model artifacts: ", output_model_dir, "\n", sep = "")
cat("- Predictions: ", output_prediction_dir, "\n", sep = "")
