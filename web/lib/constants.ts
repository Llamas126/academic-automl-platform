export const MODEL_CATALOG: { key: string; label: string; task: "classification" | "regression" }[] = [
  { key: "linear_regression", label: "Linear Regression", task: "regression" },
  { key: "ridge_regression", label: "Ridge Regression", task: "regression" },
  { key: "lasso_regression", label: "Lasso Regression", task: "regression" },
  { key: "random_forest_regressor", label: "Random Forest Regressor", task: "regression" },
  { key: "gradient_boosting_regressor", label: "Gradient Boosting Regressor", task: "regression" },
  { key: "xgboost_regressor", label: "XGBoost Regressor", task: "regression" },
  { key: "lightgbm_regressor", label: "LightGBM Regressor", task: "regression" },
  { key: "logistic_regression", label: "Logistic Regression", task: "classification" },
  { key: "random_forest_classifier", label: "Random Forest Classifier", task: "classification" },
  { key: "gradient_boosting_classifier", label: "Gradient Boosting Classifier", task: "classification" },
  { key: "xgboost_classifier", label: "XGBoost Classifier", task: "classification" },
  { key: "lightgbm_classifier", label: "LightGBM Classifier", task: "classification" },
];

export const MAX_UPLOAD_MB = 200;

export const SELECTABLE_ROLES = new Set(["numeric", "categorical", "boolean", "datetime"]);
