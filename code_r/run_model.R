library(data.table)
library(dplyr)
library(caret)
library(xgboost)
library(pROC)

set.seed(123)

# Function for converting data.frame to xgb.DMatrix
df_to_DMatrix <- function(input, label = NULL) {
  if (!is.null(label)) {
    input %>% 
      dplyr::select(-one_of(label)) %>% 
      model.matrix(~ -1 + ., data = .) %>% 
      xgb.DMatrix(data = ., label = input[, label])
  } else {
    input %>% 
      model.matrix(~ -1 + ., data = .) %>% 
      xgb.DMatrix(data = .)
  }
}

# Read data
df <- fread("data/prepared/train.csv", drop = 1, data.table = F)

# Create binary target
df <- df %>% 
  dplyr::mutate(adoptionspeed_bin = ifelse(adoptionspeed <= 2, 1, 0))

# Prepare data
df <- df %>% 
  dplyr::mutate(dog = ifelse(type == 1, 1, 0),
                mixed_breed = ifelse((breed1 != breed2) & (breed2 != 0), 0, 1),
                male = ifelse(gender == 1, 1, 0),
                female = ifelse(gender == 2, 1, 0),
                group = ifelse(gender == 3, 1, 0),
                color = factor(color1),
                vaccinated = factor(vaccinated),
                dewormed = factor(dewormed),
                sterilized = factor(sterilized)
                )

features <- c("dog", 
              "age",
              "breed1", 
              "breed2",
              "mixed_breed",
              "male",
              "female",
              "group",
              "color",
              "maturitysize",
              "furlength", 
              "vaccinated",
              "dewormed",
              "sterilized",
              "health",
              "fee",
              "state",
              "photoamt",
              "description_length",
              "sentiment_magnitude",
              "sentiment_score")

target <- "adoptionspeed_bin"

# Train/test split
in_train <- createDataPartition(y = df[, target], p = 0.7, list = FALSE)

df_train <- df %>% 
  dplyr::select(one_of(target, features)) %>% 
  dplyr::filter(row_number() %in% in_train)
  
xgb_train  <- df_train %>% 
  df_to_DMatrix(label = target)

df_test <- df %>% 
  dplyr::select(one_of(target, features)) %>% 
  dplyr::filter(!(row_number() %in% in_train))
  
xgb_test <- df_test %>% 
  df_to_DMatrix(label = target)

# Tuning

grid <- expand.grid(nround = c(50, 75, 100, 125, 150),
                    eta = 0.3,
                    gamma = 1,
                    max_depth = c(3, 6, 9),
                    subsample = 1,
                    colsample_by_tree = 1,
                    lambda = c(0, 0.01),
                    alpha = c(0, 0.01))

results <- data.frame(id = seq(nrow(grid)),
                      score = NA)

for (i in seq(nrow(grid))) {
  
  print(paste0("Parmeter ", i, "/", nrow(results)))
  
  sel_params <- grid[i, ]
  params <- list(eta = sel_params$eta,
                 gamma = sel_params$gamma,
                 max_depth = sel_params$max_depth,
                 subsample = sel_params$subsample,
                 colsample_bytree = sel_params$colsample_by_tree,
                 lambda = sel_params$lambda,
                 alpha = sel_params$alpha)
  
  mod_cv <- xgb.cv(params = params,
                   data = xgb_train,
                   nfold = 5,
                   nrounds = sel_params$nround,
                   booster = "gbtree",
                   objective = "binary:logistic",
                   eval_metric = "error",
                   verbose = 0)
  
  results$score[results$id == i] <- mod_cv$evaluation_log$test_error_mean[sel_params$nround]
}

# Get best params
sel_params <- grid[which.min(results$score), ]
best_params <- list(eta = sel_params$eta,
                    gamma = sel_params$gamma,
                    max_depth = sel_params$max_depth,
                    subsample = sel_params$subsample,
                    colsample_bytree = sel_params$colsample_by_tree,
                    lambda = sel_params$lambda,
                    alpha = sel_params$alpha)

# Run final model
mod <- xgb.train(params = params,
                 data = xgb_train,
                 nrounds = sel_params$nround,
                 booster = "gbtree",
                 objective = "binary:logistic",
                 eval_metric = "error")

# Importance in training
xgb.importance(model = mod, feature_names = features)

# Plot ensemble
xgb.plot.multi.trees(mod, feature_names = features)

# Fitted
df_fitted <- df_train %>% 
  mutate(.fitted = predict(mod, xgb_train))

# Prediction
p <- predict(mod, newdata = xgb_test)

# AUC score on test
roc_obj <- pROC::roc(response = df[-in_train, target], predictor = p)
pROC::auc(roc_obj)







