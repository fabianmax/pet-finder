library(data.table)
library(dplyr)
library(caret)
library(xgboost)

df <- fread("data/prepared/train.csv", drop = 1, data.table = F)

df <- df %>% 
  mutate(within_first_month = ifelse(adoptionspeed <= 2, 1, 0))

df_to_DMatrix <- function(data, label = NULL) {
  xgb.DMatrix(as.matrix(data), label = label)
}
  
features <- c("type", 
              "age",
              "breed1", 
              "breed2",
              "gender",
              "color1",
              "color2",
              "color3",
              "maturitysize",
              "furlength", 
              "vaccinated",
              "dewormed",
              "health",
              "fee",
              "state",
              "photoamt",
              "description_length",
              "sentiment_magnitude",
              "sentiment_score")

target <- "within_first_month"

in_train <- createDataPartition(y = df[, target], p = 0.7, list = FALSE)
target_idx <- which(colnames(df) == target)

xgb_train <- xgb.DMatrix(data = as.matrix(df[in_train, features]), label = df[in_train, target])
xgb_test <- xgb.DMatrix(data = as.matrix(df[-in_train, features]), label = df[-in_train, target])

mod_cv <- xgb.cv(params = list(eta = 0.3,
                               gamma = 0,
                               max_depth = 3,
                               subsample = 1,
                               colsample_bytree = 1,
                               lambda = 0.01,
                               alpha = 0.01),
                 data = xgb_train,
                 nfold = 3,
                 nrounds = 100,
                 booster = "gbtree",
                 objective = "binary:logistic",
                 eval_metric = "error")


mod <- xgb.train(params = list(eta = 0.3,
                            gamma = 0,
                            max_depth = 3,
                            subsample = 1,
                            colsample_bytree = 1,
                            lambda = 0.01,
                            alpha = 0.01),
              data = xgb_train,
              nrounds = 100,
              booster = "gbtree",
              objective = "binary:logistic",
              eval_metric = "error")

xgb.importance(model = mod, feature_names = features)







