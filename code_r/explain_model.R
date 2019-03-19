library(iml)
library(DALEX)
library(viridis)

set.seed(123)

statworx_blue <- rgb(20, 55, 70, maxColorValue = 255)
light_blue <- "#0085AF"

statworx_blue <- rgb(234, 54, 91, maxColorValue = 255)
light_blue <- rgb(82, 133, 213, maxColorValue = 255)

# IML ---------------------------------------------------------------------

# XGBoost predict function
xgb_predict <- function(model, newdata) {
  xgb_newdata <- df_to_DMatrix(newdata)
  p <- predict(model, newdata = xgb_newdata)
  return(p)
}

# iml predictor
target_idx <- which(names(df_train) == target)
xgb_predictor <- iml::Predictor$new(mod, 
                                    data = df_train[, -target_idx], 
                                    y = df_train[, target],
                                    predict.fun = xgb_predict)


# Feature importance ------------------------------------------------------

# Feature importance
xgb_feature_imp <- FeatureImp$new(xgb_predictor, 
                                  loss = "logLoss",
                                  compare = "ratio")
  
# Plot
ggplot(data = xgb_feature_imp$results) +
  geom_errorbar(aes(ymin = importance.05, ymax = importance.95,
                    x = reorder(feature, importance)),
                #size = 0.1,
                width = 0.4,
                color = light_blue,
                alpha = 0.8) +
  geom_point(aes(y = importance, x = reorder(feature, importance)),
             size = 2, 
             color = "white") +
  geom_point(aes(y = importance, x = reorder(feature, importance)),
             size = 2,
             color = light_blue,
             alpha = 0.8) +
  theme_minimal() +
  xlab("") + ylab("Feature Importance (logLoss)") +
  coord_flip()

ggsave("~/Desktop/feature_imp.png", width = 19, height = 13, units = "cm")


# Feature effects ---------------------------------------------------------

# PDP
iml::FeatureEffect$new(xgb_predictor, feature = "age", method = "pdp") %>% 
  .$results %>% 
  ggplot(aes(x = age, y = .y.hat)) + 
  geom_line(color = light_blue,
            size = 1,
            lty = "solid") + 
  coord_cartesian(xlim = c(0, 150)) + 
  labs(x = "Age (in months)", y = "Prediction P(y=1)") + 
  theme_minimal() + 
  geom_point(aes(x = age, y = .fitted), 
             data = df_fitted, 
             color = statworx_blue,
             alpha = 0.01)

ggsave("~/Desktop/feature_pdp.png", width = 19, height = 13, units = "cm")

# ICE
iml::FeatureEffect$new(xgb_predictor, feature = "age", method = "ice") %>% 
  .$results %>% 
  dplyr::filter(.id %in% sample(.$.id, 1000)) %>% 
  ggplot() + 
  geom_line(aes(x = age, y = .y.hat, group = .id, color = .id),
            size = 1,
            alpha = 0.15) + 
  #scale_color_viridis(option = "A", guide = FALSE) + 
  scale_color_continuous(high = light_blue, low = "white", guide = FALSE) +
  coord_cartesian(xlim = c(0, 150)) + 
  labs(x = "Age (in months)", y = "Prediction P(y=1)") + 
  theme_minimal() + 
  geom_smooth(aes(x = age, y = .y.hat), 
              color = statworx_blue, 
              fill = "white",
              lty = "solid")

ggsave("~/Desktop/feature_ice.png", width = 19, height = 13, units = "cm")


# Surrogate model ---------------------------------------------------------

# Global surrogate model
iml::TreeSurrogate$new(xgb_predictor, maxdepth = 2) %>% 
  .$results %>% 
  ggplot(aes(y = .y.hat)) + 
  geom_boxplot(color = light_blue,
               alpha = 0.1,
               fill = light_blue) + 
  facet_wrap(~ .path) + 
  theme_minimal() + 
  labs(y = "Prediction P(y=1)") + 
  theme(axis.text.x = element_blank()) +
  theme(panel.grid.minor = element_blank())

ggsave("~/Desktop/global_surrogate.png", width = 19, height = 13, units = "cm")

# Average prediction
iml::TreeSurrogate$new(xgb_predictor, maxdepth = 2) %>% 
  .$results %>% 
  group_by(.path) %>% 
  summarise(median(.y.hat.tree))


# Tests -------------------------------------------------------------------

# LIME
df %>% 
  sample_n(1) %>% 
  iml::LocalModel$new(xgb_predictor, k = 10, x.interest = .) %>% 
  plot()

# Shapley
df %>% 
  as.data.frame() %>% 
  sample_n(1) %>% 
  iml::Shapley$new(xgb_predictor, x.interest = .) %>% 
  plot()



























