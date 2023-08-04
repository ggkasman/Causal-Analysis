#EMPLOYMENT ANALYSES

#Necessary Libraries
library(dplyr)       # Data manipulation (0.8.0.1)
library(fBasics)     # Summary statistics (3042.89)
library(corrplot)    # Correlations (0.84)
library(psych)       # Correlation p-values (1.8.12)
library(grf)         # Generalized random forests (0.10.2)
library(remotes)     # Install packages from github (2.0.1)
library(readr)       # Reading csv files (1.3.1)
library(tidyr)       # Database operations (0.8.3)
library(tibble)      # Modern alternative to data frames (2.1.1)
library(knitr)       # RMarkdown (1.21)
library(kableExtra)  # Prettier RMarkdown (1.0.1)
library(ggplot2)     # general plotting tool (3.1.0)
library(aod)         # hypothesis testing (1.3.1)
library(estimatr)    # simple interface for OLS estimation w/ robust std errors ()

## Define and set seed
seed = 1234
set.seed(seed)

## Set to path where data are stored
setwd("/Users/gamzekasman/Library/CloudStorage/GoogleDrive-gamzegkasman@gmail.com/Drive'ım/LMU- Master Summer 23/ML in Metrics/Project/Data")
load("genderinequality.Rdata")

# Subset data for year 2010
data_2010 <- subset(dat, year == 2010)

# Combine all names except wage, year, id)
all_variables_names <- setdiff(names(data_2010), c("wage", "year", "id"))
df <- data_2010 %>% select(all_variables_names)

# Converting all columns to numerical
df <- data.frame(lapply(df, function(x) as.numeric(as.character(x))))

# Drop rows containing missing values
df <- na.omit(df)

# Make a data.frame containing summary statistics of interest
summ_stats <- fBasics::basicStats(df)
summ_stats <- as.data.frame(t(summ_stats))

# Rename some of the columns for convenience
summ_stats <- summ_stats[c("Mean", "Stdev", "Minimum", "1. Quartile", "Median",  "3. Quartile", "Maximum")] %>% 
  rename("Lower quartile" = '1. Quartile', "Upper quartile"= "3. Quartile")

summ_stats


# Rename variables
df <- df %>% rename(Y=emp,W=treat)

#Test-Train Split
train_fraction <- 0.80  # Use train_fraction % of the dataset to train our models
n <- dim(df)[1]
train_idx <- sample.int(n, replace=F, size=floor(n*train_fraction))
df_train <- df[train_idx,]
df_test <- df[-train_idx,]

# Causal Forests and the R-Learner

# Step 1: Fit the forest

# List of variables from df_train to be used as input features (covariates)
covariate_names <- c("female", "IQ", "KWW", "educ", "exper", "tenure", "age",
                     "married", "black", "south", "urban", "sibs", "brthord", "meduc",
                     "feduc")


# Fitting the causal forest model
cf <- causal_forest(
  X = as.matrix(df_train[, covariate_names]),
  Y = df_train$Y,
  W = df_train$W,
  num.trees = 500)

# Step 2: Examine the nuisance parameters

ggplot(data.frame(W.hat = cf$W.hat, W = factor(cf$W.orig))) +
  geom_histogram(aes(x = W.hat, y = stat(density), fill = W), alpha=0.3, position = "identity") +
  geom_density(aes(x = W.hat, color = W)) +
  xlim(0,1) +
  labs(title = "Causal forest propensity scores",
       caption = "The propensity scores are learned via GRF's regression forest")

#For the propensity model
DF <- data.frame(
  W          = df_train$W,
  e.bar      = mean(cf$W.hat),
  e.residual = cf$W.hat - mean(cf$W.hat)
)

best.linear.predictor <- lm(W ~ e.bar + e.residual + 0, data = DF)
blp.summary <- lmtest::coeftest(best.linear.predictor,
                                vcov = sandwich::vcovCL,
                                type = "HC3")

#For the outcome model
DF <- data.frame(
  Y          = df_train$Y,
  m.bar      = mean(cf$Y.hat),
  m.residual = cf$Y.hat - mean(cf$Y.hat)
)

best.linear.predictor <- lm(Y ~ m.bar + m.residual + 0, data = DF)
blp.summary <- lmtest::coeftest(best.linear.predictor,
                                vcov = sandwich::vcovCL,
                                type = "HC3")

# convert to one-sided p-values
dimnames(blp.summary)[[2]][4] <- gsub("[|]", "", dimnames(blp.summary)[[2]][4])
blp.summary[, 4] <- ifelse(blp.summary[, 3] < 0, 1 - blp.summary[, 4] / 2, blp.summary[, 4] / 2)



#Step 3(a): Predict point estimates and standard errors (training set, out-of-bag)

oob_pred <- predict(cf, estimate.variance=TRUE)

head(oob_pred, 5)

oob_tauhat_cf <- oob_pred$predictions
oob_tauhat_cf_se <- sqrt(oob_pred$variance.estimates)

#Step 3(b): Predict point estimates and standard errors (test set)

#To predict on a test set, pass it using the newdata argument.
test_pred <- predict(cf, newdata=as.matrix(df_test[covariate_names]), estimate.variance=TRUE)
tauhat_cf_test <- test_pred$predictions
tauhat_cf_test_se <- sqrt(test_pred$variance.estimates)

head(test_pred, 5)
head(oob_tauhat_cf, 100)

#Assessing heterogeneity
hist(oob_tauhat_cf, main="Causal forests: out-of-bag CATE")

#Variable Importance
var_imp <- c(variable_importance(cf))
names(var_imp) <- covariate_names
sorted_var_imp <- sort(var_imp, decreasing=TRUE)
sorted_var_imp

#Heterogeneity across subgroups
# Manually creating subgroups
num_tiles <- 4  # ntiles = CATE is above / below the median
df_train$cate <- oob_tauhat_cf
df_train$ntile <- factor(ntile(oob_tauhat_cf, n=num_tiles))

#Conditional Average treatment effects within subgroups
#Sample Conditional Average Treatment Effect:

ols_sample_ate <- lm_robust(Y ~ ntile + ntile:W, data=df_train)
estimated_sample_ate <- coef(summary(ols_sample_ate))[(num_tiles+1):(2*num_tiles), c("Estimate", "Std. Error")]
hypothesis_sample_ate <- paste0("ntile1:W = ", paste0("ntile", seq(2, num_tiles), ":W"))
ftest_pvalue_sample_ate <- linearHypothesis(ols_sample_ate, hypothesis_sample_ate)[2,"Pr(>F)"]

#AIPW
estimated_aipw_ate <- lapply(
  seq(num_tiles), function(w) {
    ate <- average_treatment_effect(cf, subset = df_train$ntile == w)
  })
estimated_aipw_ate <- data.frame(do.call(rbind, estimated_aipw_ate))

# Testing for equality using Wald test
## define L matrix that allows us to test if the ATEs in ntile 2+ are equal to ntile 1
.L <- cbind(-1, diag(num_tiles - 1))
# e.g. [,1] [,2] [,3] [,4]
# [1,]   -1    1    0    0
# [2,]   -1    0    1    0
# [3,]   -1    0    0    1
waldtest_pvalue_aipw_ate <- wald.test(Sigma = diag(estimated_aipw_ate$std.err^2),
                                      b = estimated_aipw_ate$estimate,
                                      L = .L)$result$chi2[3]

#Visualization of AIPW
# Create a point plot with error bars
plot_ate_point <- ggplot(estimated_aipw_ate, aes(x = factor(1:num_tiles), y = estimate)) +
  geom_point(color = "steelblue", size = 4) +
  geom_errorbar(aes(ymin = estimate - std.err, ymax = estimate + std.err), width = 0.2, color = "black") +
  labs(x = "Ntile", y = "Estimated ATE") +
  scale_x_discrete(labels = paste("Ntile", 1:num_tiles)) +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white"),   # Set the background color to white
        axis.line = element_line(color = "black"),        # Set axis line color to orange
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save the point plot to a quartz device with white background and orange lines
quartz("point_plot_ate", width = 8, height = 5)
print(plot_ate_point)


#conditional average treatment effect between all two pairs of n-tiles ###p_values_tile_by_tile ###
p_values_tile_by_tile <- matrix(nrow = num_tiles, ncol = num_tiles)
differences_tile_by_tile <- matrix(nrow = num_tiles, ncol = num_tiles)
stderror_tile_by_tile <- matrix(nrow = num_tiles, ncol = num_tiles)
hypotheses_grid <- combn(1:num_tiles, 2)

invisible(apply(hypotheses_grid, 2, function(x) {
  .diff <- with(estimated_aipw_ate, estimate[ntile = x[2]] - estimate[ntile = x[1]])
  .se <- with(estimated_aipw_ate, sqrt(std.err[ntile = x[2]]^2 + std.err[Ntile = x[1]]^2))
  
  differences_tile_by_tile[x[2], x[1]] <<- .diff
  stderror_tile_by_tile[x[2], x[1]] <<- .se
  p_values_tile_by_tile[x[2], x[1]] <<- 1 - pnorm(abs(.diff/.se)) + pnorm(-abs(.diff/.se))
}))

#Heterogeneity across covariates
# Define the names of the covariates
covariate_names <- c("female", "IQ", "KWW", "educ", "exper", "tenure", "age",
                     "married", "black", "south", "urban", "sibs", "brthord", "meduc",
                     "feduc")
# Regress each covariate on ntile assignment to means p
cov_means <- lapply(covariate_names, function(covariate) {
  lm_robust(as.formula(paste0(covariate, ' ~ 0 + ntile')), data = df_train)
})

# Extract the mean and standard deviation of each covariate per ntile
cov_table <- lapply(cov_means, function(cov_mean) {
  as.data.frame(t(coef(summary(cov_mean))[,c("Estimate", "Std. Error")]))
})


#Covariate variation across n-tiles
covariate_means_per_ntile <- df_train %>% select(covariate_names,ntile) %>% group_by(ntile) %>%
  summarise_all(mean) %>% select(covariate_names) 
covariate_means <- df_train %>% select(covariate_names) %>% summarise_all(mean)
ntile_weights <- table(df_train$ntile) / dim(df_train)[1] 
deviations <- covariate_means_per_ntile %>% rowwise() %>% do( . - (covariate_means)) %>% ungroup()
covariate_means_weighted_var <- deviations %>% mutate_all( function(x){sum(ntile_weights * x^2)}) %>% summarise_all(mean)
covariate_var <- df_train %>% select(covariate_names) %>% summarise_all(var)
cov_variation <- covariate_means_weighted_var / covariate_var



#CATE

#Overall
# Calculate the Average Treatment Effect (ATE) on the training set
ate_training <- mean(oob_tauhat_cf)
ate_training_se <- sqrt(mean(oob_tauhat_cf_se^2))
#pvalue
# Degrees of freedom for the t-distribution (number of samples - 1)
dof <- nrow(df_train) - 1
# Calculate the t-statistic for the ATE
t_stat_ate <- abs(ate_training) / (ate_training_se)
# Calculate the p-value for the ATE (two-tailed test)
p_value_ate <- 2 * (1 - pt(t_stat_ate, dof))


# Calculate the Average Treatment Effect (ATE) on the test set
ate_test <- mean(tauhat_cf_test)
ate_test_se <- sqrt(mean(tauhat_cf_test_se^2))
#pvalue
# Degrees of freedom for the t-distribution (number of samples - 1)
dof_test <- nrow(df_test) - 1
# Calculate the t-statistic for the ATE on the test set
t_stat_ate_test <- abs(ate_test) / (ate_test_se)
# Calculate the p-value for the ATE on the test set (two-tailed test)
p_value_ate_test <- 2 * (1 - pt(t_stat_ate_test, dof_test))


#Female

# For the training set
# Filter the data to include only females
df_females_train <- df_train[df_train$female == 1, ]
# Predict point estimates and standard errors for females using the causal forest model (training set)
training_pred_female <- predict(cf, newdata = as.matrix(df_females_train[covariate_names]), estimate.variance = TRUE)
# Extract the point estimates (treatment effect) for females (training set)
tauhat_cf_training_female <- training_pred_female$predictions
# Extract the standard errors of the point estimates for females (training set) and calculate their square root
tauhat_cf_training_se_female <- sqrt(training_pred_female$variance.estimates)
# Calculate the Average Treatment Effect (ATE) on the training set
cate_f_training <- mean(tauhat_cf_training_female)
cate_f_training_se <- sqrt(mean(tauhat_cf_training_se_female^2))
#pvalue
# Calculate the t-statistic for the CATE of females
t_stat_cate_females <- abs(cate_f_training) / (cate_f_training_se)
# Calculate the p-value for the CATE of females (two-tailed test)
p_value_cate_females <- 2 * (1 - pt(t_stat_cate_females, dof))


# For the test set
# Filter the data to include only females
df_females_test <- df_test[df_test$female == 1, ]
# Predict point estimates and standard errors for females using the causal forest model (test set)
test_pred_female <- predict(cf, newdata = as.matrix(df_females_test[covariate_names]), estimate.variance = TRUE)
# Extract the point estimates (treatment effect) for females (test set)
tauhat_cf_test_female <- test_pred_female$predictions
# Extract the standard errors of the point estimates for females (test set) and calculate their square root
tauhat_cf_test_se_female <- sqrt(test_pred_female$variance.estimates)
# Calculate the Average Treatment Effect (ATE) on the training set
cate_f_test <- mean(tauhat_cf_test_female)
cate_females_test_se <- sqrt(mean(tauhat_cf_test_se_female^2))
# Degrees of freedom for the t-distribution (number of samples - 1)
dof_test <- nrow(df_test) - 1
# Calculate the t-statistic for the CATE of females on the test set
t_stat_cate_females_test <- abs(cate_f_test) / (cate_females_test_se)
# Calculate the p-value for the CATE of females on the test set (two-tailed test)
p_value_cate_females_test <- 2 * (1 - pt(t_stat_cate_females_test, dof_test))

