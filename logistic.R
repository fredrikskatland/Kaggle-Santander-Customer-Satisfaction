require(corrplot)
require(mlbench)
require(reshape2)
require(lattice)
require(e1071)
require(AppliedPredictiveModeling)
require(caret)
require(corrplot)
require(randomForest)
require(foreach)
require(doParallel)
registerDoParallel(cores=3)

library(perturb)
colldiag(m1)

setwd("./kaggle/Santander Customer Satisfaction/")
train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")
sampsub <- read.csv("./data/sample_submission.csv")
hist(train$TARGET)

target <- train$TARGET
train$training <- 1
test$training <- 0
data <- rbind(train[-371],test)
str(data)

# Removing zero variance variables
vars <- apply(data,2,var)
data2 <- data[,!names(data) %in% names(vars[vars==0])]

training <- data2[data2$training == 1, -1]
training$TARGET <- as.factor(target)
splits <- createDataPartition(training$TARGET, times=1, p=0.75)
training2 <- training[splits[[1]],]
testing2 <- training[-splits[[1]],]
model_glm <- glm(TARGET~., training2, family="binomial")

actual <- testing2$TARGET
pred <- predict(model_glm, testing2, type="terms")
