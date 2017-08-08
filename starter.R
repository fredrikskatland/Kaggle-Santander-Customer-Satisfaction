require(corrplot)
require(mlbench)
require(reshape2)
require(lattice)
require(e1071)
require(AppliedPredictiveModeling)
require(caret)
require(corrplot)
require(randomForest)

setwd("./kaggle/Santander Customer Satisfaction/")
train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")
sampsub <- read.csv("./data/sample_submission.csv")
hist(train$TARGET)
table(train$TARGET)
prop.table(table(train$TARGET))
# Stratified sample required.

target <- train$TARGET
train$training <- 1
test$training <- 0
data <- rbind(train[-371],test)
str(data)

x<-seq(1,371, by=80)
str(data[x[1]:x[2]])
str(data[x[2]:x[3]])
str(data[x[3]:x[4]])
str(data[x[4]:x[5]])

# Removing highly correlated variables.
vars <- apply(data, 2, FUN=var)
zero_var <- names(vars[vars==0])
str(data[,names(data) %in% zero_var])
data2 <- data[,!names(data) %in% zero_var]

corr_matrix <- cor(data2,use="pairwise.complete.obs")
highCorr <- findCorrelation(corr_matrix, cutoff = 0.9)
length(highCorr)

corrplot(corr_matrix[1:150,1:150])

training <- data[data$training==1,]
training$target <- target
system.time(
  model_rf <- randomForest(as.factor(target)~., data = training[,-1], ntree=500, mtry=50)
)

model_rf$importance
