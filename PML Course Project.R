# Load packages
library(caret)
library(randomForest)
library(gbm)

# Read the data files
trainR <- read.csv("./pml-training.csv", na.strings=c("NA",""))
testR <- read.csv("./pml-testing.csv", na.strings=c("NA",""))

#Remove columns with all NAs
trainR <- trainR[, colSums(is.na(trainR)) == 0] 
testR <- testR[, colSums(is.na(testR)) == 0] 

# Remove columns with no predictive value
classe <- trainR$classe
Trainstamp <- grepl("^X|timestamp|window", names(trainR))
trainR <- trainR[, !Trainstamp]
trainC <- trainR[, sapply(trainR, is.numeric)]
trainC$classe <- classe
Teststamp <- grepl("^X|timestamp|window", names(testR))
testR <- testR[, !Teststamp]
testC <- testR[, sapply(testR, is.numeric)]

# Partition the dataset
set.seed(606)
inTrain <- createDataPartition(trainC$classe, p=0.70, list=F)
training <- trainC[inTrain, ]
testing <- trainC[-inTrain, ]

# Test three models on a small sample
smalltrain <- training[sample(nrow(training), 500), ]
smalltest <- testing[sample(nrow(testing), 150), ]

fitRf <- train(classe ~ ., method="rf", data=smalltrain)
fitGBM <- train(classe ~ ., method="gbm", data=smalltrain)
fitLDA <- train(classe ~ ., method="lda", data=smalltrain)
predRf <- predict(fitRf, smalltest)
predGBM <- predict(fitGBM, smalltest)
predLDA <- predict(fitLDA, smalltest)
confusionMatrix(predRf, smalltest$classe)
confusionMatrix(predGBM, smalltest$classe)
confusionMatrix(predLDA, smalltest$classe)

# Fit the Random Forest model with cross-validation and PCA pre-processing
trControl <- trainControl(method = "cv", number = 4)
ModelRf <- train(classe ~., preProcess="pca", method="rf", trControl = trControl, data=training)

# Test the model on the testing set
predictRf <- predict(ModelRf, testing)
confusionMatrix(testing$classe, predictRf)

# Apply the result to the cleaned testing set
answers <- predict(ModelRf, testC[, -length(names(testC))])
class(answers)