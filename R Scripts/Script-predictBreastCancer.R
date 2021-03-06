#################### Advanced Method(Principal Component Analysis) - Breast Cancer ############


# Problem Statement
# The dataset contains diagnosis data about breast cancer patients and whether they are Benign (healthy) or
# Malignant (possible disease). We need to predict whether new patients are benign or malignant based on
# model built on this data.



##################### Data Engineering & Analysis ##############################

setwd("E:/Mission Machine Learning/Git/predictBreastCancer")

cancer_data <- read.csv("Data/breast_cancer.csv")
str(cancer_data)

summary(cancer_data)

head(cancer_data)


# Exploratory Data Analysis


library(psych)

#given the large number of variables, split into 3 sets and see correlation to diagnosis.
pairs.panels(cancer_data[, c(2,3:10)])

pairs.panels(cancer_data[, c(2,11:20)])

pairs.panels(cancer_data[, c(2,21:32)])

# Principal Component Analysis - In this section, we first scale the data and discover the principal
# components of the data. Then we only pick the top components that have the heaviest influence on the
# target.

#scale the data first
scaled_data <- scale(cancer_data[, 3:32])
#convert into principal components
pca_data <- prcomp(scaled_data)

plot(pca_data)

summary(pca_data)


#Get only the first 3 components
final_data <- data.frame(pca_data$x[,1:3])
#add diagnosis to the data frame
final_data$diagnosis <- cancer_data$diagnosis
pairs.panels(final_data)

# The first 3 principal components influences 75% of the target, so we only pick the top 3. A correlation
# analysis shows that these 3 have very good correlation to the target. Also the 3 PCs dont have any correlation
# amongst them.



############################## Modeling & Prediction ###################################
library(caret)
inTrain <- createDataPartition(y=final_data$diagnosis ,p=0.7,list=FALSE)
training <- final_data[inTrain,]
testing <- final_data[-inTrain,]
dim(training);dim(testing)

table(training$diagnosis); table(testing$diagnosis)
# 
# Model Building and Testing We will build different models based on 4 different algorithms. Then we
# predict on the test data and measure accuracy. Finally, we compare the algorithms for their accuracy and speed.
# The “caret” package in R provides a convenient unified interface for using any of the algorithms for modeling
# and prediction. It has an extensive library of algorithms http://topepo.github.io/caret/modelList.html . This
# can be used to compare performance of different algorithms for a given dataset

predlist <- c("bagFDA", #Bagging
              "LogitBoost", #Boosting
              "nnet", #Neural Networks
              "svmRadialCost") #Support vector machines
#Create a result data set
results <- data.frame( Algorithm=character(), Duration=numeric(), Accuracy=numeric(),
                       stringsAsFactors=FALSE)
#loop through algorithm list and perform model building and prediction
for (i in 1:length(predlist)) {
  pred <- predlist[i]
  print(paste("Algorithm = ",pred ))
  #Measure Time
  startTime <- as.integer(Sys.time())
  #Build model
  model <- train( diagnosis ~ ., data=training, method=pred)
  #Predict
  10
  predicted <- predict(model, testing)
  #Compare results
  matrix<- confusionMatrix(predicted, testing$diagnosis)
  #Measure end time
  endTime <- as.integer(Sys.time())
  #Store result
  thisresult <- c( as.character(pred), endTime-startTime, as.numeric(matrix$overall[1]))
  results[i,1] <- pred
  results[i,2] <- endTime-startTime
  results[i,3] <- round(as.numeric(matrix$overall[1]) * 100, 2)
}


#Print results
results
