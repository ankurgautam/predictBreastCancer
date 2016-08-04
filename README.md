# Advanced Methods - Breast Cancer

## Problem Statement
The dataset contains diagnosis data about breast cancer patients and whether they are Benign (healthy) or
Malignant (possible disease). We need to predict whether new patients are benign or malignant based on model built on this data.

##Techniques Used
1. Principal Component Analysis
2. Training and Testing
3. Confusion Matrix
4. Neural Networks
5. Support Vector Machines
6. Bagging
7. Boosting

##Data Engineering & Analysis


###Loading and understanding the dataset
```{r}
setwd("C:/Modules/Machine Learning Algorithms/Advanced Methods")

cancer_data <- read.csv("breast_cancer.csv")

str(cancer_data)
```

```
## 'data.frame': 569 obs. of 32 variables:
## $ id : int 87139402 8910251 905520 868871 9012568 906539 925291 87880 862989 89827 ...
## $ diagnosis : Factor w/ 2 levels "B","M": 1 1 1 1 1 1 1 2 1 1 ...
## $ radius_mean : num 12.3 10.6 11 11.3 15.2 ...
## $ texture_mean : num 12.4 18.9 16.8 13.4 13.2 ...
## $ perimeter_mean : num 78.8 69.3 70.9 73 97.7 ...
## $ area_mean : num 464 346 373 385 712 ...
## $ smoothness_mean : num 0.1028 0.0969 0.1077 0.1164 0.0796 ...
## $ compactness_mean : num 0.0698 0.1147 0.078 0.1136 0.0693 ...
## $ concavity_mean : num 0.0399 0.0639 0.0305 0.0464 0.0339 ...
## $ points_mean : num 0.037 0.0264 0.0248 0.048 0.0266 ...
## $ symmetry_mean : num 0.196 0.192 0.171 0.177 0.172 ...
## $ dimension_mean : num 0.0595 0.0649 0.0634 0.0607 0.0554 ...
## $ radius_se : num 0.236 0.451 0.197 0.338 0.178 ...
## $ texture_se : num 0.666 1.197 1.387 1.343 0.412 ...
## $ perimeter_se : num 1.67 3.43 1.34 1.85 1.34 ...
## $ area_se : num 17.4 27.1 13.5 26.3 17.7 ...
## $ smoothness_se : num 0.00805 0.00747 0.00516 0.01127 0.00501 ...
## $ compactness_se : num 0.0118 0.03581 0.00936 0.03498 0.01485 ...
## $ concavity_se : num 0.0168 0.0335 0.0106 0.0219 0.0155 ...
## $ points_se : num 0.01241 0.01365 0.00748 0.01965 0.00915 ...
## $ symmetry_se : num 0.0192 0.035 0.0172 0.0158 0.0165 ...
## $ dimension_se : num 0.00225 0.00332 0.0022 0.00344 0.00177 ...
## $ radius_worst : num 13.5 11.9 12.4 11.9 16.2 ...
## $ texture_worst : num 15.6 22.9 26.4 15.8 15.7 ...
## $ perimeter_worst : num 87 78.3 79.9 76.5 104.5 ...
## $ area_worst : num 549 425 471 434 819 ...
## $ smoothness_worst : num 0.139 0.121 0.137 0.137 0.113 ...
## $ compactness_worst: num 0.127 0.252 0.148 0.182 0.174 ...
## $ concavity_worst : num 0.1242 0.1916 0.1067 0.0867 0.1362 ...
## $ points_worst : num 0.0939 0.0793 0.0743 0.0861 0.0818 ...
## $ symmetry_worst : num 0.283 0.294 0.3 0.21 0.249 ...
## $ dimension_worst : num 0.0677 0.0759 0.0788 0.0678 0.0677 ...
```
```{r}
summary(cancer_data)
```
```
## id diagnosis radius_mean texture_mean
## Min. :8.67e+03 B:357 Min. : 6.98 Min. : 9.71
## 1st Qu.:8.69e+05 M:212 1st Qu.:11.70 1st Qu.:16.17
## Median :9.06e+05 Median :13.37 Median :18.84
## Mean :3.04e+07 Mean :14.13 Mean :19.29
## 3rd Qu.:8.81e+06 3rd Qu.:15.78 3rd Qu.:21.80
## Max. :9.11e+08 Max. :28.11 Max. :39.28
## perimeter_mean area_mean smoothness_mean compactness_mean
## Min. : 43.8 Min. : 144 Min. :0.0526 Min. :0.0194
## 1st Qu.: 75.2 1st Qu.: 420 1st Qu.:0.0864 1st Qu.:0.0649
## Median : 86.2 Median : 551 Median :0.0959 Median :0.0926
## Mean : 92.0 Mean : 655 Mean :0.0964 Mean :0.1043
## 3rd Qu.:104.1 3rd Qu.: 783 3rd Qu.:0.1053 3rd Qu.:0.1304
## Max. :188.5 Max. :2501 Max. :0.1634 Max. :0.3454
## concavity_mean points_mean symmetry_mean dimension_mean
## Min. :0.0000 Min. :0.0000 Min. :0.106 Min. :0.0500
## 1st Qu.:0.0296 1st Qu.:0.0203 1st Qu.:0.162 1st Qu.:0.0577
## Median :0.0615 Median :0.0335 Median :0.179 Median :0.0615
2
## Mean :0.0888 Mean :0.0489 Mean :0.181 Mean :0.0628
## 3rd Qu.:0.1307 3rd Qu.:0.0740 3rd Qu.:0.196 3rd Qu.:0.0661
## Max. :0.4268 Max. :0.2012 Max. :0.304 Max. :0.0974
## radius_se texture_se perimeter_se area_se
## Min. :0.112 Min. :0.360 Min. : 0.757 Min. : 6.8
## 1st Qu.:0.232 1st Qu.:0.834 1st Qu.: 1.606 1st Qu.: 17.9
## Median :0.324 Median :1.108 Median : 2.287 Median : 24.5
## Mean :0.405 Mean :1.217 Mean : 2.866 Mean : 40.3
## 3rd Qu.:0.479 3rd Qu.:1.474 3rd Qu.: 3.357 3rd Qu.: 45.2
## Max. :2.873 Max. :4.885 Max. :21.980 Max. :542.2
## smoothness_se compactness_se concavity_se points_se
## Min. :0.00171 Min. :0.00225 Min. :0.0000 Min. :0.00000
## 1st Qu.:0.00517 1st Qu.:0.01308 1st Qu.:0.0151 1st Qu.:0.00764
## Median :0.00638 Median :0.02045 Median :0.0259 Median :0.01093
## Mean :0.00704 Mean :0.02548 Mean :0.0319 Mean :0.01180
## 3rd Qu.:0.00815 3rd Qu.:0.03245 3rd Qu.:0.0420 3rd Qu.:0.01471
## Max. :0.03113 Max. :0.13540 Max. :0.3960 Max. :0.05279
## symmetry_se dimension_se radius_worst texture_worst
## Min. :0.00788 Min. :0.000895 Min. : 7.93 Min. :12.0
## 1st Qu.:0.01516 1st Qu.:0.002248 1st Qu.:13.01 1st Qu.:21.1
## Median :0.01873 Median :0.003187 Median :14.97 Median :25.4
## Mean :0.02054 Mean :0.003795 Mean :16.27 Mean :25.7
## 3rd Qu.:0.02348 3rd Qu.:0.004558 3rd Qu.:18.79 3rd Qu.:29.7
## Max. :0.07895 Max. :0.029840 Max. :36.04 Max. :49.5
## perimeter_worst area_worst smoothness_worst compactness_worst
## Min. : 50.4 Min. : 185 Min. :0.0712 Min. :0.0273
## 1st Qu.: 84.1 1st Qu.: 515 1st Qu.:0.1166 1st Qu.:0.1472
## Median : 97.7 Median : 686 Median :0.1313 Median :0.2119
## Mean :107.3 Mean : 881 Mean :0.1324 Mean :0.2543
## 3rd Qu.:125.4 3rd Qu.:1084 3rd Qu.:0.1460 3rd Qu.:0.3391
## Max. :251.2 Max. :4254 Max. :0.2226 Max. :1.0580
## concavity_worst points_worst symmetry_worst dimension_worst
## Min. :0.000 Min. :0.0000 Min. :0.156 Min. :0.0550
## 1st Qu.:0.114 1st Qu.:0.0649 1st Qu.:0.250 1st Qu.:0.0715
## Median :0.227 Median :0.0999 Median :0.282 Median :0.0800
## Mean :0.272 Mean :0.1146 Mean :0.290 Mean :0.0839
## 3rd Qu.:0.383 3rd Qu.:0.1614 3rd Qu.:0.318 3rd Qu.:0.0921
## Max. :1.252 Max. :0.2910 Max. :0.664 Max. :0.2075
```
```{r}
head(cancer_data)
```
```
## id diagnosis radius_mean texture_mean perimeter_mean area_mean
## 1 87139402 B 12.32 12.39 78.85 464.1
## 2 8910251 B 10.60 18.95 69.28 346.4
## 3 905520 B 11.04 16.83 70.92 373.2
## 4 868871 B 11.28 13.39 73.00 384.8
## 5 9012568 B 15.19 13.21 97.65 711.8
## 6 906539 B 11.57 19.04 74.20 409.7
## smoothness_mean compactness_mean concavity_mean points_mean
## 1 0.10280 0.06981 0.03987 0.03700
## 2 0.09688 0.11470 0.06387 0.02642
## 3 0.10770 0.07804 0.03046 0.02480
## 4 0.11640 0.11360 0.04635 0.04796
## 5 0.07963 0.06934 0.03393 0.02657
3
## 6 0.08546 0.07722 0.05485 0.01428
## symmetry_mean dimension_mean radius_se texture_se perimeter_se area_se
## 1 0.1959 0.05955 0.2360 0.6656 1.670 17.43
## 2 0.1922 0.06491 0.4505 1.1970 3.430 27.10
## 3 0.1714 0.06340 0.1967 1.3870 1.342 13.54
## 4 0.1771 0.06072 0.3384 1.3430 1.851 26.33
## 5 0.1721 0.05544 0.1783 0.4125 1.338 17.72
## 6 0.2031 0.06267 0.2864 1.4400 2.206 20.30
## smoothness_se compactness_se concavity_se points_se symmetry_se
## 1 0.008045 0.011800 0.01683 0.012410 0.01924
## 2 0.007470 0.035810 0.03354 0.013650 0.03504
## 3 0.005158 0.009355 0.01056 0.007483 0.01718
## 4 0.011270 0.034980 0.02187 0.019650 0.01580
## 5 0.005012 0.014850 0.01551 0.009155 0.01647
## 6 0.007278 0.020470 0.04447 0.008799 0.01868
## dimension_se radius_worst texture_worst perimeter_worst area_worst
## 1 0.002248 13.50 15.64 86.97 549.1
## 2 0.003318 11.88 22.94 78.28 424.8
## 3 0.002198 12.41 26.44 79.93 471.4
## 4 0.003442 11.92 15.77 76.53 434.0
## 5 0.001767 16.20 15.73 104.50 819.1
## 6 0.003339 13.07 26.98 86.43 520.5
## smoothness_worst compactness_worst concavity_worst points_worst
## 1 0.1385 0.1266 0.12420 0.09391
## 2 0.1213 0.2515 0.19160 0.07926
## 3 0.1369 0.1482 0.10670 0.07431
## 4 0.1367 0.1822 0.08669 0.08611
## 5 0.1126 0.1737 0.13620 0.08178
## 6 0.1249 0.1937 0.25600 0.06664
## symmetry_worst dimension_worst
## 1 0.2827 0.06771
## 2 0.2940 0.07587
## 3 0.2998 0.07881
## 4 0.2102 0.06784
## 5 0.2487 0.06766
## 6 0.3035 0.08284
```

###Exploratory Data Analysis 


```{r}
library(psych)

#given the large number of variables, split into 3 sets and see correlation to diagnosis.
pairs.panels(cancer_data[, c(2,3:10)])
```

Pearson Correlation: 
![alt text](https://github.com/ankurgautam/predictBreastCancer/blob/master/Viz/pairspanel1.png "Correlations")

```{r}
pairs.panels(cancer_data[, c(2,11:20)])

```

Pearson Correlation: 
![alt text](https://github.com/ankurgautam/predictBreastCancer/blob/master/Viz/pairspanel2.png "Correlations")

```{r}
pairs.panels(cancer_data[, c(2,21:32)])
```

Pearson Correlation: 
![alt text](https://github.com/ankurgautam/predictBreastCancer/blob/master/Viz/pairspanel3.png "Correlations")

Based on the correlation co-efficients, let us eliminate default, balance, day, month, campaign, poutcome
because of very low correlation. There are others too with very low correlation, but let us keep it for example
sake.

Principal Component Analysis - In this section, we first scale the data and discover the principal
components of the data. Then we only pick the top components that have the heaviest influence on the
target.

```{r}
#scale the data first
scaled_data <- scale(cancer_data[, 3:32])

#convert into principal components
pca_data <- prcomp(scaled_data)

plot(pca_data)
```
PCA Plot: 
![alt text](https://github.com/ankurgautam/predictBreastCancer/blob/master/Viz/pcaPlot.png "PCA Plot")

```{r}
summary(pca_data)
```

```
## Importance of components:
## PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8
## Standard deviation 3.644 2.386 1.6787 1.407 1.284 1.0988 0.8217 0.6904
## Proportion of Variance 0.443 0.190 0.0939 0.066 0.055 0.0403 0.0225 0.0159
## Cumulative Proportion 0.443 0.632 0.7264 0.792 0.847 0.8876 0.9101 0.9260
## PC9 PC10 PC11 PC12 PC13 PC14
## Standard deviation 0.6457 0.5922 0.5421 0.51104 0.49128 0.39624
## Proportion of Variance 0.0139 0.0117 0.0098 0.00871 0.00805 0.00523
## Cumulative Proportion 0.9399 0.9516 0.9614 0.97007 0.97812 0.98335
## PC15 PC16 PC17 PC18 PC19 PC20
## Standard deviation 0.30681 0.28260 0.24372 0.22939 0.22244 0.17652
## Proportion of Variance 0.00314 0.00266 0.00198 0.00175 0.00165 0.00104
## Cumulative Proportion 0.98649 0.98915 0.99113 0.99288 0.99453 0.99557
## PC21 PC22 PC23 PC24 PC25 PC26
## Standard deviation 0.173 0.16565 0.15602 0.1344 0.12442 0.09043
## Proportion of Variance 0.001 0.00091 0.00081 0.0006 0.00052 0.00027
## Cumulative Proportion 0.997 0.99749 0.99830 0.9989 0.99942 0.99969
## PC27 PC28 PC29 PC30
## Standard deviation 0.08307 0.03987 0.02736 0.0115
## Proportion of Variance 0.00023 0.00005 0.00002 0.0000
## Cumulative Proportion 0.99992 0.99997 1.00000 1.0000
```

```{r}
#Get only the first 3 components
final_data <- data.frame(pca_data$x[,1:3])

#add diagnosis to the data frame
final_data$diagnosis <- cancer_data$diagnosis

pairs.panels(final_data)
```
Pearson Correlation: 
![alt text](https://github.com/ankurgautam/predictBreastCancer/blob/master/Viz/pairpanel4.png "Correlations")

The first 3 principal components influences 75% of the target, so we only pick the top 3. A correlation
analysis shows that these 3 have very good correlation to the target. Also the 3 PCs dont have any correlation
amongst them.


##Modeling & Prediction

Split Training and Testing - Split training and testing datasets in the ratio of 70-30
```{r}
library(caret)

inTrain <- createDataPartition(y=final_data$diagnosis ,p=0.7,list=FALSE)
training <- final_data[inTrain,]
testing <- final_data[-inTrain,]
dim(training);dim(testing)

table(training$diagnosis); table(testing$diagnosis)
```
Model Building and Testing - We will build different models based on 4 different algorithms. Then we
predict on the test data and measure accuracy. Finally, we compare the algorithms for their accuracy and speed.
The “caret” package in R provides a convenient unified interface for using any of the algorithms for modeling
and prediction. It has an extensive library of algorithms http://topepo.github.io/caret/modelList.html . This
can be used to compare performance of different algorithms for a given dataset.

```{r}
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
```

```
## Algorithm Duration Accuracy
## 1 bagFDA 69 96.47
## 2 LogitBoost 2 97.65
## 3 nnet 8 97.06
## 4 svmRadialCost 3 94.12
``` 

##Conclusions
Given that there is one main principal component PC1, most algorithms will perform with excellent accuracy.
The large predictors can be easily compressed using PCA and then used for prediction.
