#importing Datasets
dataset = read.csv('50_Startups.csv')

#Encoding Categorical data
dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'), 
                       labels = c(1,2,3))

set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple Linear Regression to the training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

#Predict the Test set data
Y_Predict = predict(regressor, newdata = test_set)

# Building model using Backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)

summary(regressor)