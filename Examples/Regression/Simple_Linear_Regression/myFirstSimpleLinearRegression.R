#Simple Linear regression Model

#importing Datasets
dataset = read.csv("Salary_Data.csv")

#Splitting the dataset into the training set and test set
#install.packages('caTools')
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple Linear Regression to the training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

#Predict the Test set data
Y_Predict = predict(regressor, newdata = test_set)

#Visualising the training set results
#install.packages("ggplot2")
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
             colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#Visualising the test set results
#install.packages("ggplot2")
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colours = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            colours = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')