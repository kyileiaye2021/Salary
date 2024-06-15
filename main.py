'''
#Linear Regression Model
#Salary Prediction based on the year experience

#Predict the salary based on the year experience
#input var(independent): year
#output var(dependent): salary

#Key Takeaways
#1. X (independent feature) should be 2D dimensional array because fit() func allows X as 2D
#2. predict() func of LinearRegression return numpy array
#3. when accesssing the element of the pandas array, use pandas_array.iloc[index]
#4. for numpy_arrays, directly access by using index
#5. predict() func expects a 2D array as an argument
#6. The input value of years_experience should have the feature name as the X feature
'''

#importing necessary library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#loading the dataset from kaggle
data = pd.read_csv("Salary.csv")
df = pd.DataFrame(data) 

#separate dependent and independent variable
X = df[["YearsExperience"]] #independent variable #formatting X is a DataFrame which is two dimensional
y = df["Salary"] #dependent variable


#Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)#format the feature 'X' to be a 2-dimensional array so convert a pandas series to a dataframe

#model prediction
y_pred = model.predict(X_test) #return numpy array


'''  
#can also use this to iterate over the pandas and numpy
#enumerate func represents index , value
#so, enumerate(zip(x,y)) - index, (x,y)
for i, (experience, salary) in enumerate(zip(X_test["YearsExperience"], y_pred)):
    print(f"Experience: {experience}, Actual Salary:{y_test.iloc[i]}, Predicted Salary: {salary:.2f}")
'''

#when accessing the elements of the panda array, use panda_array.iloc[index]
#when accessing the elements of numpy array, use numpy_array[index]
for i in range(len(y_test)):
    print(f"Experience: {X_test['YearsExperience'].iloc[i]}, Actual Salary:{y_test.iloc[i]}, Predicted Salary: {y_pred[i]:.2f}")

print()
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

#scatter plot graph
plt.scatter(X_test,y_test)
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.plot(X_test, y_pred)
plt.title("Actual vs. Predicted Salary")
plt.show()
plt.close()

#Taking input of years of Experience from the user
years_experience = float(input("Enter the years of experience: "))
input_df = pd.DataFrame({"YearsExperience": [years_experience]}) #give the same feature name to the input data (otherwise we got a warning)
predicted_salary = model.predict(input_df)
print(f"Your Predicted Salary is around ${predicted_salary[0]:.2f}. However, it's subject to change!") #return the index 0 because predicted_salary is the numpy array


