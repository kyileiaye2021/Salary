# Salary Prediction Based On Year Experience

A simple **linear regression model** is used to predict the salary based on the years of experience.
In this project, a salary dataset from **Kaggle** is used. In the project, year experience is an independent variable and salary is dependent variable. Linear Regression model was built first, using the dataset, pandas, numpy, skilearn, and matplotlib libraries. 
Then, the user can enter their year of experience and the model will predict how much salary they can expect. 

### Key Takeaways
1. X (independent feature) should be 2D dimensional array because fit() func allows X as 2D.
2. predict() func of LinearRegression return numpy array.
3. when accesssing the element of the pandas array, use pandas_array.iloc[index].
4. for numpy_arrays, directly access by using index.
5. predict() func expects a 2D array as an argument.
6. The input value of years_experience should have the feature name as the X feature.
