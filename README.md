# LinearRegression
The code provided is an example of a machine learning project that predicts brain weight based on various features such as gender, age range, and head size. Here's a brief summary of how the code works:

1. Data Preparation:
   - The code imports the necessary libraries, including Pandas, NumPy, Matplotlib, and Seaborn.
   - It reads the brain data from a CSV file into a Pandas DataFrame called `brain`.
   - The code performs exploratory data analysis using Seaborn to visualize the relationship between head size and brain weight.

2. Data Preprocessing:
   - The code prepares the data for training by splitting it into input features (`X`) and the target variable (`y`).
   - The `train_test_split` function from scikit-learn is used to split the data into training and testing sets.

3. Model Training:
   - The code creates an instance of `LinearRegression` from scikit-learn to build a linear regression model.
   - It uses `OneHotEncoder` to encode categorical features (`Gender` and `Age Range`).
   - The `make_column_transformer` function is used to create a column transformer that applies the one-hot encoding to the specified columns and leaves other columns unchanged.
   - A pipeline is created using `make_pipeline` that combines the column transformer and the linear regression model.
   - The pipeline is then trained on the training data.

4. Model Evaluation:
   - The trained model is used to make predictions on the testing data.
   - The `r2_score` function is used to evaluate the model's performance by calculating the coefficient of determination between the predicted values and the actual values.

5. Further Analysis:
   - The code includes an additional analysis where it iterates over multiple random states to split the data and train the model.
   - It stores the performance scores (`r2_score`) for each iteration in a list called `scores`.
   - The maximum score and its corresponding random state index are calculated using NumPy functions.
   - Finally, the code demonstrates how to use the trained model to predict the brain weight for a new sample.

This code serves as a basic example of using linear regression and one-hot encoding for predicting brain weight based on certain features. The performance of the model is evaluated using the coefficient of determination (R-squared).
