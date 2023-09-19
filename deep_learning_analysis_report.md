# deep learning Analysis Report

## Overview of the Analysis

The objective of this analysis is to create a binary classifier using a dataset provided by Alphabet Soup to predict the probability of funding applicants to achieve success. The dataset includes various features that can be used to train the model using different machine learning methods, and the ultimate goal is to achieve an accuracy score of over 75%. 

To reach this goal, several steps were taken. First, the given dataset was preprocessed to handle missing data and convert categorical variables into numerical ones. Then, deep neural network model was trained on the preprocessed dataset. The model assessed and compared the result using standard performance metrics such as model_loss and model_accuracy,to determine the Loss and accuracy.  

## Results

## Preprocessing of Data

The objective of the model is to predict the success of funding applicants, which is indicated by the IS_SUCCESSFUL column in the dataset. 

## For initial model

All columns in the dataset, except for the target variable and non-relevant variables such as EIN and names, are considered feature variables. These features provide pertinent information about the data and are useful for predicting the target variable. Non-relevant variables were removed to prevent any potential noise that could interfere with the model's accuracy in the initial module.

The APPLICATION_TYPE and CLASSIFICATION columns contained rare occurrences, so I implemented binning or bucketing to consolidate them. Additionally, I transformed categorical data into numeric data through one-hot encoding, which makes the data suitable for machine learning algorithms that require numerical input. 

To prepare the data for model training, I partitioned it into separate sets for features and targets, as well as for training and testing. Lastly, I scaled the data to ensure uniformity in the distribution of the data values.

For the neural network, I opted for 3 layers: an input layer with 80 neurons in the first layer, 30 neurons in the second layer, and an output layer with a single neuron. Given that the model's objective was binary classification, I utilized the relu activation function in the first and second layers, and the sigmoid activation function in the output layer. 

After training the model for 100 epochs, an accuracy of 72.76% was achieved for the testing data. No overfitting or underfitting was apparent during the testing, indicating that the model was appropriately optimized for the data at hand.

## For Optimization model

Only EIN column was removed as non beneficial variable.The NAME,APPLICATION_TYPE and CLASSIFICATION columns contained rare occurrences, so I implemented binning or bucketing to consolidate them.

For the neural network, I maintained the 3 layers, but with different  input layer of 10 neurons in the first layer, 20 neurons in the second layer, and an output layer with a single neuron, and still utilized the relu activation function in the first and second layers, and the sigmoid activation function in the output layer.

After training the model for 100 epochs, an accuracy of 78.35% was achieved for the testing data.




## Summary

The optimization model exceeded the target accuracy of 75%. To enhance the performance of the model, there are some possible approaches, such as utilizing hyperparameter optimization using the keras-tuner or improving data cleaning processes. 

Notably, retaining the NAME variable in the model was crucial to achieving a 75% accuracy rate within the current model design. Further tuning and refining could potentially improve the model performance.
