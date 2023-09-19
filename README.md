# deep-learning-challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that could predict whether applicants was successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively.


Step 1: Preprocess the Data:

Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset. This step prepared me for Step 2, where I  compiled, trained, and evaluated the neural network model.

Started by uploading the starter file to Google Colab, then using the information provided in the Challenge files, followed the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

What variable(s) are the target(s) for your model?

What variable(s) are the feature(s) for your model?

Dropped the EIN and NAME columns.

Determined the number of unique values for each column.

For columns that have more than 10 unique values, determined the number of data points for each unique value.

Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then checked if the binning was successful.

Used pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Used these arrays and the train_test_split function to split the data into training and testing datasets.

Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.


Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, I  designed a neural network, or deep learning model, to create a binary classification model that could predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I thought about how many inputs there were before determining the number of neurons and layers in my model. After completing this step, I  compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

I continued using the file in Google Colab in which I performed the preprocessing steps from Step 1.

Created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Created the first hidden layer and choose an appropriate activation function.

I added a second hidden layer with an appropriate activation function.

Created an output layer with an appropriate activation function.

Checked the structure of the model.

Compiled and trained the model.

Created a callback that saves the model's weights every five epochs.

Evaluated the model using the test data to determine the loss and accuracy.

Saved and exported my results to an HDF5 file and named the file AlphabetSoupCharity.h5.


Step 3: Optimize the Model

Using my knowledge of TensorFlow, I  optimized my model to achieve a target predictive accuracy higher than 75%.