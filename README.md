# Neural_Network_Charity_Analysis

## Overview of the analysis:
Create a binary classifier that is capable of predicting whether applicants will be successful if funded by a foundation (example company Alphabet Soup).
- Using Pandas and the Scikit-Learn’s StandardScaler(), preprocess a dataset in order to compile, train, and evaluate a neural network model.
- Using TensorFlow, design a neural network, or deep learning model, to create a binary classification model that can predict if a funded organization will be successful based on the features in the dataset.
- Using TensorFlow, optimize the model in order to achieve a target predictive accuracy higher than 75%.

## Results:
The initial data consisted of the following columns:
- EIN
- NAME
- APPLICATION_TYPE
- AFFILIATION
- CLASSIFICATION
- USE_CASE
- ORGANIZATION
- STATUS
- INCOME_AMT
- SPECIAL_CONSIDERATIONS
- ASK_AMT
- IS_SUCCESSFUL

The target for the model is the column/category "IS_SUCCESSFUL". Due to the fields EIN and NAME not contributing any usable value to the machine learning model, they were removed.

The features for the model were the following columns, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.

Utilizing "nunique()" method to see how many unique values each column contained, it was determined to reduce the values contained within columns APPLICATION_TYPE and CLASSIFICATION by binning.

![image_name](/Resources/nunique.png)

Through the use of density plots for columns APPLICATION_TYPE and CLASSIFICATION, cutoff points were created to bin "rare" categorical values as "Other".

![image_name](/Resources/output_5_1.png)|![image_name](/Resources/output_8_1.png)

Afterwhich all columns containing values of type "object", were encoded into binary categories using "one-hot encoding".

The initial Compiling, Training, and Evaluating of the Machine learning model utilized the following configuration:

![image_name](/Resources/initial_model_build.png)

Initially I was only able to achieve a target model performance of 72.64%, which was below the acceptable accuracy of 75%.

![image_name](/Resources/Evaluate_model.png)

So I create a method that would create a new Sequential model with hyperparameter options, to attempt to target the most efficient and accurate neural network arrangement using "Keras_tuner".

After attempting 5 optimizations I was still unable to surpass 75% accuracy.

## Summary:
A 3 layer model consisting of relu, tanh, and sigmoid activations yielded a 72.7% accuracy, which is the best the model could produce using various number of neurons and layers. 

Trying a random forest classifier, as it is less influenced by outliers, would be my next recommendation.
