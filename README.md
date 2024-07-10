# Happy Customers - Apziva Project (#1)
By Samuel Alter

Apziva: 18RTcr7zXIKyc7qb

## TL;DR
* Project centered on **predicting customer happiness** based on the results of a survey, sourced from a **food delivery company**, to attain ≥73% accuracy
  * The stretch goal was to determine which features are most important for predicting customer happiness
* Survey had **126 total observations** with 69 positive and 57 negative, for a **positive rate of about 55%**
* The **low number of observations was the largest challenge of the project**, namely: how to increase models' performance when there are few observations
* I **discretized the dataset** to **simplify the modeling**
* **`LazyClassifier`** was used for initial model exploration
* **`Hyperopt`** was employed to search the hyperparameter space. **`RFE`** was used for all models except the `LogisticRegression` to try and achieve the stretch goal. I opted for many shallow estimators to reduce overfitting while keeping efficiency high.
  * Algorithms used: `ExtraTreesClassifier`; `XGBoost`; `DecisionTreeClassifier`; `RandomForestClassifier`; `LGBMClassifier`; `LogisticRegression`, searching through all relevant solvers; and `LogisticRegression`, searching with just the `liblinear` solver 
* The accuracies were still **too volatile** as the **random seed dictated success and failure**
* Finally, I used the stacking and voting ensembling methods, using the tuned hyperparameters from the models above as parameters
  * **Stacking** achieved an **accuracy in the low 60%s**
  * Voting fared very poorly

**Take-home messages:**
* The company's **delivery time** elicited the **highest average satisfaction**
* Upon opening their order, the customers rated the **contents of the order** the **worst average satisfaction**
* The **results** of our modeling show:
  * The low number of observations have a big effect on the modeling performance and RFE was not conclusive  
  * We were still able to improve upon the baseline accuracy **from about 55%** to **over 60%**
* I **suggest** that the company:
  * Continue to have **good delivery times**  
  * Ensure that the **contents of the order** are what the customer wanted  
  * The company would do well to **gather more survey responses**, which would **help improve the performance of the models**
* <mark>**_Question for the reader: What algorithms are best for small datasets? What should I try next?_**</mark>

## Overview
This project centers on training a model to predict customer satisfaction based on results of a customer survey from a delivery company. 

### The dataset
The dataset consists of the following:
* `Y`: The target attribute, indicating whether the customer noted their happiness or unhappiness
* `X1`: Order was delivered on time
* `X2`: Contents of the order was as expected
* `X3`: I ordered everything that I wanted to order
* `X4`: I paid a good price for my order
* `X5`: I am satisfied with my courier
* `X6`: The app makes ordering easy for me

Attributes `X1` through `X6` are on a 1 to 5 scale, with 5 indicating most agreement with the statement.

### Goals
* Train a model that predicts whether a customer is happy or not, based on their answers to the survey. 
* Reach 73% accuracy or higher with my modeling
  * Or explain why my solution is superior.

* **Stretch Goal**: determine which features are more important.
  * What is the minimal set of attributes or features that would preserve the most information, while at the same time increasing predictability?
  * See if any question can be eliminated in the next survey round.

## EDA
![Distribution of customer happiness in target (y). 54.76% of respondents were happy, while 45.0% of them were unhappy](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/197cb671-eebb-4526-9bb4-24a5800beef1)

54.76% of the respondents were happy, while 45.0% of them were unhappy. The roughly 55% base rate of customer happiness will serve as the baseline for comparing our modeling efforts' success.

![2_xdistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/852df7d2-6022-416e-8734-a77a8917f7d2)

This plot illustrates well the distribution of responses received in the survey. This is helpful to understand the overall trends in the data.

![3_xmeandistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/ca427f60-15f5-4563-b8eb-f0549228524c)

The delivery time and app experience had the highest mean satisfaction in the survey. Customers were least satisfied with what they expected of the contents of their order.

![4_Correlation Matrix](https://github.com/sralter/Happy_Customers/assets/25013680/e32bad69-6165-4c1e-94a8-7d348d484c08)

The results of the correlation matrix show that if one aspect of the experience is positive, the customer will rate others positive as well. One interesting correlation to highlight is the courier and time are connected, which makes sense: the courier is the person that gives you your order, and if the courier is on time you probably will rate the courier highly too.

### EDA Summary
In the dataset that we were given, with 126 observations, roughly half of the respondents were unhappy. From a business standpoint, this is an opportunity to increase the amount of satisfied customers. Hence the survey, ostensibly to understand how the company can improve the satisfaction of their customers.

The results from the survey show that the delivery time and the app experience are places where the company is doing well. Areas for improvement are ensuring that the order is prepared correctly and customers being able to find what they need when they place an order.

We need to shift to modeling to understand which survey questions are most important and which can be removed. We will do this in the subsequent sections below.

## Modeling
We discretized the features into binary so that if the respondents scored a 4 or 5, I would label that a 1; otherwise, I would label it a 0. I called this engineered dataset the "threshold" dataset. This process will simplify the analysis for the models.

### `lazypredict`
[`lazypredict`](#https://lazypredict.readthedocs.io/en/latest/) is a very helpful package that can run through generic builds of a multitude of models in order to get a high-level understanding of the performance of these models on your particular dataset. It saves a lot of time that would be spent manually exploring the accuracy of different models.

The following table shows the first ten rows of the results from one run of `lazypredict`:

|                         Model | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken |
|------------------------------:|---------:|------------------:|--------:|---------:|-----------:|
|                   BernoulliNB |     0.77 |              0.79 |    0.79 |     0.76 |       0.00 |
|               NearestCentroid |     0.77 |              0.79 |    0.79 |     0.76 |       0.01 |
| QuadraticDiscriminantAnalysis |     0.77 |              0.77 |    0.77 |     0.77 |       0.00 |
|                    GaussianNB |     0.77 |              0.77 |    0.77 |     0.77 |       0.01 |
|            AdaBoostClassifier |     0.69 |              0.70 |    0.70 |     0.69 |       0.03 |
|    LinearDiscriminantAnalysis |     0.69 |              0.70 |    0.70 |     0.69 |       0.01 |
|                 XGBClassifier |     0.69 |              0.70 |    0.70 |     0.69 |       0.08 |
|             RidgeClassifierCV |     0.69 |              0.70 |    0.70 |     0.69 |       0.01 |
|               RidgeClassifier |     0.69 |              0.70 |    0.70 |     0.69 |       0.01 |
|        RandomForestClassifier |     0.69 |              0.70 |    0.70 |     0.69 |       0.09 |

The following algorithms were chosen to be run in their default formulations as they usually scored highly in the `LazyClassifier` exploration:
* `XGBClassifier`
* `LGBMClassifier`
* `DecisionTreeClassifier`
* `QuadraticDiscriminantAnalysis`

The results of this modeling, however, were poor. This led us to try `Hyperopt`, a powerful tool that can help search for the optimal hyperparameters. 
* RFE was used to help select a subset of the features to help answer the stretch goal of the project

### `Hyperopt`
The following algorithms were used. When applicable, I opted for many shallow estimators to reduce overfitting while increasing speed and efficiency of the search:
* `ExtraTreesClassifier` with Recursive Feature Elimination (RFE)
* `XGBoost` with RFE
* `DecisionTreeClassifier` with RFE
* `RandomForestClassifier` with RFE
* `LGBMClassifier` with RFE
* `LogisticRegression`, searching through all relevant solvers
* `LogisticRegression`, searching with just the `liblinear` solver  

As with the generic models, this effort was not fruitful. I tried one more attempt, this time with the ensemble methods of **stacking** and **voting**.

### Ensembling Methods
By combining the outputs of multiple models together into a metamodel, we could potentially achieve a better accuracy.
* Stacking
  > Achieved accuracies in the low 0.60s
* Voting
  > Results were poor

## Conclusion
* The company's **delivery time** elicited the **highest average satisfaction**
* Upon opening their order, the customers rated the **contents of the order** the **worst average satisfaction**

The results of our modeling show:
* The low number of observations have a big effect on the modeling performance and feature elimination was not conclusive
* That being said, we were able to improve upon the baseline accuracy **from about 55%** to **over 60%**.

I suggest that the company:
* Continue to have good delivery times
* Ensure that the contents of the order are what the customer wanted
* The company would do well to **gather more survey responses**, which would **help improve the performance of the models**

<mark> **Finally, I ask you, dear reader: what algorithms are best for small datasets? What should I try next?** </mark>
