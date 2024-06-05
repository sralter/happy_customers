# Happy Customers - An Apziva Project (#1)
By Samuel Alter

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

54.76% of the respondents were happy, while 45.0% of them were unhappy.

![1_xdistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/852df7d2-6022-416e-8734-a77a8917f7d2)

This plot illustrates well the distribution of responses received in the survey. Although it is harder to draw conclusions from this figure, I think it is still valid to understand the overall trends in the data. Figure 3 has more explanatory value.

![1_xmeandistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/ca427f60-15f5-4563-b8eb-f0549228524c)

The delivery time and app experience had the highest mean satisfaction in the survey. Customers were least satisfied with what they expected of the contents of their order.

![Correlation Matrix](https://github.com/sralter/Happy_Customers/assets/25013680/e32bad69-6165-4c1e-94a8-7d348d484c08)

The results of the correlation matrix show that if one aspect of the experience is positive, the customer will rate others positive as well. One interesting correlation to highlight is the courier and time are connected, which makes sense: the courier is the person that gives you your order, and if the courier is on time you probably will rate the courier highly too.

### EDA Summary
In the dataset that we were given, roughly half of the respondents were unhappy. From a business standpoint, this is an opportunity to increase the amount of satisfied customers. Hence the survey, ostensibly to understand how the company can improve the satisfaction of their customers.

The results from the survey show that the delivery time and the app experience are places where the company is doing well. Areas for improvement are ensuring that the order is prepared correctly and customers being able to find what they need when they place an order.

We need to do more modeling to understand which survey questions are most important and which can be removed. We will do this in the subsequent sections below.

## Modeling
We will use 1574 as the random seed for our modeling efforts.

### `lazypredict`
[`lazypredict`](#https://lazypredict.readthedocs.io/en/latest/) is a very helpful package that can run through generic builds of a multitude of models in order to get a high-level understanding of the performance of these models on your particular dataset. It saves a lot of time that would be spent manually exploring the accuracy of different models.

The following table shows the first ten rows of the results from `lazypredict`:

|Model                        |Accuracy           |Balanced Accuracy  |ROC AUC            |F1 Score           |Time Taken           |
|-----------------------------|-------------------|-------------------|-------------------|-------------------|---------------------|
|LGBMClassifier               |0.7307692307692307 |0.7261904761904762 |0.7261904761904762 |0.7295582977741899 |0.16534090042114258  |
|BernoulliNB                  |0.7307692307692307 |0.7261904761904762 |0.7261904761904762 |0.7295582977741899 |0.006081819534301758 |
|SGDClassifier                |0.6153846153846154 |0.625              |0.625              |0.6108058608058609 |0.009582042694091797 |
|NearestCentroid              |0.6153846153846154 |0.6190476190476191 |0.6190476190476191 |0.6153846153846154 |0.0072171688079833984|
|Perceptron                   |0.6153846153846154 |0.6071428571428572 |0.6071428571428572 |0.6107226107226107 |0.00656580924987793  |
|NuSVC                        |0.5769230769230769 |0.5714285714285714 |0.5714285714285714 |0.575020182216584  |0.010294675827026367 |
|GaussianNB                   |0.5769230769230769 |0.5654761904761905 |0.5654761904761905 |0.5671747607231478 |0.006016969680786133 |
|CalibratedClassifierCV       |0.5769230769230769 |0.5476190476190477 |0.5476190476190476 |0.5014553014553015 |0.021075963973999023 |
|QuadraticDiscriminantAnalysis|0.5384615384615384 |0.5357142857142857 |0.5357142857142857 |0.5384615384615384 |0.008083820343017578 |
|LogisticRegression           |0.5384615384615384 |0.5297619047619048 |0.5297619047619048 |0.5328671328671329 |0.010271072387695312 |

`SGDClassifier` showed good results with the generic build of the model. We will also try modeling with the `XGBoost` algorithm.

### `XGBoost`
The results of the basic form of the model yield an accuracy of 46% - pretty dismal and lower than the base of 54% (in the entire dataset, 54% of respondents were happy).

#### Grid Search with XGBoost
* The best score on the training set was 67% using the following parameters:

|Parameter name | Value|
|-|-|
|`alpha`|0|
|`gamma`|0|
|`lambda`|0.275|
|`learning_rate`|1.6302342008295345|
|`max_depth`|2|
|`min_child_weight`|3.5|
|`n_estimators`|53|
* When the testing data is run with these parameters, we get a cross-validated accuracy of: 57.88% ± 8.47%

This is a better result than the base model, though I'm certain there's more we can do to increase the accuracy.

Apziva: UP2IqAzAWrVBrULk
