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

Apziva: UP2IqAzAWrVBrULk
