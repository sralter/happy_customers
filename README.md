# HappyCustomer - An Apziva Project (#1)
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
![1_ydistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/4ebc9726-9dbe-4909-82c2-a8d40d827796)

54.76% of the respondents were happy, while 45.0% of them were unhappy.


![1_xdistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/4b6ea23b-3d17-48d4-8a4c-cb468f65a6c7)

![1_xmeandistribution](https://github.com/sralter/UP2IqAzAWrVBrULk/assets/25013680/ca427f60-15f5-4563-b8eb-f0549228524c)

The delivery time and app experience had the highest mean satisfaction in the survey. Customers were least satisfied with what they expected of the contents of their order.
