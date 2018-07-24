# Challenger 1 Recommender System

This folder contains scripts used for implementing a recommender system based on the Extremely Randomized Trees algorithm.

## Motivation

Our client wishes to leverage their retail data on users' past purchases to inform future marketing decisions. In particular, the client has the ability to send out promotion material on as granular as the department level. For this reason, we constructed a model with the aim of predicting the "relative probability" (a score) that on a given reference date, a user will purchase an item within a specified department within the next 30 days. If we wished to send promotion material to every user, the idea would be to send each user an item in the department where they have the highest score. If we wished to send promotion material with a maximum volume constraint = n, the idea would be to determine each users' highest scored department, and then send the top n user-department pairs from that set.

## Input Features

The input features for the model are gathered over a 60 day window prior (lookback window) to the reference date. Additionally, input feature definitions can be broken into several groupings.

### Continuous Reference Level Features
Reference level features are those calculated at the reference user-department-date level. The continuous fields used are: 
 - original_retail_price_amt
 - shipped_cost_amt
 - shipped_sold_amt
 - margin
 - discount
 - markdown
For each of those fields, we caluculated:
 - The sum of their past amounts over all departments
 - The sum of their past amounts within the reference department
 - The mean of their past amounts over all departments
 - The mean of their past amounts within the reference department
 Additionally, we calculated:
 - Days since most recent order over all departments
 - Days since most recent order within the reference department
 - Average number of days between orders over all departments
 - Average number of days between orders within the reference department
 - The number of past orders over all departments
 - The number of past orders within the reference department
 - The user's age

### Continuous Department Level Features
The continuous fields looked at are: original_retail_price_amt, shipped_cost_amt, shipped_sold_amt, margin, discount, markdown
For each of those fields, we caluculated:
 - The sum of their past amounts over all orders within the reference department
 - The mean of their past amounts over all orders within the reference department
Additionally, for each field below, we calulated the percentage of orders fitting that category, within the reference department 
 - fabric_category_desc: [Cotton/Cotton Bl, Synthetic/Syn Blend, Linen/Linen Bl],
 - pay_type_cd: [VISA, MC, DISC, CASH, DEBIT, JJC, CK],
 - end_use_desc: [Core, Wearever, Pure Jill],
 - price_cd: [SP, FP],
 - master_channel: [D, R]

### Categorical Reference Level Features
The fields used for these features are: first_order_date, first_catalog_order_date, first_retail_order_date, first_web_order_date, prior_order_date
For each of those fields, we caluculated:
 - An binary indicator telling whether a past date exists

## Training
Training the model requires two steps:

 1) **Data Gathering:** [gather_data.py](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/gather_data.py) (run with: gather_data.py train)
 2) **Model Training:**  [modeling.py](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/modeling.py) (run with: modeling.py train)

For the data gathering step, parameters defining the training set need to be defined. Those parameters (with example training values given) are:
 - num_users: '3000'
 - min_date: '2017-10-01'
 - max_date: '2018-02-10'
 - valid_departments: "('Knit Tops','Woven Shirts','Dresses','Pants')"
 - lookback_window: '60'
 - lookfront_window: '30'

As a note, num_users = 3000 here does not mean there are only 3000 samples. There will actually be a data sample for each date (call reference date) falling between the min_date and max_date, a cross join between the 3000 users, references dates, and departments in valid_departments. Some of those tuples, however, are removed if they are closer than lookfront_window to the max_date or lookback_window to the min_date.

For the model training step, the model used is an Extremely Randomized Trees classifier (please see [here](http://scikit-learn.org/stable/modules/ensemble.html#forest) for details). Tree ensemble classifiers tend fall midway on the bias-variance spectrum, so they are good general purpose models that can capture a good deal of nonlinearity. Jjill does possess a large amount of data, which might suggest that a "deep learning" type approach might be appropriate, but suitable hardware & environment requirements (gpus, db with high performance, etc) should first be met before considering such an approach. 

As output from training, some evaluation metrics are provided. Using a evaluation set of data samples involving users who are distinct from the training set users, we subsample from that set to create a class-balanced subset. We then assess the prediction accuracy (# of correct predictions / # of predictions) on the subset and compare it to the random of 50%. For a model using the variables defined above, our prediction accuracy came out to ~66%. 
 
## Evaluation (Top N Performance)
Like training, this evaluation requires two steps: 

 1) **Data Gathering:** [gather_data.py](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/gather_data.py) (run with: gather_data.py eval)
 2) **Model Training:**  [modeling.py](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/modeling.py) (run with: modeling.py eval)

For the data gathered, the time period should be distinct from training. Here are example parameters used for our analysis:
 - num_users: '3000'
 - min_date: '2018-03-17'
 - max_date: '2018-04-17'
 - valid_departments: "('Knit Tops','Woven Shirts','Dresses','Pants')"
 - lookback_window: '60'
 - lookfront_window: '30'

Only samples from the latest reference date in the returned dataset are kept. These are then used for the evaluation described at the end of the motivation section above. We can visualize these results with a couple of charts.