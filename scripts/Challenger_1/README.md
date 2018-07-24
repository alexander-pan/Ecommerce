# Challenger 1 Recommender System

This folder contains scripts used for implementing a recommender system based on the Extremely Randomized Trees algorithm.

## Motivation

Our client wishes to leverage their retail data on users' past purchases to inform future marketing decisions. In particular, the client has the ability to send out promotion material on as granular as the department level. For this reason, we constructed a model with the aim of predicting the "relative probability" (a score) that on a given reference date, a user will purchase an item within a specified department within the next 30 days. If we wished to send promotion material to every user, the idea would be to send each user an item in the department where they have the highest score. If we wished to send promotion material with a maximum volume constraint = n, the idea would be to determine each users' highest scored department, and then send the top n user-department pairs from that set.

## Input Features

The input features for the model are gathered over a 60 day window prior (lookback window) to the reference date. Additionally, input feature definitions can be broken into groupings.

### Continuous Features
The basic continuous fields used are: 
 - original_retail_price_amt
 - shipped_cost_amt
 - shipped_sold_amt
 - margin
 - discount
 - markdown

For each of those basic fields, we calculated:
 - The sum of their past amounts for the user, within the reference department
 - The sum of their past amounts for the user, over all departments
 - The sum of their past amounts over all orders within the reference department
 - The mean of their past amounts for the user, within the reference department
 - The mean of their past amounts for the user, over all departments
 - The mean of their past amounts over all orders within the reference department

We also calculated:
 - Days since most recent order for the user, within the reference department
 - Days since most recent order for the user, over all departments
 - Average number of days between orders for the user, within the reference department
 - Average number of days between orders for the user, over all departments
 - The number of past orders for the user, within the reference department
 - The number of past orders for the user, over all departments
 - The user's age

Additionally, for each categorical fields below, we calulated the percentage of orders fitting the field's subcategory, over all orders within the reference department 
 - fabric_category_desc: [Cotton/Cotton Bl, Synthetic/Syn Blend, Linen/Linen Bl],
 - pay_type_cd: [VISA, MC, DISC, CASH, DEBIT, JJC, CK]
 - end_use_desc: [Core, Wearever, Pure Jill]
 - price_cd: [SP, FP]
 - master_channel: [D, R]

### Categorical Features
The basic categorical fields used are: 
 - first_order_date
 - first_catalog_order_date
 - first_retail_order_date
 - first_web_order_date
 - prior_order_date

For each of those basic fields, we caluculated:
 - An binary indicator telling whether a past date exists

## Training
Training the model requires two steps:

 1) [Data Gathering](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/gather_data.py) (run with: gather_data.py train)
 2) [Model Training](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/modeling.py) (run with: modeling.py train)

For the data gathering step, parameters for the training set must be defined. These parameters (with training values given) are:
 - num_users: '3000'
 - min_date: '2017-10-01'
 - max_date: '2018-02-10'
 - valid_departments: "('Knit Tops','Woven Shirts','Dresses','Pants')"
 - lookback_window: '60'
 - lookfront_window: '30'

Note: num_users = 3000 does not mean there are only 3000 samples. There will actually be a data sample for each row in the cross join of each date (reference date) falling between the min_date and max_date, the 3000 users, and the departments in valid_departments. Some of those tuples, however, are removed if they are closer than lookfront_window to the max_date or lookback_window to the min_date.

For the modeling step, our data is randomized and class-balanced before splitting into training & evaluation sets (70% training; the training & evaluation sets contain samples from completely distinct users). The model used is an Extremely Randomized Trees classifier (please see [here](http://scikit-learn.org/stable/modules/ensemble.html#forest) for details). Tree ensemble classifiers tend fall midway on the bias-variance spectrum, so they are good general purpose models that can capture a good deal of nonlinearity. Jjill does possess a large amount of data, which might suggest that a "deep learning" type approach might be appropriate, but suitable hardware & environment requirements (gpus, db with high performance, etc.) should first be met before considering such an approach. 

As output from training, some evaluation metrics are provided. Using the evaluation set mentioned above, we assess the prediction accuracy (# of correct predictions / # of predictions) on the set and compare it to the random accuracy of 50%. For a model using the variables defined above, our prediction accuracy comes out to ~66%. 
 
## Evaluation (Top N Performance)
Like training, this evaluation requires two steps: 

 1) [Data Gathering](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/gather_data.py) (run with: gather_data.py eval)
 2) [Model Training](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/modeling.py) (run with: modeling.py eval)

For the data gathered, the time period should be distinct from training. Here are example parameters used for our analysis:
 - num_users: '3000'
 - min_date: '2018-03-17'
 - max_date: '2018-04-17'
 - valid_departments: "('Knit Tops','Woven Shirts','Dresses','Pants')"
 - lookback_window: '60'
 - lookfront_window: '30'

Only samples from the latest reference date in the returned dataset are kept. These are then used for the evaluation described at the end of the motivation section above. We can then visualize these results with a couple of charts. The first chart below has, on the y-axis, the prediction accuracy of class 1 (the class where a user does purchase an item from the reference department within the 30 day period after the reference date). On the x-axis is the number of top-scored samples evaluated. The percentages below each tick mark are the x-value divided by the total number of samples scored. As expected of a functioning model, we can see the curve (blue) decrease as x increases. The red line is the percentage of class 1 samples over the whole dataset gathered, so it acts as a random-guess baseline. 

![Top scored accuracy top preds](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/top_scored_accuracy_top_preds.png)

Instead of looking at all scored samples, the next chart below only looks at the best score for each user (so resulting subset only has 1 department for each user). Using that subset of data, we create a plot using the same methods as the chart above. Again, we can see the curve (blue) decrease as x increases.

![Top scored accuracy top preds 1 per user](https://github.com/alexander-pan/Ecommerce/blob/master/scripts/Challenger_1/top_scored_accuracy_top_preds_1p_user.png)