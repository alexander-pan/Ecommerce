# Dataset Summary
- Z = 30 days from Reference Date
- ILINK - ilink
- REF_DATE_MONTH = Reference date month
- REF_DATE_DAY = Reference date day
- REF_DATE_YEAR = Reference date year
- NUM_PAST_ORDERS = # of Orders in past Z
- SUM_PAST_SHIPPED_SOLD_AMT = Sum of Shipped Sold Amt in past Z
- AVG_PAST_SHIPPED_SOLD_AMT = Avg Shipped Sold Amt in past Z
- STDDEV_PAST_SHIPPED_SOLD_AMT = Std dev shipped sold amt in past Z
- VAR_PAST_SHIPPED_SOLD_AMT = variance shipped sold amt in past Z
- SUM_PAST_DISCOUNT = sum discount in past Z
- AVG_PAST_DISCOUNT = avg discoutn in past Z
- STDDEV_PAST_DISCOUNT = std dev in past Z
- VAR_PAST_DISCOUNT = variance discount in past Z
- BOUGHT_DEPT = whether they bought in a Dept, where dept = 'Woven Shirts', 'Dresses', 'Pants', 'Knit Tops', 'Other_Dept'
- HAS_Y' = (binary) The category and if any of the purchases were of this sub-category
- %\_IN_Y' = (percentage) % that customer bought in this sub-category

## Instructions in how to build datasets
- To Build/Transform from Master Dataset Follow instructions
- buildNumeric_X_Baseline.py, where X = approx. count of unique customers
1. Run "python buildNumeric_15k_Baseline.py"
  - builds numerical fts and "Bought_Dept" features
2. Run "python buildCategorical_15k_Baseline.py" 
  - builds the categorical fts: "HAS_Y" and "%\_IN_Y", where Y is the sub-categories in a categorical feature
  - current categorical fts: End_use_desc, master_channel, pay_type_cd, price_cd, fabric_category_desc
3. Run "python MergeCatNumeric.py" (include right files from /data directory)
 - Merges the numerical dataset with categorical dataset to make the completed customer dataset
 
## To Train Model
1. Include the right file for completed dataset
2. Run "python RFC_DeptVsAll_BaselineV1.py" (file V1/V2/etc. based on version)


