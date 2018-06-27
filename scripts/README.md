# Include Scripts to Build/Transform Categorical and Numerical Fts

## Workflow of Scripts to Run to Build/Transform Datasets
X = Timeframe to extract user purchase data
1. Run buildNumeric_X ~
2. Run buildCategorical_X ~
3. Run MergeCatNumeric_X
4a) In Process of creating clustering script to label customers by clusters (if useful)
4b) Run RFC_Y to train and save model 
