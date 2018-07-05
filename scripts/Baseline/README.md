# Instructions in how to build datasets
- To Build/Transform from Master Dataset Follow instructions
- buildNumeric_X_Baseline.py, where X = approx. count of unique customers
1. Run "python buildNumeric_15k_Baseline.py"
  - builds numerical fts and "Bought_Dept" features
2. Run "python buildCategorical_15k_Baseline.py" 
  - builds the categorical fts: "HAS_Y" and "%\_IN_Y", where Y is the 
  - current categorical fts
3. Run "python MergeCatNumeric.py"
