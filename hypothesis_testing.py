# ============================================================
# Hypothesis Testing: Electronics vs Grocery - Total_Sales
# Sales Data Analysis Project - Data Analytics Internship
# ============================================================

# Step 1: Import the required libraries
# pandas -> to read and handle the dataset
# scipy.stats -> to perform the statistical T-Test
import pandas as pd
from scipy import stats

# Step 2: Read the dataset from the CSV file
# Make sure sales_dataset.csv is in the same folder as this script
df = pd.read_csv("sales_dataset.csv", sep="\t")
print(df.columns.tolist())

# Step 3: Filter the dataset to keep only Electronics and Grocery categories
electronics_df = df[df["Category"] == "Electronics"]
grocery_df = df[df["Category"] == "Grocery"]

# Step 4: Extract the Total_Sales column for both categories
# These are the two samples we will compare using the T-Test
electronics_sales = electronics_df["Total_Sales"]
grocery_sales = grocery_df["Total_Sales"]

# Step 5: Print the number of records in each category
# This tells us how many orders belong to each category
print("Number of records in Electronics:", len(electronics_sales))
print("Number of records in Grocery:", len(grocery_sales))

# Step 6: Calculate and print the mean (average) Total_Sales for each category
electronics_mean = electronics_sales.mean()
grocery_mean = grocery_sales.mean()

print("Average Total_Sales for Electronics:", round(electronics_mean, 2))
print("Average Total_Sales for Grocery:", round(grocery_mean, 2))

# Step 7: Perform an Independent Samples T-Test
# This test checks if the means of the two independent groups
# (Electronics and Grocery) are significantly different from each other
t_statistic, p_value = stats.ttest_ind(electronics_sales, grocery_sales)

# Step 8: Print the T-Statistic and P-Value
# T-Statistic -> shows the size of the difference relative to the variation in the data
# P-Value -> tells us the probability that the observed difference happened by random chance
print("T-Statistic:", round(t_statistic, 4))
print("P-Value:", round(p_value, 4))

# Step 9: Interpret the result automatically based on the P-Value
# A P-Value less than 0.05 means the result is statistically significant
significance_level = 0.05

if p_value < significance_level:
    print("Result: Reject the Null Hypothesis.")
    print("Conclusion: There IS a statistically significant difference in average Total_Sales between Electronics and Grocery.")
else:
    print("Result: Fail to Reject the Null Hypothesis.")
    print("Conclusion: There is NO statistically significant difference in average Total_Sales between Electronics and Grocery.")
