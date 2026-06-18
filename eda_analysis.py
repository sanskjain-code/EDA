"""
===========================================================
 SALES DATASET - EXPLORATORY DATA ANALYSIS (EDA)
 Beginner-Friendly Version
===========================================================

This script performs 10 basic EDA analyses on a sales dataset.
Each section is kept simple and well-commented so you can
understand exactly what each line of code is doing.

Columns expected in the dataset:
Order_ID, Order_Date, Customer_ID, Customer_Name, Age, Gender,
City, Product, Category, Quantity, Unit_Price, Total_Sales
"""

# -----------------------------------------------------------
# STEP 1: IMPORT THE LIBRARIES WE NEED
# -----------------------------------------------------------
# pandas -> for loading and working with the data (tables)
# matplotlib.pyplot -> for creating charts
# seaborn -> makes charts look nicer and is easier to use

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Just makes the charts look a little nicer (optional)
sns.set_style("whitegrid")


# -----------------------------------------------------------
# STEP 2: LOAD THE DATASET
# -----------------------------------------------------------
# Change "sales_dataset.csv" to the actual name/path of your file

df = pd.read_csv("sales_dataset.csv", sep="\t")
print(df.columns.tolist())

# Let's look at the first 5 rows to make sure it loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

# Check basic info: column names, data types, missing values
print("\nDataset Info:")
df.info()

print("\nAny missing values?")
print(df.isnull().sum())


# -----------------------------------------------------------
# STEP 3: CONVERT ORDER_DATE TO A PROPER DATE FORMAT
# -----------------------------------------------------------
# This lets us group sales by month later on

df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
print(df['Order_Date'].dtype)

# Create a new column that stores just the Month (e.g., "2024-01")
df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)

# =============================================================
# ANALYSIS 1: SALES TREND OVER TIME (MONTHLY)
# =============================================================
# Goal: See how total sales change month by month

monthly_sales = df.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='blue')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("01_monthly_sales_trend.png")
plt.show()


# =============================================================
# ANALYSIS 2: TOP 10 BEST-SELLING PRODUCTS
# =============================================================
# Goal: Find which products generate the most sales

product_sales = df.groupby('Product')['Total_Sales'].sum()
top_10_products = product_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(top_10_products.index, top_10_products.values, color='skyblue')
plt.title("Top 10 Products by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("02_top_products.png")
plt.show()


# =============================================================
# ANALYSIS 3: SALES BY CATEGORY
# =============================================================
# Goal: See which product category brings in the most revenue

category_sales = df.groupby('Category')['Total_Sales'].sum()

plt.figure(figsize=(7, 7))
plt.pie(category_sales.values, labels=category_sales.index, autopct='%1.1f%%')
plt.title("Sales Share by Category")
plt.tight_layout()
plt.savefig("03_category_sales_pie.png")
plt.show()


# =============================================================
# ANALYSIS 4: SALES BY CITY
# =============================================================
# Goal: Find which cities generate the most sales

city_sales = df.groupby('City')['Total_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(city_sales.index, city_sales.values, color='lightgreen')
plt.title("Total Sales by City")
plt.xlabel("Total Sales")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("04_city_sales.png")
plt.show()


# =============================================================
# ANALYSIS 5: CUSTOMER AGE DISTRIBUTION
# =============================================================
# Goal: Understand the age range of our customers

plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=10, color='orange', edgecolor='black')
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("05_age_distribution.png")
plt.show()


# =============================================================
# ANALYSIS 6: GENDER DISTRIBUTION
# =============================================================
# Goal: See the male vs female split of customers

gender_counts = df['Gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Gender Distribution of Customers")
plt.tight_layout()
plt.savefig("06_gender_distribution.png")
plt.show()


# =============================================================
# ANALYSIS 7: AVERAGE ORDER VALUE BY CATEGORY
# =============================================================
# Goal: Find which category has the highest average order value
# (Average Order Value = average Total_Sales per order)

avg_order_value = df.groupby('Category')['Total_Sales'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(avg_order_value.index, avg_order_value.values, color='salmon')
plt.title("Average Order Value by Category")
plt.xlabel("Category")
plt.ylabel("Average Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("07_average_order_value.png")
plt.show()


# =============================================================
# ANALYSIS 8: QUANTITY vs UNIT PRICE
# =============================================================
# Goal: Check if cheaper products are bought in larger quantities

plt.figure(figsize=(8, 5))
plt.scatter(df['Unit_Price'], df['Quantity'], alpha=0.5, color='purple')
plt.title("Quantity vs Unit Price")
plt.xlabel("Unit Price")
plt.ylabel("Quantity")
plt.tight_layout()
plt.savefig("08_quantity_vs_unitprice.png")
plt.show()


# =============================================================
# ANALYSIS 9: TOP 10 CUSTOMERS BY NUMBER OF ORDERS
# =============================================================
# Goal: Identify our most frequent (loyal) customers

# value_counts() counts how many times each Customer_ID appears
top_customers = df['Customer_ID'].value_counts().head(10)

plt.figure(figsize=(10, 5))
plt.barh(top_customers.index.astype(str), top_customers.values, color='teal')
plt.title("Top 10 Customers by Number of Orders")
plt.xlabel("Number of Orders")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig("09_top_customers.png")
plt.show()


# =============================================================
# ANALYSIS 10: CORRELATION BETWEEN NUMERIC COLUMNS
# =============================================================
# Goal: Check if numeric columns (Age, Quantity, Unit_Price, Total_Sales)
# are related to each other. Values close to 1 or -1 mean strong
# relationship; values close to 0 mean little to no relationship.

numeric_data = df[['Age', 'Quantity', 'Unit_Price', 'Total_Sales']]
correlation = numeric_data.corr()

plt.figure(figsize=(7, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("10_correlation_heatmap.png")
plt.show()


# -----------------------------------------------------------
# DONE!
# -----------------------------------------------------------
print("\nAll 10 charts have been created and saved as PNG files")
print("in the same folder as this script.")