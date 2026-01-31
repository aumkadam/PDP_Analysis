# PDP_Analysis
# NOVICA Marketplace Data Analysis (Python EDA)

## Overview
This project performs an exploratory data analysis (EDA) on product-level marketplace data from NOVICA to understand sales performance, pricing behavior, and revenue drivers across thousands of SKUs. The analysis focuses on identifying underperforming products, pricing inefficiencies, and opportunities for revenue optimization.

## Objective
- Analyze product performance across the NOVICA marketplace
- Identify underperforming and high-potential SKUs
- Study pricing, sales volume, and revenue distribution
- Generate data-driven insights to support business and merchandising decisions

## Dataset
- **File:** `novica_final.csv`
- **Description:** Product-level marketplace dataset containing pricing, sales, and revenue metrics
- **Size:** Thousands of SKUs (long-tail heavy marketplace data)
- **Note:** Dataset is included in the repository for reproducibility

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Jupyter / Python scripting

## Analysis Performed
Key analyses included in this project:
- Data cleaning and preprocessing
- SKU-level sales and revenue distribution analysis
- Identification of long-tail underperforming products
- Price vs. performance relationship analysis
- Revenue concentration and Pareto (80/20) analysis
- Exploratory visualizations to surface business insights

## Key Insights
- A large proportion of SKUs contribute minimally to total revenue, highlighting a strong long-tail effect
- Small subsets of products drive a majority of sales and revenue
- Pricing inconsistencies exist across similar product categories
- Clear opportunities for SKU rationalization and targeted optimization

## Files in Repository
- `Capstone EDA.py` – Python script containing the full exploratory data analysis
- `novica_final.csv` – Dataset used for analysis
- `README.md` – Project documentation

## How to Run
1. Clone the repository
2. Ensure required Python libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn
