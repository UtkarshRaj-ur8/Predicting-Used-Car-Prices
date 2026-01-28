# Predicting-Used-Car-Prices
# Overview
The used car market is notorious for opaque pricing. Buyers struggle to know if they are paying too much, and sellers struggle to set competitive prices. In this project, I built a machine learning engine to predict the fair market value of a vehicle based on its specifications (Age, Kilometers Driven, Fuel Type, etc.).

This tool aims to solve the "Information Asymmetry" problem in the auto market, providing data-driven price transparency for both buyers and dealerships.

# Technologies Used
- Core Stack: Python, Pandas, NumPy
- Machine Learning: Scikit-Learn (Random Forest, Gradient Boosting, Linear Regression, Decision Trees)
- Feature Engineering: OneHotEncoder, StandardScaler, Scikit-Learn Pipelines
- Visualization: Seaborn, Matplotlib, Plotly (for interactive charts)
- Deployment: Joblib (Model Persistence)

# Features
- End-to-End Pipeline: A complete workflow from raw data cleaning (handling null values, parsing strings like "50,000 km") to final model inference.
- Smart Feature Extraction:
   - Derived CarAge from the "Year" column.
   - Parsed "Engine Capacity" and "Power" from string formats (e.g., "1197 cc") into numerical features.
- Interactive EDA: Visualized depreciation curves to show how value drops over time and how different brands retain value differently.
- Single-Instance Prediction: Includes a helper function that takes a raw dictionary of car details (e.g., {'Model': 'City', 'Km': 50000}) and returns a price estimate in currency format.

# The Process
1. Exploratory Data Analysis (EDA):
- Discovery: "Car Age" and "Kilometers Driven" have the strongest negative correlation with price (Depreciation).
- Discovery: "Fuel Type" matters—Diesel cars often command a different premium compared to Petrol based on the region and engine size.
- Multicollinearity: Found that CarAge and ModelYear were perfectly correlated, so I dropped one to reduce redundancy.

2.Data Cleaning:
- Handled complex string data (e.g., removing "km", "bhp", "CC" units) to convert columns to numeric types.
- Imputed missing values in "Seats" and "Engine" columns using the mode of similar car models.

3. Model Selection:
- Benchmarked Linear Regression vs. Tree-Based Models.
- Linear Models struggled with the non-linear depreciation curves.
- Random Forest and Gradient Boosting performed best because they could capture "threshold" effects (e.g., price drops sharply after 100k km).

4.Final Model:
- Selected Random Forest Regressor for its high accuracy and resistance to overfitting on this dataset.

# What I Learned
- Real-World Data is Messy: 80% of the effort went into parsing strings (removing units like 'km/kg', 'bhp') and standardizing names.
- The "Vintage" Effect: While older cars generally cost less, certain "Classic" models or specific brands (like Toyota) depreciate much slower. Capturing this required the model to interact "Brand" with "Age".
- Pipeline Usage: I used ColumnTransformer to apply different preprocessing steps (Scaling for numericals, One-Hot for categorical) simultaneously, which makes the code much cleaner and safer for deployment.

# Overall Growth
- Technical: Improved my regex (Regular Expression) skills for data cleaning and mastered ColumnTransformer pipelines.
- Business: Understood how to frame "Error Metrics" (RMSE) as "Pricing Tolerance." (e.g., "The model is accurate within ±₹20,000").

# How can it be improved?
- Image Processing: Using CNNs to analyze photos of the car to detect scratches or dents, which significantly lower value.
- Live Scraper: Building a web scraper to fetch real-time listings from auto websites to keep the training data current.
- Geospatial Adjustment: Adding "City" or "State" logic, as cars in coastal areas (rust risk) or metros (traffic wear) might be valued differently.
