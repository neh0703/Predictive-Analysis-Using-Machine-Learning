# Predictive-Analysis-Using-Machine-Learning

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: NEHA KARAL

*INTERN ID*: CT08DY1991

*DOMAIN*: DATA ANALYST

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

**Project title: Predictive Analysis Using Machine Learning**

This Jupyter notebook, titled "Predictive Analysis Using Machine Learning," provides a thorough exploration of predictive modeling techniques applied to a superstore sales dataset. The primary objective is to build and evaluate machine learning models for two distinct tasks: regression to predict profit margins and classification to determine whether a transaction is profitable. By leveraging Python libraries such as scikit-learn, pandas, numpy, matplotlib, and seaborn, the notebook demonstrates end-to-end data science workflows, from data preprocessing to model evaluation and visualization.

The analysis begins with loading the dataset, "Sample_Superstore_Cleaned (1).csv," which contains transactional data including sales, quantity, discount, profit, and categorical attributes like ship mode, segment, region, and product categories. The dataset's shape is printed, and an initial preview is shown to understand its structure. Feature engineering plays a crucial role, transforming raw data into meaningful predictors. Date columns ('Order Date' and 'Ship Date') are converted to datetime format, enabling the calculation of shipping delay in days. Year and month are extracted from order dates for temporal insights. High-cardinality categorical features such as City, State, Product ID, and Customer ID are encoded using frequency encoding to capture their prevalence without creating excessive dummy variables. A binary classification target, 'is_profitable,' is derived from profit values (1 if profit > 0, else 0), allowing for both regression and classification modeling.

Feature sets are defined, separating numeric features (e.g., Sales, Quantity, Discount, shipping delay, frequency-encoded variables, and temporal features) from categorical ones (e.g., Ship Mode, Segment, Region, Category, Sub-Category). The data is split into training and testing sets (80-20 ratio) with a fixed random state for reproducibility. Preprocessing pipelines are constructed using scikit-learn's ColumnTransformer: numeric features undergo imputation (median strategy) and standardization, while categorical features are one-hot encoded with handling for unknown categories.

For regression, two models are trained to predict 'Profit': Linear Regression and Random Forest Regressor. Performance is evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). The Random Forest model, with 200 estimators, generally outperforms Linear Regression, as indicated by lower error metrics and higher R², highlighting the importance of non-linear relationships in the data.

Classification focuses on predicting profitability. Logistic Regression and Random Forest Classifier are employed. Metrics include Accuracy, Precision, Recall, F1-Score, and ROC AUC. The Random Forest Classifier achieves strong results, with high accuracy and AUC, suggesting effective handling of complex interactions. A confusion matrix is visualized using a heatmap, providing insights into true positives, false positives, and misclassifications.

Visualization enhances interpretability. Feature importance from the Random Forest Regressor is plotted, revealing top contributors like Sales, Discount, and specific categorical encodings (e.g., Sub-Category). The confusion matrix for the classifier is displayed as a heatmap. An ROC curve illustrates the trade-off between true positive and false positive rates, with the AUC quantifying model discriminatory power.

Overall, this notebook serves as an educational resource for data analysts and machine learning practitioners, showcasing best practices in feature engineering, pipeline-based preprocessing, model selection, and evaluation. It emphasizes the value of ensemble methods like Random Forest for handling diverse data types and non-linearities. The code is modular, reusable, and well-documented, making it adaptable for similar predictive tasks. By integrating regression and classification, it provides a holistic view of predictive analytics, potentially informing business decisions in retail optimization, such as pricing strategies or inventory management. The notebook's focus on metrics and visualizations ensures actionable insights, bridging technical modeling with practical interpretation.
