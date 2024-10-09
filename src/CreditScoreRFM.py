import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


# Load your dataset
df = pd.read_csv('./data/transformed_data.csv')  # Replace with your transformed data file


df.head()

import pandas as pd
import numpy as np
from datetime import datetime

class CreditScoreRFM:
    def __init__(self, rfm_data):
        self.rfm_data = rfm_data

    def calculate_rfm(self):
        """
        Calculates Recency, Frequency, and Monetary values for each customer.
        """
        # Convert 'TransactionStartTime' to datetime
        self.rfm_data['TransactionStartTime'] = pd.to_datetime(self.rfm_data['TransactionStartTime'])

        # Set the end date as the latest transaction date
        end_date = self.rfm_data['TransactionStartTime'].max()

        # Calculate Recency (days since the last transaction for each customer)
        self.rfm_data['Last_Transaction_Date'] = self.rfm_data.groupby('CustomerId')['TransactionStartTime'].transform('max')
        self.rfm_data['Recency'] = (end_date - self.rfm_data['Last_Transaction_Date']).dt.days

        # Calculate Frequency (number of transactions for each customer)
        self.rfm_data['Frequency'] = self.rfm_data.groupby('CustomerId')['TransactionId'].transform('count')

        # Calculate Monetary (sum of 'Amount_to_Value_Ratio' for each customer)
        self.rfm_data['Monetary'] = self.rfm_data.groupby('CustomerId')['Amount_to_Value_Ratio'].transform('sum')

        # Create a summary DataFrame for scoring
        rfm_summary = self.rfm_data[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()

        return rfm_summary
    
    def calculate_rfm_scores(self, rfm_data):
        """
        Calculates RFM scores based on the Recency, Frequency, and Monetary values.
        """
        # Quartile-based scoring for Recency (lower is better)
        rfm_data['r_quartile'] = pd.qcut(rfm_data['Recency'], 4, labels=['4', '3', '2', '1'])  
        
        # Quartile-based scoring for Frequency (higher is better)
        rfm_data['f_quartile'] = pd.qcut(rfm_data['Frequency'], 4, labels=['1', '2', '3', '4'])  
        
        # Quartile-based scoring for Monetary (higher is better)
        rfm_data['m_quartile'] = pd.qcut(rfm_data['Monetary'], 4, labels=['1', '2', '3', '4'])  

        # Calculate overall RFM Score
        rfm_data['RFM_Score'] = (rfm_data['r_quartile'].astype(int) * 0.1 +
                                  rfm_data['f_quartile'].astype(int) * 0.45 +
                                  rfm_data['m_quartile'].astype(int) * 0.45)

        return rfm_data

    def assign_label(self, rfm_data):
        """
        Assign 'Good' or 'Bad' based on the RFM Score threshold.
        """
        high_threshold = rfm_data['RFM_Score'].quantile(0.75)
        low_threshold = rfm_data['RFM_Score'].quantile(0.5)
        rfm_data['Risk_Label'] = rfm_data['RFM_Score'].apply(lambda x: 'Good' if x >= low_threshold else 'Bad')
        return rfm_data


# Initialize the RFM class with the dataset
rfm = CreditScoreRFM(df)

# Calculate RFM values
rfm_data = rfm.calculate_rfm()

# Display the first few rows of the RFM data
rfm_data.head()


# Calculate RFM scores
rfm_data_with_scores = rfm.calculate_rfm_scores(rfm_data)

# Display the RFM scores
rfm_data_with_scores.head()


# Assign risk labels
rfm_data_with_labels = rfm.assign_label(rfm_data_with_scores)

# Display the RFM data with risk labels
rfm_data_with_labels.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot for RFM Scores
sns.pairplot(rfm_data_with_labels[['Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Risk_Label']], hue='Risk_Label')
plt.show()

# Histogram for RFM Scores
sns.histplot(rfm_data_with_labels['RFM_Score'], bins=20, kde=True)
plt.title('Distribution of RFM Scores')
plt.show()



from joblib import dump, load

# Save the model to a file
dump(rfm, './model/credit_score_rfm.joblib')








