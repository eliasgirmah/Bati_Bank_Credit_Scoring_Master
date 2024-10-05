# Bati Bank Credit Scoring using Machine Learning Model

Bati Bank, a prominent financial services provider, is partnering with an eCommerce company to introduce a buy-now-pay-later service. This initiative requires the development of a robust Credit Scoring Model that will evaluate customers' creditworthiness and assign a credit score. The model is expected to predict the likelihood of default and help determine optimal loan amounts and terms.
The credit scoring system is designed to:
1.	Identify high-risk vs. low-risk customers based on transaction history.
2.	Predict default risk for potential customers.
3.	Assign credit scores based on estimated default probabilities.
4.	Optimize loan amounts and durations based on customer risk profiles.


## Project Directory Structure

The repository is organized into the following directories:

`.github/workflows/`: Contains configurations for GitHub Actions, enabling continuous integration and automated testing.

`.vscode/`: Configuration files for the Visual Studio Code editor, optimizing the development environment.

`notebooks/`: Jupyter notebooks used for tasks such as data exploration, feature engineering, and preliminary modeling.

`src/`: Python scripts for data preprocessing, feature extraction, and the implementation of the credit scoring model.

`tests/`: Unit tests to ensure the correctness and robustness of the implemented model and data processing logic.
`api`: Contains the implementation of the machine learning model API, allowing interaction with the model through RESTful endpoints.


## Installation Instructions

To run the project locally, follow these steps:

1. Clone the Repository:
>>>>
    git clone https://github.com/eliasgirmah/Bati_Bank_Credit_Scoring_Master.git`

    cd bati-bank-credit-scoring
>>>>

2. Set up the Virtual Environment:

Create a virtual environment to manage the project's dependencies:

>>>
    python -m venv .venv
    .venv\Scripts\activate
>>>

3. Install Dependencies:

Install the required Python packages by running:
>>>
    pip install -r requirements.txt
>>>


