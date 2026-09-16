# Fraud Detection & Real-Time Risk Scoring System
An end-to-end machine learning system for detecting fraudulent credit card transactions and generating transaction risk scores through a REST API.
## Project Overview
Credit card fraud detection is an imbalanced classification problem where fraudulent transactions represent a very small percentage of all transactions.
This project builds an end-to-end fraud detection pipeline that covers:- Data cleaning- Exploratory Data Analysis (EDA)- Feature engineering- Class imbalance handling- Machine learning model training- Model comparison- Cross-validation- Hyperparameter tuning- Fraud probability estimation- Risk scoring- FastAPI deployment- Docker containerization- Docker Compose deployment- Automated testing- GitHub Actions CI
## Tech Stack- Python- Pandas- NumPy- Scikit-learn- Matplotlib- Seaborn- Joblib- FastAPI- Pydantic- Uvicorn- Pytest- Docker- Docker Compose- Git- GitHub- GitHub Actions
## Dataset
The project uses the Credit Card Fraud Detection dataset from Kaggle.
Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
The dataset contains anonymized transaction features including:- `Time`- `V1` to `V28`- `Amount`- `Class`
Where:- `Class = 0` fi Legitimate transaction- `Class = 1` fi Fraudulent transaction
The dataset is highly imbalanced, making precision, recall, F1-score, and PR-AUC important evaluation metrics.
The dataset itself is not included in this repository because of its size.
## Machine Learning Pipeline
```text
Raw Transaction Data
Data Cleaning
Exploratory Data Analysis
Feature Engineering
Train/Test Split
Class Imbalance Handling
Model Training
Cross-Validation
Hyperparameter Tuning
Model Evaluation
Final Random Forest Model
Risk Scoring
FastAPI
Docker
Docker Compose
```
## Feature Engineering
Additional time and transaction-related features were created:- `Hour`- `Day`- `Hour_sin`- `Hour_cos`- `Amount_log`
The original anonymized features `V1`–`V28` were retained.
The dataset does not contain customer IDs, merchant IDs, or device IDs, so customer-level or merchant-level behavioral features were not created.
## Class Imbalance
Fraudulent transactions represent a very small proportion of the dataset.
To address this imbalance, class weighting was used during model training.
The Random Forest model used class weights so that fraudulent transactions received substantially higher importance during training.
## Models Evaluated
The following classification algorithms were evaluated.
### Random Forest
Test performance:- Precision: 95.83%- Recall: 72.63%- F1-score: 82.63%- ROC-AUC: 93.79%- PR-AUC: 80.84%
### Support Vector Machine
Test performance:- Precision: 53.28%- Recall: 68.42%- F1-score: 59.91%
### Gradient Boosting
Test performance:- Precision: 18.37%- Recall: 83.16%- F1-score: 30.10%
Model evaluation focused on the trade-off between detecting fraudulent transactions and limiting false fraud alerts.
## Cross-Validation
Stratified 5-fold cross-validation was used to evaluate model performance across multiple training and validation splits.
Random Forest cross-validation results:- Mean Precision: 92.94%- Mean Recall: 78.57%- Mean F1-score: 84.99%
SVM cross-validation results:
- Mean Precision: 60.18%- Mean Recall: 71.16%- Mean F1-score: 65.08%
Gradient Boosting cross-validation results:- Mean Precision: 22.12%- Mean Recall: 88.36%- Mean F1-score: 35.31%
## Hyperparameter Tuning
Random Forest hyperparameters were optimized using `GridSearchCV`.
The selected parameters were:
text
n_estimators = 100
max_depth = 20
min_samples_split = 2
min_samples_leaf = 1

The best cross-validation F1-score was approximately: 85.93%
## Model Evaluation
The final tuned Random Forest model was evaluated on the test set.
| Metric    | Score |
| Precision | 95.83% |
| Recall    | 72.63% |
| F1-score  | 82.63% |
| ROC-AUC   | 93.79% |
| PR-AUC    | 80.84% |
Because fraud detection is a highly imbalanced classification problem, accuracy was not treated as the main evaluation metric.
Precision, recall, F1-score, ROC-AUC, and PR-AUC were used to better understand model performance.
## Risk Scoring
The trained Random Forest model produces a model-estimated fraud probability.
This probability is converted into a risk score:

Risk Score = Fraud Probability × 100

The application currently maps the score into three application-level risk categories:0–29   
fi Low Risk
30–69  fi Medium Risk
70–100 fi High Risk
```
The application also returns a decision:

Allow Transaction
Review Transaction
Block Transaction
```
These thresholds are project-level rules and are not universal financial industry standards.
## API
The project provides a REST API using FastAPI.
### Health Check
```text
GET /health
```
Example response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
### Fraud Prediction
```text
POST /predict
```
The endpoint accepts transaction features and returns the prediction, fraud probability, risk score, risk level, and decision.
Example response:
```json
{
  "prediction": 1,
  "fraud_probability": 0.5498,
  "risk_score": 54.98,
  "risk_level": "Medium Risk",
  "decision": "Review Transaction"
}
```
## API Input
The prediction endpoint accepts:- `Time`- `V1` to `V28`- `Amount`
The API validates input values using Pydantic.
For example, negative transaction amounts are rejected by the API validation layer.
## Project Structure
```text
fraud-detection-risk-scoring/

 api/
    __init__.py
    main.py
    model_service.py
    preprocessing.py

 models/
   feature_names.pkl
   random_forest_fraud_model.pkl

 notebook/
   fraud_detection_eda_feature_engineering.ipynb
   model_training.ipynb

 tests/
   test_api.py

.github/
    workflows/
    
    ci.yml

.dockerignore
.gitignore
Dockerfile
docker-compose.yml
pytest.ini
README.md
requirements.txt

## Project Components
### `api/main.py`
Defines the FastAPI application, API routes, request model, and input validation.
### `api/preprocessing.py`
Contains the feature engineering logic used by the API before prediction.
### `api/model_service.py`
Loads the trained model and feature names, performs prediction, calculates the fraud probability and risk score, and generates the application-level risk decision.
### `models/`
Contains the serialized trained Random Forest model and the feature-name list required during inference.
### `tests/`
Contains automated tests for the API.
### `.github/workflows/ci.yml`
Defines the GitHub Actions continuous integration workflow that runs tests and builds the Docker image.
### `Dockerfile`
Defines how the FastAPI application is packaged into a Docker image.
### `docker-compose.yml`
Defines the local deployment configuration for running the API as a Docker Compose service.
## Running the Project Locally
### 1. Clone the repository
```bash
git clone https://github.com/mahi655/fraud-detection-risk-scoring.git
cd fraud-detection-risk-scoring
```
### 2. Create a virtual environment
```
python -m venv .venv
```
### 3. Activate the environment
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
### 4. Install dependencies
```
pip install -r requirements.txt
```
## Run the API Without Docker
```
uvicorn api.main:app --reload
```
The API will be available at:
```text
http://127.0.0.1:8000
```
Interactive API documentation:
```text
http://127.0.0.1:8000/docs
```
## API Usage
Open the Swagger documentation:
```text
http://127.0.0.1:8000/docs
```
Use the `POST /predict` endpoint to send a transaction and receive a real-time risk assessment.
The API response contains:
```text
Prediction
Fraud Probability
Risk Score
Risk Level
Decision
```
## Testing
Run the automated API tests:
```bash
pytest -v
```
The test suite verifies:- Root endpoint- Health endpoint- Valid prediction request- Invalid transaction input
## Docker Deployment
The application can be packaged and deployed locally using Docker.
Build the Docker image:
```
docker build -t fraud-detection-api .
```
Run the container:
```
docker run -d -p 8000:8000 --name fraud-detection-container fraud-detection-api
```
Check the running container:
```
docker ps
```
## Docker Compose Deployment
The project also includes Docker Compose for easier local deployment.
Build and start the application:
```
docker compose up -d --build
```
Check the running service:
```
docker compose ps
```
Stop the application:
```
docker compose down
```
The API will be available at:
```text
http://127.0.0.1:8000
```
Swagger documentation:
```text
http://127.0.0.1:8000/docs
```
## Continuous Integration
GitHub Actions automatically runs the CI pipeline when changes are pushed to `main` or when a pull request is created.
The pipeline performs:

GitHub Push / Pull Request
Checkout Repository
Set Up Python
Install Dependencies
Run Pytest
Build Docker Image
```
This helps verify that the automated tests pass and that the Docker image can be built successfully.
## Git and GitHub Workflow
The project uses Git for version control and GitHub for repository management.
The development workflow includes:

Local Development
Git Commit
GitHub
Pull Request
GitHub Actions
Tests + Docker Build
Merge to main

## Deployment Architecture
The application is deployed locally using a production-style containerized setup:

Client
FastAPI REST API
Model Service
Feature Engineering
Random Forest Model
Fraud Probability
Risk Score
Risk Level + Decision
```
The application runs inside Docker and is managed locally using Docker Compose.
No cloud deployment is required for this project.
## Key Learning Outcomes
Through this project, the following concepts were implemented:- Imbalanced classification- Fraud detection- Feature engineering- Class weighting- Precision/Recall trade-offs- F1-score- ROC-AUC- PR-AUC- Stratified cross-validation- Hyperparameter tuning- Model serialization- REST API development- Input validation- Automated testing- Docker
- Docker Compose- Git- GitHub- GitHub Actions- Continuous Integration
## Future Improvements
Possible future improvements include:- Probability calibration- Validation-based decision threshold selection- More advanced anomaly detection methods- Database integration- Authentication and authorization- Logging and monitoring- Model versioning- Real-time transaction streaming- Model monitoring and retraining