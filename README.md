# Crop Price Prediction using Random Forest ML

This project demonstrates the use of **Random Forest Regression** to predict crop prices based on key agricultural features. The web application, built with **Flask**, allows users to upload their datasets and visualize the model's predictions through metrics like **R²**, **Mean Squared Error (MSE)**, and a **scatter plot** comparing actual vs. predicted values.

## How the Project Works

### Backend (Flask and Machine Learning):
1. **File Upload Handling**:
   - Users upload their crop dataset through the HTML form.
   - The uploaded file is saved to the `static/uploads/` directory.

2. **Data Validation**:
   - The application checks the dataset for required columns: `Month`, `Year`, `Rainfall`, and `WPI`.

3. **Data Preprocessing**:
   - The features (`Month`, `Year`, `Rainfall`) are extracted as inputs for the model.
   - The target variable is `WPI` (Wholesale Price Index).
   - The dataset is split into training and testing sets using an 80-20 split ratio.

4. **Random Forest Model**:
   - A **Random Forest Regressor** is trained on the training data with 100 estimators and a fixed random state for reproducibility.
   - The trained model is used to predict crop prices (`WPI`) on the test data.

5. **Evaluation and Metrics**:
   - The model's performance is evaluated using:
     - **R² (R-Squared)**: Measures how well the model explains the variability of the target.
     - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.

6. **Visualization**:
   - A scatter plot is generated to visualize the relationship between actual and predicted prices.
   - The plot includes an ideal fit line for reference.
   - The plot is saved as `scatter_plot.png` in the `static/` directory.

7. **Results Rendering**:
   - The results page displays:
     - The R² score.
     - The Mean Squared Error.
     - The scatter plot of actual vs. predicted values.
     - A list of predicted prices.

### Frontend (HTML Templates):
1. **Index Page (`index.html`)**:
   - Provides a file upload form with clear instructions.
   - Displays error messages if the uploaded file is invalid.

2. **Results Page (`results.html`)**:
   - Displays the model evaluation metrics (R², MSE).
   - Shows the scatter plot of actual vs. predicted values.
   - Lists the predicted prices for user reference.

## Features
- **File Upload**: Upload CSV files containing crop-related data.
- **Data Validation**: Ensures the dataset includes required features such as `Month`, `Year`, `Rainfall`, and `WPI`.
- **Random Forest Regression**:
  - Trains the model on user-provided data.
  - Predicts crop prices (`WPI`).
  - Evaluates model performance using R² and MSE.
- **Visualization**:
  - Scatter plot of actual vs. predicted prices.
  - Metrics displayed for easy interpretation.

## Technologies Used
- **Python**:
  - `Flask`: Backend framework for handling file uploads and serving the web app.
  - `scikit-learn`: Machine learning library for Random Forest implementation.
  - `pandas`: Data manipulation and preprocessing.
  - `matplotlib`: Visualization library for generating scatter plots.
- **HTML/CSS**: Frontend for user interaction.


