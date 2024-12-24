from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

# Initialize the Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = os.path.join('static', 'uploads')  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Folder for saving plot images
PLOT_FOLDER = os.path.join('static', 'images')  
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_and_run_ml():
    if request.method == 'POST':
       
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template('index.html', error="No file selected.")
        
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save the uploaded file

        # Load the dataset
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            return render_template('index.html', error=f"Error reading the file: {e}")

        # Validate columns in the dataset
        required_columns = ['Month', 'Year', 'Rainfall', 'WPI']
        if not all(col in data.columns for col in required_columns):
            return render_template('index.html', error="File must contain columns: Month, Year, Rainfall, WPI.")
        
        # Features and target
        X = data[['Month', 'Year', 'Rainfall']]
        y = data['WPI']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and metrics
        predicted_prices = model.predict(X_test)
        r_squared = r2_score(y_test, predicted_prices)
        mse = mean_squared_error(y_test, predicted_prices)

        # Generate scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predicted_prices, alpha=0.7, label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual WPI')
        plt.ylabel('Predicted WPI')
        plt.title('Actual vs Predicted WPI')
        plt.legend()
        
        plot_filename = 'scatter_plot.png'
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"Scatter plot saved at: {plot_path}")  

        
        predicted_prices_list = predicted_prices.tolist()

        # Render results page
        return render_template(
            'results.html',
            r_squared=r_squared,
            mse=mse,
            plot_image=f'images/{plot_filename}', 
            predicted_prices=json.dumps(predicted_prices_list)  
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
