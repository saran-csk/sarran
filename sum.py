import os
import glob

# Path to the directory containing images
image_dir = r"C:\Users\K Saran\Downloads\project python\image.png"

# List all files in the directory


# Use glob to find all PNG images in the directory
image_paths = glob.glob(os.path.join(image_dir, "*.png"))
print("Image paths found:", image_paths)

import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


import cv2
import pytesseract
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from PIL import Image
import os

# Ensure Tesseract is configured correctly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract data from image
def extract_data_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Debug: Print the extracted text
    print(f"Extracted text from {image_path}:")
    print(text)

    # Extract date and price from text (example format: '2023-06-01 1.1234')
    data = []
    for line in text.split('\n'):
        if line:
            try:
                date_str, price = line.split()
                date = pd.to_datetime(date_str)
                price = float(price)
                data.append([date, price])
            except ValueError:
                continue

    return pd.DataFrame(data, columns=['Date', 'Price'])

# Train the model
def train_model(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    
    X = data['Date_ordinal'].values.reshape(-1, 1)
    y = data['Price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Predict future prices
def predict_future_prices(model, last_date, minutes=5):
    future_dates = pd.date_range(start=last_date, periods=minutes, freq='T')
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    predicted_prices = model.predict(future_dates_ordinal)
    
    return future_dates, predicted_prices

# Main function
def main():
    # Path to the single image file
    image_path = r"C:\Users\K Saran\Downloads\project python\image.png"

    # Check if the image file exists
    if not os.path.isfile(image_path):
        print("The specified image file does not exist. Please check the path and try again.")
        return

    # Extract data from the image
    data = extract_data_from_image(image_path)

    # Check if data is extracted correctly
    if 'Date' not in data.columns:
        print("No 'Date' column found in the extracted data. Please check the data extraction process.")
        return

    # Ensure data is sorted by date
    data = data.sort_values(by='Date')

    # Train the model
    model = train_model(data)

    # Predict future prices
    last_date = data['Date'].max()
    future_dates, predicted_prices = predict_future_prices(model, last_date)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Price'], label='Historical Prices')
    plt.plot(future_dates, predicted_prices, label='Predicted Prices (Next 5 Minutes)', linestyle='--', color='r')
    plt.title('Trading Chart with Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
