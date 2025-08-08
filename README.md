# Time Series Analysis and Sales Forecasting Dashboard

This project is an interactive web application built with Streamlit that allows users to upload their time-series data, perform sales forecasting using an LSTM model, and interact with their data through a conversational chatbot.

## Features

-   **CSV Data Upload**: Easily upload your own time-series data in CSV format.
-   **Interactive Dashboard**:
    -   Key Performance Indicators (KPIs) for historical and forecasted data.
    -   Interactive visualizations for historical sales, forecasted sales, and a combined view using Plotly.
-   **LSTM-Based Forecasting**:
    -   Generates future sales predictions based on historical data.
    -   Users can specify the number of future periods to predict.
-   **Model Performance Analysis**:
    -   A dedicated tab to evaluate the forecasting model's performance.
    -   Metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Model Accuracy.
    -   Visual comparison of actual vs. predicted values on the test set.
-   **Data Preview**:
    -   View the original uploaded data, the aggregated data used for modeling, and the generated forecast data.
-   **Conversational Chatbot**:
    -   Ask questions about your dataset in natural language.
    -   Supports queries for basic statistics (sum, average, min, max, count) and advanced statistics (correlation, standard deviation, summary).

## Setup and Installation

To get this project up and running on your local machine, follow these steps.

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The `requirements.txt` file contains all the necessary Python libraries. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run data.py
```

Your web browser should automatically open with the application running.

## How to Use

1.  **Upload Data**: Use the file uploader in the sidebar to upload your CSV file.
2.  **Configure Controls**:
    -   Select the column from your data that represents the **X-axis (date/time)**.
    -   Select the column that represents the **Y-axis (value)** you want to forecast.
    -   Enter the **number of future periods** you wish to predict.
3.  **Generate Dashboard**: Click the "Generate Dashboard" button. The application will process the data, train the model, and display the dashboard.
4.  **Explore Tabs**:
    -   **Dashboard**: View KPIs and visualizations.
    -   **Data Preview**: Inspect the raw and processed dataframes.
    -   **Model Performance**: Analyze the accuracy of the forecast.
5.  **Ask Questions**: Use the chatbot in the sidebar to ask questions about your original dataset (e.g., "what is the average of sales?").

## Dependencies

The project relies on the following major Python libraries: `streamlit`, `pandas`, `plotly`, `numpy`, `tensorflow`, and `scikit-learn`.