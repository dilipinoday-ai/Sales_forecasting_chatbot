import streamlit as st
import pandas as pd
import plotly.graph_objects as go
  # Holt-Winters import removed
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import base64

st.set_page_config(page_title="Time Series Analysis and Sales Forecasting", layout="wide")

@st.cache_data
def generate_forecast_data(_df, x_axis, y_axis, periods):
    """
    Caches the entire forecasting process to prevent re-training the model on every interaction.
    """
    # Prepare data
    df_agg = _df.copy()
    df_agg[x_axis] = pd.to_datetime(df_agg[x_axis], errors='coerce')
    df_agg = df_agg.dropna(subset=[x_axis, y_axis])
    if pd.api.types.is_numeric_dtype(df_agg[y_axis]):
        df_agg = df_agg.groupby(x_axis)[y_axis].sum().reset_index()
    else:
        df_agg = df_agg[[x_axis, y_axis]].drop_duplicates()

    ts = df_agg.set_index(x_axis)[y_axis].astype(float)
    freq = pd.infer_freq(ts.index)
    if freq is None:
        freq = 'D'

    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    lookback = min(10, len(ts_scaled)-1) if len(ts_scaled) > 1 else 1
    X, y = [], []
    for i in range(lookback, len(ts_scaled)):
        X.append(ts_scaled[i - lookback:i, 0])
        y.append(ts_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if X.shape[0] == 0:
        raise ValueError("Not enough data for LSTM forecasting.")
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split for evaluation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(lookback, 1))) # Changed activation to tanh for better performance
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Evaluation and Forecasting
    mae, rmse = 0, 0
    y_test_inv, y_pred_inv = np.array([]), np.array([])
    if len(X_test) > 0:
        y_pred_test = model.predict(X_test, verbose=0)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred_test).flatten()
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    model.fit(X, y, epochs=50, batch_size=32, verbose=0) # Retrain on full data
    last_seq = ts_scaled[-lookback:].reshape(1, lookback, 1)
    
    # --- Corrected iterative forecasting loop ---
    forecast_scaled = []
    current_batch = last_seq.copy()
    for _ in range(periods):
        next_pred_scaled = model.predict(current_batch, verbose=0)[0, 0]
        if not np.isfinite(next_pred_scaled):
            next_pred_scaled = 0.0  # Prevent inf/nan from propagating
        forecast_scaled.append(next_pred_scaled)
        current_batch = np.append(current_batch[:, 1:, :], [[[next_pred_scaled]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    forecast = np.where(np.isfinite(forecast), forecast, np.nan)
    hist_df = pd.DataFrame({x_axis: ts.index, 'Value': ts.values, 'Type': 'Historical'})
    forecast_index = pd.date_range(start=ts.index[-1], periods=periods + 1, freq=freq)[1:]
    forecast_df = pd.DataFrame({x_axis: forecast_index, 'Value': forecast, 'Type': 'Forecast'})
    test_dates = ts.index[lookback + split_idx:] if len(X_test) > 0 else pd.to_datetime([])

    return df_agg, hist_df, forecast_df, mae, rmse, y_test_inv, y_pred_inv, test_dates, len(X_test) > 0

def get_image_as_base64(path: str) -> str:
    """Reads an image file and returns it as a base64 encoded string."""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Logo file not found at {path}. Please make sure it's in the same folder as the script.")
        return None
    except PermissionError:
        st.error(f"Permission denied for logo file at '{path}'. Please check that it is a file and not a directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the logo file: {e}")
        return None

# --- HEADER ---
# Path to your local logo file. Make sure it's in the same directory as this script.
LOGO_PATH = "logo.png"  # Assumes your logo is named logo.png
logo_base64 = get_image_as_base64(LOGO_PATH)

if logo_base64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" width="200">
            <h1 style="margin-left: 20px;">📊 Time Series Analysis and Sales Forecasting</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("📊 Time Series Analysis and Sales Forecasting")

# --- SIDEBAR FOR CONTROLS & CHAT ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

    # Initialize session state
    if "show_dashboard" not in st.session_state:
        st.session_state.show_dashboard = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Conditional controls and chat based on file upload
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        columns = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis (date/time) column", columns)
        y_axis = st.selectbox("Select Y-axis (value) column", columns)
        periods = st.number_input("Number of future periods to predict", min_value=1, max_value=10000, value=10)

        if st.button("Generate Dashboard"):
            st.session_state.show_dashboard = True
            st.session_state.chat_history = [] # Reset chat on new dashboard

        st.divider()

        st.header("🤖 Ask a Question")
        # The chatbot should use the original, un-aggregated dataframe
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        # This is the new chat input box
        if prompt := st.chat_input("Ask a question about your data..."):
            # Add user message to chat history and display it
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display bot response
            prompt_lower = prompt.lower()
            with st.chat_message("assistant"):
                responses = []
                mentioned_cols = [col for col in numeric_columns if col.lower() in prompt_lower]

                # --- Handle special cases first ---
                if "correlation" in prompt_lower or "corr" in prompt_lower:
                    if len(mentioned_cols) == 2:
                        col1, col2 = mentioned_cols
                        correlation = df[col1].corr(df[col2])
                        responses.append(f"The correlation between **{col1}** and **{col2}** is **{correlation:.4f}**.")
                    else:
                        responses.append("To calculate correlation, please mention exactly two numeric columns in your question.")
                elif "describe" in prompt_lower or "summary" in prompt_lower or "statistics" in prompt_lower:
                    if not mentioned_cols:
                        responses.append(f"Please mention a numeric column to describe. Available columns: {', '.join(numeric_columns)}")
                    else:
                        for col in mentioned_cols:
                            stats = df[col].describe()
                            try:
                                # Use to_markdown for a nice table format if tabulate is installed
                                stats_table = stats.to_frame().to_markdown()
                            except ImportError:
                                # Fallback to a simpler string representation if tabulate is not installed
                                st.warning("For a nicer table format, please install the 'tabulate' library: `pip install tabulate`")
                                stats_table = stats.to_frame().to_string()
                            responses.append(f"Here is a statistical summary for **{col}**:\n\n```\n{stats_table}\n```")
                else:
                    # --- Handle standard operations ---
                    operations = {
                        "average": "mean", "mean": "mean", "avg": "mean",
                        "sum": "sum", "total": "sum",
                        "min": "min", "minimum": "min",
                        "max": "max", "maximum": "max",
                        "count": "count", "how many": "count",
                        "median": "median",
                        "standard deviation": "std", "std": "std", "volatility": "std",
                        "unique count": "nunique", "distinct count": "nunique", "number of unique": "nunique"
                    }
                    mentioned_ops = [op for op in operations if op in prompt_lower]
                    if mentioned_ops and mentioned_cols:
                        # Handle ambiguous requests with multiple operations and columns
                        if len(mentioned_ops) > 1 and len(mentioned_cols) > 1:
                            responses.append(
                                "Your request is ambiguous. Please ask for either multiple metrics for one column "
                                "(e.g., 'what is the sum and mean of sales?') or one metric for multiple columns "
                                "(e.g., 'what is the sum of sales and profit?')."
                            )
                        else:
                            for col in mentioned_cols:
                                for op_keyword in mentioned_ops:
                                    op_func = operations[op_keyword]
                                    result = getattr(df[col], op_func)()
                                    if op_func in ['count', 'nunique']:
                                        responses.append(f"The {op_keyword} of **{col}** is **{result:,}**.")
                                    else:
                                        responses.append(f"The {op_keyword} of **{col}** is **{result:,.2f}**.")

                if not responses:
                    response = (f"Sorry, I can only answer questions about:\n- **Basic stats:** average, sum, min, max, count, median\n"
                                f"- **Advanced stats:** standard deviation, unique count, summary, correlation\n\n"
                                f"Please mention an operation and a numeric column. Available columns: **{', '.join(numeric_columns)}**.")
                else:
                    response = "\n\n".join(responses)

                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.divider()

        # Display chat history using the new chat elements
        st.header("📜 Chat History")
        # The most recent exchange is displayed above when submitted.
        # This loop shows the history *before* that last exchange.
        for chat in st.session_state.chat_history[:-2]:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

# --- MAIN CONTENT AREA ---
if not uploaded_file:
    st.info("Please upload a CSV file using the sidebar to get started.")
else:
    if st.session_state.show_dashboard:
        try:
            # Call the cached function to get stable results
            df_agg, hist_df, forecast_df, mae, rmse, y_test_inv, y_pred_inv, test_dates, has_test_data = generate_forecast_data(df, x_axis, y_axis, periods)
            
            # --- CREATE TABS FOR DISPLAY ---
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📄 Data Preview", "📈 Model Performance"])

            with tab1: # Dashboard
                st.header("Key Metrics")
                total_hist = hist_df['Value'].sum()
                avg_hist = hist_df['Value'].mean()
                finite_forecast = forecast_df['Value'][np.isfinite(forecast_df['Value'])]
                total_forecast = finite_forecast.sum() if not finite_forecast.empty else 0.0
                avg_forecast = finite_forecast.mean() if not finite_forecast.empty else 0.0
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Historical Sales", f"{total_hist:,.2f}")
                kpi2.metric("Average Historical Sales", f"{avg_hist:,.2f}")
                kpi3.metric("Total Forecasted Sales", f"{total_forecast:,.2f}")
                kpi4.metric("Average Forecasted Sales", f"{avg_forecast:,.2f}")

                st.header("Historical Sales Visualization")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=hist_df[x_axis], y=hist_df['Value'], mode='lines', name='Historical', line=dict(color='teal')))
                fig_hist.update_layout(title="Historical Sales Over Time", xaxis_title=x_axis, yaxis_title='Value')
                st.plotly_chart(fig_hist, use_container_width=True)

                st.header("Forecasted Sales Visualization")
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=forecast_df[x_axis], y=forecast_df['Value'], mode='lines', name='Forecast', line=dict(color='blue')))
                fig_forecast.update_layout(title="Forecasted Sales Over Time", xaxis_title=x_axis, yaxis_title='Value')
                st.plotly_chart(fig_forecast, use_container_width=True)

                st.header("Historical vs. Forecasted Sales")
                fig_combined = go.Figure()
                fig_combined.add_trace(go.Scatter(x=hist_df[x_axis], y=hist_df['Value'], mode='lines', name='Historical', line=dict(color='teal')))
                fig_combined.add_trace(go.Scatter(x=forecast_df[x_axis], y=forecast_df['Value'], mode='lines', name='Forecast', line=dict(color='orange')))
                fig_combined.update_layout(title="Historical vs. Forecasted Sales Over Time", xaxis_title=x_axis, yaxis_title='Value')
                st.plotly_chart(fig_combined, use_container_width=True)

            with tab2: # Data Preview
                st.header("Data Preview")
                st.dataframe(df, use_container_width=True)
                st.header("Aggregated Data for Visualization")
                st.dataframe(df_agg, use_container_width=True)
                st.header("Forecasted Data")
                st.dataframe(forecast_df, use_container_width=True)

            with tab3: # Model Performance
                st.header("Model Evaluation Metrics (on Test Set)")
                if has_test_data:
                    # --- Calculations for new metrics ---
                    avg_hist_value = hist_df['Value'].mean()
                    percentage_error = (mae / avg_hist_value) * 100 if avg_hist_value > 0 else 0
                    model_accuracy = 100 - percentage_error

                    # --- Display all metrics in a 2x2 grid ---
                    col1, col2 = st.columns(2)
                    col1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
                    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
                    col1.metric("Average Percentage Error", f"{percentage_error:.2f}%")
                    col2.metric("Model Accuracy", f"{model_accuracy:.2f}%")
                    
                    st.header("Model Performance on Test Set")
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=test_dates, y=y_test_inv, mode='lines', name='Actual Values', line=dict(color='teal')))
                    fig_perf.add_trace(go.Scatter(x=test_dates, y=y_pred_inv, mode='lines', name='Predicted Values', line=dict(color='orange', dash='dash')))
                    fig_perf.update_layout(title="Model Predictions vs. Actual Values on Test Data", xaxis_title=x_axis, yaxis_title='Value')
                    st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.warning("Not enough data to create a test set for evaluation.")

        except Exception as e:
            st.error(f"An error occurred during dashboard generation: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;padding-top: 10px;border-top: 1px solid #d6d6d8;'>© Copyright 2007-2025. inoday - All Rights Reserved | Privacy Policy An ISO 9001:2015 Certified Company</p>", unsafe_allow_html=True)