import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import base64
import datetime

st.set_page_config(page_title="Universal Data Analysis and Visualization", layout="wide")

@st.cache_data
def generate_forecast_data(_df, x_axis, y_axis, periods):
    """
    Caches the entire forecasting process for numerical time series data only.
    """
    # Prepare data
    df_agg = _df.copy()
    df_agg[x_axis] = pd.to_datetime(df_agg[x_axis], errors='coerce')
    df_agg = df_agg.dropna(subset=[x_axis, y_axis])
    df_agg = df_agg.groupby(x_axis)[y_axis].sum().reset_index()

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
    model.add(LSTM(50, activation='tanh', input_shape=(lookback, 1)))
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

    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    last_seq = ts_scaled[-lookback:].reshape(1, lookback, 1)
    
    forecast_scaled = []
    current_batch = last_seq.copy()
    for _ in range(periods):
        next_pred_scaled = model.predict(current_batch, verbose=0)[0, 0]
        if not np.isfinite(next_pred_scaled):
            next_pred_scaled = 0.0
        forecast_scaled.append(next_pred_scaled)
        current_batch = np.append(current_batch[:, 1:, :], [[[next_pred_scaled]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    forecast = np.where(np.isfinite(forecast), forecast, np.nan)
    hist_df = pd.DataFrame({x_axis: ts.index, 'Value': ts.values, 'Type': 'Historical'})
    forecast_index = pd.date_range(start=ts.index[-1], periods=periods + 1, freq=freq)[1:]
    forecast_df = pd.DataFrame({x_axis: forecast_index, 'Value': forecast, 'Type': 'Forecast'})
    test_dates = ts.index[lookback + split_idx:] if len(X_test) > 0 else pd.to_datetime([])

    return df_agg, hist_df, forecast_df, mae, rmse, y_test_inv, y_pred_inv, test_dates, len(X_test) > 0

def determine_chart_type(df, x_col, y_col):
    """
    Determines the appropriate chart type based on X and Y column data types.
    """
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])
    
    # More robust datetime detection for object/string columns
    if df[x_col].dtype == 'object' and not x_is_datetime:
        try:
            # Try to parse a sample of the data
            sample_size = min(100, len(df))
            sample_data = df[x_col].dropna().head(sample_size)
            parsed_dates = pd.to_datetime(sample_data, errors='coerce')
            # If more than 80% of sample can be parsed as dates, consider it datetime
            if parsed_dates.notna().sum() / len(sample_data) > 0.8:
                x_is_datetime = True
        except:
            x_is_datetime = False
    
    if x_is_datetime and y_is_numeric:
        return "time_series", "Time Series (with forecasting capability)"
    elif x_is_numeric and y_is_numeric:
        return "scatter", "Scatter Plot (correlation analysis)"
    elif not x_is_numeric and not y_is_numeric:
        return "heatmap", "Heatmap (categorical vs categorical)"
    elif not x_is_numeric and y_is_numeric:
        return "bar", "Bar Chart (categorical vs numerical)"
    elif x_is_numeric and not y_is_numeric:
        return "stacked_bar", "Stacked Bar Chart (numerical vs categorical)"
    else:
        return "scatter", "Default Scatter Plot"

def prepare_data_for_visualization(df, x_col, y_col, chart_type):
    """
    Prepares data based on the determined chart type.
    """
    df_viz = df[[x_col, y_col]].copy()
    df_viz = df_viz.dropna()
    
    if chart_type == "time_series":
        df_viz[x_col] = pd.to_datetime(df_viz[x_col], errors='coerce')
        df_viz = df_viz.dropna()
        if pd.api.types.is_numeric_dtype(df_viz[y_col]):
            df_viz = df_viz.groupby(x_col)[y_col].sum().reset_index()
    elif chart_type == "heatmap":
        # Create a contingency table for categorical vs categorical
        df_viz = pd.crosstab(df_viz[x_col], df_viz[y_col])
    elif chart_type == "bar":
        # Aggregate numerical values by categorical x
        df_viz = df_viz.groupby(x_col)[y_col].agg(['mean', 'sum', 'count']).reset_index()
    elif chart_type == "stacked_bar":
        # Count occurrences of categorical y for numerical x bins
        if df_viz[x_col].nunique() > 20:  # If too many unique values, create bins
            df_viz[f'{x_col}_binned'] = pd.cut(df_viz[x_col], bins=10)
            # Convert intervals to strings to avoid JSON serialization issues
            df_viz[f'{x_col}_binned'] = df_viz[f'{x_col}_binned'].astype(str)
            df_viz = df_viz.groupby([f'{x_col}_binned', y_col]).size().reset_index(name='count')
        else:
            df_viz = df_viz.groupby([x_col, y_col]).size().reset_index(name='count')
    
    return df_viz

def create_visualization(df_viz, x_col, y_col, chart_type, original_df):
    """
    Creates the appropriate visualization based on chart type.
    """
    if chart_type == "time_series":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_viz[x_col], y=df_viz[y_col], mode='lines+markers', 
                                name='Data', line=dict(color='teal')))
        fig.update_layout(title=f"{y_col} Over Time", xaxis_title=x_col, yaxis_title=y_col)
        
    elif chart_type == "scatter":
        fig = px.scatter(original_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        
        # Add correlation coefficient
        corr = original_df[x_col].corr(original_df[y_col])
        fig.add_annotation(text=f"Correlation: {corr:.3f}", 
                          xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False)
        
    elif chart_type == "heatmap":
        fig = px.imshow(df_viz, text_auto=True, aspect="auto", 
                       title=f"Relationship between {x_col} and {y_col}")
        
    elif chart_type == "bar":
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_viz[x_col], y=df_viz['mean'], name='Average',
                            marker_color='teal'))
        fig.update_layout(title=f"Average {y_col} by {x_col}", 
                         xaxis_title=x_col, yaxis_title=f'Average {y_col}')
        
    elif chart_type == "stacked_bar":
        if f'{x_col}_binned' in df_viz.columns:
            fig = px.bar(df_viz, x=f'{x_col}_binned', y='count', color=y_col,
                        title=f"Distribution of {y_col} across {x_col} ranges")
        else:
            fig = px.bar(df_viz, x=x_col, y='count', color=y_col,
                        title=f"Distribution of {y_col} by {x_col}")
    else:
        fig = px.scatter(original_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    
    return fig

def can_forecast(df, x_col, y_col, chart_type):
    """
    Determines if forecasting is possible for the given data combination.
    """
    if chart_type == "time_series":
        return True
    elif chart_type in ["scatter", "bar"] and pd.api.types.is_numeric_dtype(df[y_col]):
        # Can forecast if we have enough data points and Y is numerical
        return len(df) > 20
    else:
        return False


import numpy as np

def generate_non_timeseries_forecast(df, x_col, y_col, chart_type, forecast_points=10):
    """
    Generates forecasts for non-time-series data using trend analysis or regression.
    """
    if chart_type == "scatter" and pd.api.types.is_numeric_dtype(df[x_col]):
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        df_clean = df[[x_col, y_col]].dropna()
        X = df_clean[x_col].values.reshape(-1, 1)
        y = df_clean[y_col].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast points
        x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
        x_range = x_max - x_min
        forecast_x = np.linspace(x_max, x_max + x_range * 0.3, forecast_points)
        forecast_y = model.predict(forecast_x.reshape(-1, 1))
        
        return pd.DataFrame({x_col: forecast_x, 'Forecast': forecast_y}), model.score(X, y)
    
    elif chart_type == "bar":
        # For categorical X vs numerical Y, predict for new categories or extrapolate trends
        df_clean = df[[x_col, y_col]].dropna()
        category_means = df_clean.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        
        # Simple trend-based forecast for top categories
        trend = np.polyfit(range(len(category_means)), category_means.values, 1)
        
        # Forecast next few periods assuming category performance follows trend
        forecast_values = []
        for i in range(forecast_points):
            next_val = np.polyval(trend, len(category_means) + i)
            forecast_values.append(max(0, next_val))  # Ensure non-negative
        
        forecast_categories = [f"Predicted_{i+1}" for i in range(forecast_points)]
        return pd.DataFrame({x_col: forecast_categories, 'Forecast': forecast_values}), 0.0
    
    return None, 0.0


def generate_insights_universal(df, x_col, y_col, chart_type):
    """
    Generates insights based on the chart type and data.
    """
    insights = []
    
    if chart_type == "time_series":
        # Time series insights
        df_clean = df.dropna(subset=[x_col, y_col])
        if len(df_clean) >= 2:
            first_val = df_clean[y_col].iloc[0]
            last_val = df_clean[y_col].iloc[-1]
            trend_pct = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0
            trend_direction = "increased" if last_val > first_val else "decreased"
            
            insights.append(f"📈 **Trend**: {y_col} has {trend_direction} by {abs(trend_pct):.1f}% over the time period.")
            insights.append(f"🔼 **Peak**: Highest value was {df_clean[y_col].max():,.2f}")
            insights.append(f"🔽 **Lowest**: Lowest value was {df_clean[y_col].min():,.2f}")
    
    elif chart_type == "scatter":
        # Correlation insights
        corr = df[x_col].corr(df[y_col])
        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if corr > 0 else "negative"
        insights.append(f"🔗 **Correlation**: There is a {strength} {direction} correlation ({corr:.3f}) between {x_col} and {y_col}.")
    
    elif chart_type == "bar":
        # Categorical vs numerical insights
        grouped = df.groupby(x_col)[y_col].agg(['mean', 'std', 'count'])
        top_category = grouped['mean'].idxmax()
        top_value = grouped['mean'].max()
        
        insights.append(f"🏆 **Top Category**: '{top_category}' has the highest average {y_col} at {top_value:.2f}")
        insights.append(f"📊 **Categories**: Analysis covers {len(grouped)} different categories")
    
    elif chart_type == "heatmap":
        # Categorical vs categorical insights
        total_combinations = df.groupby([x_col, y_col]).size()
        most_common = total_combinations.idxmax()
        most_common_count = total_combinations.max()
        
        insights.append(f"🔥 **Most Common**: '{most_common[0]}' + '{most_common[1]}' combination appears {most_common_count} times")
        insights.append(f"📈 **Unique Combinations**: {len(total_combinations)} different combinations found")
    
    return "\n\n".join(insights) if insights else "No specific insights available for this data combination."

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
LOGO_PATH = "logo.png"
logo_base64 = get_image_as_base64(LOGO_PATH)

if logo_base64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" width="200">
            <h1 style="margin-left: 20px;">📊 Time series analysis and sales forecasting </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("📊 Universal Data Analysis and Visualization")

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
        
        st.subheader("Column Selection")
        x_axis = st.selectbox("Select X-axis column", columns)
        y_axis = st.selectbox("Select Y-axis column", columns)

        # Determine chart type and show info
        if x_axis != y_axis:
            chart_type, chart_description = determine_chart_type(df, x_axis, y_axis)
            st.info(f"**Visualization Type**: {chart_description}")
            
            # Show data type information with better detection
            x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_axis])
            if df[x_axis].dtype == 'object' and not x_is_datetime:
                try:
                    sample_size = min(100, len(df))
                    sample_data = df[x_axis].dropna().head(sample_size)
                    parsed_dates = pd.to_datetime(sample_data, errors='coerce')
                    if parsed_dates.notna().sum() / len(sample_data) > 0.8:
                        x_is_datetime = True
                except:
                    x_is_datetime = False
            
            if x_is_datetime:
                x_type = "Date/Time"
            elif pd.api.types.is_numeric_dtype(df[x_axis]):
                x_type = "Numerical"
            else:
                x_type = "Categorical/Text"
                
            y_type = "Numerical" if pd.api.types.is_numeric_dtype(df[y_axis]) else "Categorical/Text"
            
            col1, col2 = st.columns(2)
            col1.metric("X-axis Type", x_type)
            col2.metric("Y-axis Type", y_type)

            # Special controls for forecasting
            periods = 0
            forecast_possible = can_forecast(df, x_axis, y_axis, chart_type)
            
            if forecast_possible:
                st.subheader("🔮 Forecasting Controls")
                
                if chart_type == "time_series" and pd.api.types.is_numeric_dtype(df[y_axis]):
                    # Time series forecasting with date selection
                    try:
                        temp_df = df[[x_axis]].copy()
                        temp_df[x_axis] = pd.to_datetime(temp_df[x_axis], errors='coerce').dropna()
                        
                        if not temp_df.empty:
                            min_date = temp_df[x_axis].min()
                            max_date = temp_df[x_axis].max()
                            
                            forecast_end_date = st.date_input(
                                "Select forecast end date",
                                value=max_date + pd.to_timedelta(30, unit='D'),
                                min_value=max_date,
                                help="Choose a date in the future to forecast up to."
                            )
                            periods = len(pd.date_range(start=max_date, end=forecast_end_date, freq='D')) - 1
                            st.info(f"🔮 Forecasting {periods} periods ahead")
                    except:
                        st.warning("Could not parse dates for forecasting")
                
                else:
                    # Non-time-series forecasting
                    forecast_points = st.slider(
                        "Number of forecast points",
                        min_value=5,
                        max_value=50,
                        value=10,
                        help="Select how many future points to predict"
                    )
                    periods = forecast_points
                    
                    if chart_type == "scatter":
                        st.info("📈 Will use linear regression to extrapolate trend")
                    elif chart_type == "bar":
                        st.info("📊 Will extrapolate category trends")
                
            if st.button("Generate Dashboard"):
                st.session_state.show_dashboard = True
                st.session_state.chat_history = []
                st.session_state.chart_type = chart_type
                st.session_state.periods = periods
                st.session_state.forecast_possible = forecast_possible
        else:
            st.warning("Please select different columns for X and Y axes.")

        st.divider()

        st.header("🤖 Ask a Question")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if prompt := st.chat_input("Ask a question about your data..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            prompt_lower = prompt.lower()
            with st.chat_message("assistant"):
                responses = []
                mentioned_cols = [col for col in numeric_columns if col.lower() in prompt_lower]

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
                            stats_table = stats.to_frame().to_markdown()
                            responses.append(f"Here is a statistical summary for **{col}**:\n\n{stats_table}")
                else:
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
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.divider()

        st.header("📜 Chat History")
        for chat in st.session_state.chat_history[:-2]:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

# --- MAIN CONTENT AREA ---
if not uploaded_file:
    st.info("Please upload a CSV file using the sidebar to get started.")
else:
    if st.session_state.show_dashboard:
        try:
            chart_type = st.session_state.get('chart_type', 'scatter')
            periods = st.session_state.get('periods', 0)
            forecast_possible = st.session_state.get('forecast_possible', False)
            
            # Prepare data for visualization
            df_viz = prepare_data_for_visualization(df, x_axis, y_axis, chart_type)
            
            # Determine tab structure based on forecasting capability
            if forecast_possible and periods > 0:
                if chart_type == "time_series":
                    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📄 Data Preview", "📈 Model Performance"])
                else:
                    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📄 Data Preview", "🔮 Forecasting"])
            else:
                tab1, tab2 = st.tabs(["📊 Dashboard", "📄 Data Preview"])

            with tab1:
                st.header("Data Visualization")
                
                # Create and display the main visualization
                fig = create_visualization(df_viz, x_axis, y_axis, chart_type, df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display insights
                with st.expander("🔍 Data Analysis Insights"):
                    insights = generate_insights_universal(df, x_axis, y_axis, chart_type)
                    st.markdown(insights)
                
                # Additional statistics
                st.header("Key Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    if pd.api.types.is_numeric_dtype(df[y_axis]):
                        st.metric(f"Avg {y_axis}", f"{df[y_axis].mean():.2f}")
                    else:
                        st.metric(f"Unique {y_axis}", f"{df[y_axis].nunique()}")
                with col3:
                    if pd.api.types.is_numeric_dtype(df[x_axis]):
                        st.metric(f"Avg {x_axis}", f"{df[x_axis].mean():.2f}")
                    else:
                        st.metric(f"Unique {x_axis}", f"{df[x_axis].nunique()}")
                with col4:
                    st.metric("Missing Values", f"{df[[x_axis, y_axis]].isnull().sum().sum()}")

            with tab2:
                st.header("Data Preview")
                st.subheader("Original Data")
                st.dataframe(df, use_container_width=True)
                
                st.subheader("Processed Data for Visualization")
                if isinstance(df_viz, pd.DataFrame):
                    st.dataframe(df_viz, use_container_width=True)
                else:
                    st.dataframe(df_viz.to_frame() if hasattr(df_viz, 'to_frame') else df_viz, use_container_width=True)

            # Forecasting tab (if applicable)
            if forecast_possible and periods > 0 and 'tab3' in locals():
                with tab3:
                    if chart_type == "time_series":
                        # Existing time series forecasting logic
                        st.header("Time Series Forecasting")
                        try:
                            df_agg, hist_df, forecast_df, mae, rmse, y_test_inv, y_pred_inv, test_dates, has_test_data = generate_forecast_data(df, x_axis, y_axis, periods)
                            
                            if has_test_data:
                                avg_hist_value = hist_df['Value'].mean()
                                percentage_error = (mae / avg_hist_value) * 100 if avg_hist_value > 0 else 0
                                model_accuracy = 100 - percentage_error

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
                                
                                # Combined historical and forecast chart
                                st.header("Historical vs. Forecasted Data")
                                fig_combined = go.Figure()
                                fig_combined.add_trace(go.Scatter(x=hist_df[x_axis], y=hist_df['Value'], mode='lines', name='Historical', line=dict(color='teal')))
                                fig_combined.add_trace(go.Scatter(x=forecast_df[x_axis], y=forecast_df['Value'], mode='lines', name='Forecast', line=dict(color='orange')))
                                fig_combined.update_layout(title="Historical vs. Forecasted Data Over Time", xaxis_title=x_axis, yaxis_title=y_axis)
                                st.plotly_chart(fig_combined, use_container_width=True)
                            else:
                                st.warning("Not enough data to create a test set for evaluation.")
                        except Exception as e:
                            st.error(f"Time series forecasting failed: {str(e)}")
                    
                    else:
                        # Non-time-series forecasting
                        st.header("Trend-Based Forecasting")
                        try:
                            forecast_df, accuracy = generate_non_timeseries_forecast(df, x_axis, y_axis, chart_type, periods)
                            
                            if forecast_df is not None:
                                col1, col2 = st.columns(2)
                                if chart_type == "scatter":
                                    col1.metric("R² Score", f"{accuracy:.3f}")
                                    col2.metric("Forecast Points", f"{len(forecast_df)}")
                                else:
                                    col1.metric("Forecast Method", "Trend Extrapolation")
                                    col2.metric("Forecast Points", f"{len(forecast_df)}")
                                
                                st.header("Forecast Visualization")
                                if chart_type == "scatter":
                                    # Scatter plot with forecast line
                                    fig_forecast = go.Figure()
                                    fig_forecast.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='markers', name='Historical Data', marker=dict(color='teal')))
                                    fig_forecast.add_trace(go.Scatter(x=forecast_df[x_axis], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='orange', dash='dash')))
                                    fig_forecast.update_layout(title=f"Historical Data with Trend Forecast", xaxis_title=x_axis, yaxis_title=y_axis)
                                    st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                elif chart_type == "bar":
                                    # Bar chart with forecast bars
                                    historical_data = df.groupby(x_axis)[y_axis].mean().reset_index()
                                    
                                    fig_forecast = go.Figure()
                                    fig_forecast.add_trace(go.Bar(x=historical_data[x_axis], y=historical_data[y_axis], name='Historical Average', marker_color='teal'))
                                    fig_forecast.add_trace(go.Bar(x=forecast_df[x_axis], y=forecast_df['Forecast'], name='Forecast', marker_color='orange'))
                                    fig_forecast.update_layout(title=f"Historical vs Forecasted {y_axis}", xaxis_title=x_axis, yaxis_title=y_axis)
                                    st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                st.header("Forecast Data")
                                st.dataframe(forecast_df, use_container_width=True)
                            
                            else:
                                st.error("Could not generate forecast for this data combination.")
                        
                        except Exception as e:
                            st.error(f"Forecasting failed: {str(e)}")
                            st.exception(e)

        except Exception as e:
            st.error(f"An error occurred during dashboard generation: {str(e)}")
            st.exception(e)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;padding-top: 10px;border-top: 1px solid #d6d6d8;'>© Copyright 2007-2025. inoday - All Rights Reserved | Privacy Policy An ISO 9001:2015 Certified Company</p>", unsafe_allow_html=True)