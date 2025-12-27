# ----- IMPORTS -----
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns


# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Citi Bike 2022 – NYC Dashboard",
    layout="wide"
)

# ----- DATA LOADING -----
@st.cache_data
def load_data():
    df = pd.read_csv("data/citibike_2022_small.csv")
    df["date"] = pd.to_datetime(df["date"])
    if "trip_count" not in df.columns:
        df["trip_count"] = 1
    return df

df = load_data()

# ----- SIDEBAR: PAGE SELECTION -----
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "Intro",
        "How Temperature Influences Daily Trips",
        "Top Stations",
        "Citibike Ride Map",
        "Ride Demand Heatmap: Hour vs. Weekday",
        "Recommendations"
    ]
)

# ----- FUNCTIONS FOR CHARTS -----

def plot_top_stations(df, column="start_station_name", top_n=10):
    data = df[column].value_counts().head(top_n).reset_index()
    data.columns = ["Station", "Trips"]
    fig = px.bar(
        data,
        x="Station",
        y="Trips",
        title=f"Top {top_n} Start Stations in NYC",
        labels={"Station": "Station Name", "Trips": "Number of Trips"}
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_trips_dual(daily_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for daily trips and rolling average
    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["trips"],
            name="Daily Trips",
            mode="lines"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["trips_rolling"],
            name="7-day Moving Average",
            mode="lines"
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Daily Citi Bike Trips and 7-day Moving Average (2022)",
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Daily Trips", secondary_y=False)
    fig.update_yaxes(title_text="7-day Moving Average", secondary_y=True)

    return fig


# ----- PREPARE DAILY DATA FOR DUAL-AXIS LINE CHART -----
daily = df.groupby("date").size().reset_index(name="trips")
daily["trips_rolling"] = daily["trips"].rolling(window=7).mean()


# ----- PAGE CONTENT -----

if page == "Intro":
    st.title("Citi Bike 2022 – NYC Dashboard")

    col_text, col_img = st.columns([3, 2], gap="large")

    with col_text:
        st.markdown("""
        Welcome to the **Citi Bike 2022 Dashboard**.  

        This dashboard explores Citi Bike usage patterns across New York City in 2022.
        You can use it to:

        - See **how daily rides change over time**
        - Identify **the most popular start stations**
        - View a **spatial overview of trips** on the Kepler map
        - Analyze **ride demand patterns** by hour and weekday
        - **Final recommendations** based on the analysis.
        """)

    with col_img:
        st.image(
            "Imgs/citi_bike_intro.jpg",
            caption="Source: Unsplash – Photo by Lavi Cella",
            width=450 
        )

    
elif page == "How Temperature Influences Daily Trips":
    st.subheader("Daily Trips and 7-day Moving Average")

    fig_line = plot_trips_dual(daily)
    st.plotly_chart(fig_line, width="stretch")

    st.markdown("""
        **Interpretation:**  
        The dark blue line shows the **exact number of Citi Bike trips per day**, while the light blue line shows the 
        **7-day moving average**, which smooths short-term fluctuations.
        
        We can see that:
        - **Seasonal effect:** Warmer months show higher trip volumes, indicating a strong relationship between temperature and ridership.

        - **Operational pressure:** Sustained high demand in summer months may challenge bike availability and require more intensive redistribution.

        - **Trend clarity with 7-day average:** The 7-day moving average highlights the overall seasonal trend and helps filter out daily noise for clearer interpretation.
    """)


elif page == "Top Stations":
    st.subheader("Most Popular Start Stations")

    fig_bar = plot_top_stations(df)
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("""
        **Interpretation:**  
        This bar chart shows the **top start stations** by number of trips in 2022.  
        A small group of stations clearly dominates usage, indicating **high-demand locations** 
        where bike availability is especially important.
        
        These stations are likely:
        - Close to major **transport hubs** (e.g., train or subway stations),
        - Located in **dense residential or business areas** such as Midtown Manhattan,
        - Positioned near popular **tourist spots**.

        Ensuring enough bikes and docks at these stations can significantly improve 
        the overall user experience and reduce shortages.
    """)

elif page == "Citibike Ride Map":
    st.subheader("Citi Bike Trips Map – Kepler.gl")

    custom_css = """
    <style>
        iframe {
            transform: translateZ(0);
        }
    </style>
    """

    try:
        with open("Docs/citibike_small_map.html", "r", encoding="utf-8") as f:
            kepler_html = f.read()

        kepler_html = kepler_html.replace(
            "longitude:-122",
            "longitude:-74.0060"
        ).replace(
            "latitude:37.",
            "latitude:40.7128"
        )

        st.components.v1.html(custom_css + kepler_html, height=700, scrolling=True)

        st.markdown("""
        ### Interpretation  
        New York shows the highest concentration of Citi Bike activity in central Manhattan,
        especially around Midtown and Downtown. High-density areas indicate strong commuter
        and tourist traffic, while outer zones show lower demand.
        """)

    except FileNotFoundError:
        st.error("Kepler map file not found. Check that 'Docs/citibike_small_map.html' exists.")

elif page == "Ride Demand Heatmap: Hour vs. Weekday":
    st.subheader("Peak Demand: Heatmap by Hour and Weekday")


    df["started_at"] = pd.to_datetime(df["started_at"])
    df["hour"] = df["started_at"].dt.hour
    df["weekday"] = df["started_at"].dt.day_name()

    pivot = df.pivot_table(
        index="weekday",
        columns="hour",
        values="trip_count",
        aggfunc="count"
    ).fillna(0)

    ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(ordered_days)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, ax=ax)
    ax.set_title("Citi Bike demand by hour and weekday")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")

    st.pyplot(fig)

    st.markdown("""
    ### Interpretation
    This heatmap shows at which hours and on which days Citi Bike demand peaks.

    - **Weekday rush hours (7–9 AM, 4–7 PM)** are the highest points of demand.
    - **Weekend usage is more spread out**, centered around late morning and afternoon.
    - This pattern is critical for supply planning:
        - Boost bike supply in central Manhattan during rush hours.
        - Bring more bikes to recreational areas on weekends.
        - Redistribution teams should focus on commuter patterns.

    This chart supports a clear plan to optimize Citi Bike availability across NYC.
    """)

elif page == "Recommendations":
    st.header("Recommendations")

    st.markdown("""
    Based on the overall analysis of Citi Bike usage patterns in New York City, 
    several clear opportunities emerge for optimizing bike supply and improving system efficiency.

    ### 1. Increase bike availability during weekday rush hours  
    The strongest peaks occur on **weekdays between 7–9 AM and 4–7 PM**, driven by commuters.  
    - Focus bike redistribution teams in **Midtown, Downtown, and Lower Manhattan**.  
    - Prepare these stations with higher bike inventory before rush hour starts.

    ### 2. Strengthen supply in high-demand central stations  
    Both the map and bar charts show that usage is highest in the **central business district**.  
    - These zones need **double the supply** compared to outer areas.  
    - Consider adding temporary or pop-up docking capacity during busy seasons.

    ### 3. Adjust weekend supply for recreational demand  
    Weekends show **late-morning and early-afternoon peaks**, especially near parks and waterfronts.  
    - Reallocate bikes toward areas like **Central Park, Hudson River Greenway, and Brooklyn waterfront**.  
    - Less focus needed on commuting hot-spots during weekends.

    ### 4. Improve overnight bike redistribution  
    Data indicates significant imbalance by the end of the day.  
    - Implement automated overnight balancing routes.  
    - Use predictive modeling to prepare inventory for the next day’s rush hours.

    ### 5. Use historical patterns to forecast future demand  
    The consistent seasonality and hourly patterns provide a strong basis for predictive planning.  
    - Leverage machine learning or time-series forecasting for bike distribution schedules.  
    - Combine this with weather forecasts, which significantly influence demand.

    ### Conclusion  
    With strategic redistribution, time-dependent supply planning, and predictive insights, 
    Citi Bike can significantly reduce shortages and improve availability across New York City.
    """)


# ----- END OF FILE -----