#!/usr/bin/env python
# coding: utf-8

# 
# # Citi Bikes Strategy Dashboard — Streamlit App
# 

# In[1]:


# Importing libraries

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numerize.numerize import numerize
from PIL import Image
from streamlit_keplergl import keplergl_static  

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title='Citi Bikes Strategy Dashboard', layout='wide')
st.title('Citi Bikes Strategy Dashboard')

# ------------------------------
# Sidebar navigation
# ------------------------------
st.sidebar.title('Aspect Selector')
page = st.sidebar.selectbox(
    'Select an aspect of the analysis',
    ['Intro page', 'Weather component and bike usage', 'Most popular stations',
     'Interactive map with aggregated bike trips', 'Recommendations']
)

# ------------------------------
# Data sources
# ------------------------------
DATA_PATH = 'merged_trips_weather_2022_sample.csv'
TOP20_PATH = 'top20_startstations_2022.csv'
MAP_PATH  = 'kepler_popular_trips_map.html'

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date')
if os.path.exists(TOP20_PATH):
    top20 = pd.read_csv(TOP20_PATH)
else:
    top20 = pd.DataFrame(columns=['start_station_name','value'])

# Normalize top20 columns if needed
if 'start_station_name' not in top20.columns:
    for c in top20.columns:
        if 'start' in c.lower() and 'station' in c.lower():
            top20 = top20.rename(columns={c: 'start_station_name'})
if 'value' not in top20.columns:
    for c in top20.columns:
        if c.lower() in ['count','trips','trip_count','value']:
            top20 = top20.rename(columns={c: 'value'})

# ------------------------------
# Derived fields & quick stats
# ------------------------------
def season_of(d):
    m = d.month
    if m in (12,1,2): return 'Winter'
    if m in (3,4,5): return 'Spring'
    if m in (6,7,8): return 'Summer'
    return 'Fall'

df['season'] = df['date'].apply(season_of)

# Daily aggregates
daily = df.groupby('date').size().reset_index(name='trips')
sample_total = int(len(df))
date_min, date_max = df['date'].min().date(), df['date'].max().date()
avg_daily = float(daily['trips'].mean())
median_daily = float(daily['trips'].median())
peak_row = daily.loc[daily['trips'].idxmax()]
peak_date = peak_row['date'].date()
peak_dow = peak_row['date'].day_name()
peak_trips = int(peak_row['trips'])

# Month/season
monthly = daily.assign(month=daily['date'].dt.to_period('M')).groupby('month')['trips'].sum().reset_index()
monthly['month_name'] = monthly['month'].dt.strftime('%B')
monthly_sorted = monthly.sort_values('trips', ascending=False)
seasonal = df.groupby('season').size().reset_index(name='trips').sort_values('trips', ascending=False)

# Weekdays vs weekends
daily['dow'] = daily['date'].dt.day_name()
daily['is_weekend'] = daily['dow'].isin(['Saturday','Sunday'])
weekend_total = int(daily.loc[daily['is_weekend'],'trips'].sum())
weekday_total = int(daily.loc[~daily['is_weekend'],'trips'].sum())
weekend_share = weekend_total / (weekend_total + weekday_total) if (weekend_total + weekday_total) > 0 else 0.0

# Vehicle types (if present)
if 'rideable_type' in df.columns:
    type_counts = df['rideable_type'].value_counts().reset_index()
    type_counts.columns = ['rideable_type','trips']
else:
    type_counts = pd.DataFrame(columns=['rideable_type','trips'])

# Top20 (year) prep
if not top20.empty:
    top20 = top20[['start_station_name','value']].dropna().sort_values('value', ascending=False).reset_index(drop=True)
    top20_total = int(top20['value'].sum()) if 'value' in top20.columns else 0
    if top20_total > 0:
        top20['share_of_top20'] = top20['value'] / top20_total
    else:
        top20['share_of_top20'] = 0.0
else:
    top20_total = 0

# Sample station counts
if 'start_station_name' in df.columns:
    sample_station_counts = df['start_station_name'].value_counts().reset_index()
    sample_station_counts.columns = ['start_station_name','trips']
    sample_station_counts['share_of_sample'] = sample_station_counts['trips'] / sample_total if sample_total else 0.0
else:
    sample_station_counts = pd.DataFrame(columns=['start_station_name','trips','share_of_sample'])

# Handy picks for text
top_months = monthly_sorted.head(3).copy()
if top20_total > 0 and len(top20) >= 5:
    t1_name, t1_val = top20.iloc[0]['start_station_name'], int(top20.iloc[0]['value'])
    t2_name, t2_val = top20.iloc[1]['start_station_name'], int(top20.iloc[1]['value'])
    t3_name, t3_val = top20.iloc[2]['start_station_name'], int(top20.iloc[2]['value'])
    top1_share = top20.iloc[0]['share_of_top20']
    top3_share = (top20.iloc[0]['value'] + top20.iloc[1]['value'] + top20.iloc[2]['value']) / top20_total
    top5_share = top20.head(5)['value'].sum() / top20_total
else:
    t1_name = t2_name = t3_name = ''
    t1_val = t2_val = t3_val = 0
    top1_share = top3_share = top5_share = 0.0

# ------------------------------
# Utility for pretty numbers
# ------------------------------
def pct(x):
    return f"{x*100:.2f}%"

def fmt(x):
    try:
        return numerize(x)
    except Exception:
        return f"{x:,}"

# ------------------------------
# PAGES
# ------------------------------

if page == 'Intro page':
    st.subheader('What are we seeing here (big picture)')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Trips in sample', f"{sample_total:,}")
    with col2:
        st.metric('Date range', f"{date_min} → {date_max}")
    with col3:
        st.metric('Avg daily trips', f"{avg_daily:.1f}")
    with col4:
        st.metric('Median daily trips', f"{int(median_daily):,}")

    st.markdown(f"- The single busiest day is **{peak_date} ({peak_dow})** with **{peak_trips}** trips.")
    tm_text = ", ".join([f"**{m}** ({t:,})" for m,t in zip(top_months['month_name'], top_months['trips'])])
    seas_text = ", ".join([f"**{s}** ({int(t):,})" for s,t in zip(seasonal['season'], seasonal['trips'])])

    st.subheader('When it’s busiest')
    st.markdown(f"- The line climbs in late spring, stays high through summer, and remains strong in early fall.\n- Top months in this sample: {tm_text}.\n- By season: {seas_text}.")

elif page == 'Weather component and bike usage':
    st.subheader('What the time-series tells me')
    fig = go.Figure(go.Scatter(x=daily['date'], y=daily['trips'], name='Daily Trips'))
    fig.update_layout(title='Daily trips in the sample', xaxis_title='Date', yaxis_title='Trips', plot_bgcolor='white', height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"- Weekdays carry **{pct(1 - weekend_share)}** of trips (weekends **{pct(weekend_share)}**).\n- The single biggest day is **{peak_date} ({peak_dow})**.\n- Peaks are in summer into early fall; winter is quieter.")

elif page == 'Most popular stations':
    st.subheader('What stands out in the station rankings')

    if not top20.empty:
        fig = go.Figure(go.Bar(
            x=top20['start_station_name'],
            y=top20['value'],
            marker={'color': top20['value'], 'colorscale': 'Blues'}
        ))
        fig.update_layout(title='Top 20 start stations (2022)', xaxis_title='Start station', yaxis_title='Trips', height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"- **{t1_name}** is #1 with **{t1_val:,}** rides (**{pct(top1_share)}** of the Top‑20 total).\n"
            f"- Top three — **{t1_name}**, **{t2_name}** ({t2_val:,}), **{t3_name}** ({t3_val:,}) — account for **{pct(top3_share)}**.\n"
            f"- Top five together make up **{pct(top5_share)}** of Top‑20 rides."
        )
    else:
        st.info("Top‑20 file missing or lacks 'start_station_name'/'value'.")

    if not sample_station_counts.empty:
        st.subheader('Sample perspective (starts in the sample file)')
        show = sample_station_counts.head(3).copy()
        show['share_%'] = (show['trips'] / sample_total * 100).round(2)
        st.dataframe(show, use_container_width=True)

elif page == 'Interactive map with aggregated bike trips':
    st.subheader('What the map shows us')
    st.write('I see dense clusters around PATH and the Hudson waterfront. Those are the places people start most often and move between.')

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, 'r') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=900)
        st.caption('Kepler.gl export (attached file). Use the embedded filters to drill down.')
    else:
        st.info(f"Map file not found: {MAP_PATH}")

    st.markdown('- Compare weekday commute peaks vs weekend leisure in the map filters.\n- Tie this back to Top‑20 to plan docks, staging, and valet.')

else:
    st.subheader('Recommendation')

    rec_lines = []
    if t1_name:
        rec_lines.append(f"1. Focus docks/valet/rebalancing on **{t1_name}**, **{t2_name}**, **{t3_name}** — together about **{pct(top3_share)}** of Top‑20 rides (Top‑5 ≈ **{pct(top5_share)}**).")
    rec_lines.append("2. Staff up in Summer and Fall — the busiest seasons; in this sample June/Aug/Sept stand out.")
    rec_lines.append(f"3. Plan a Saturday morning push — even though weekdays carry **{pct(1 - weekend_share)}**, the single busiest day is **{peak_date} ({peak_dow})**.")
    if not type_counts.empty:
        type_share_text = ", ".join([f"{rt.replace('_',' ')} ~{(trips/sample_total*100):.1f}%" for rt,trips in zip(type_counts['rideable_type'], type_counts['trips'])])
        rec_lines.append(f"4. Match the bike mix to use — in this area I’m seeing {type_share_text}. Keep charging/maintenance aligned, especially near PATH and the waterfront.")
    rec_lines.append("5. Design short rebalancing loops around PATH/waterfront hubs — e.g., Grove St PATH → Hoboken Terminal → South Waterfront Walkway → City Hall.")

    st.markdown("\n".join(rec_lines))


# In[ ]:




