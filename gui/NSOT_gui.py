import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from basketball_reference_scraper.teams import *
from basketball_reference_scraper.players import *
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns


# Loading in the main csv
nba = pd.read_csv('datasets/player_stats.csv')

# Loading in the predicted csv
pred = pd.read_csv("datasets/predictions.csv")
pred = pred.set_index("Player")
pred = pred.dropna()
pred = pred.drop("Unnamed: 0", axis=1)

# Sidebar Config
st.set_page_config(page_title='NSOT Dashboard',
                   layout="wide")
st.sidebar.title('NBA Salary Optimization Tool')
st.sidebar.header('Player & Season Selection')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2000, 2021))))
nba = nba.set_index("Player")
nba = nba.dropna()
nba = nba.drop("Unnamed: 0", axis=1)
nba.head()

# Grab player name
filteredNBA = nba[nba.index.str.contains(str(selected_year))]

selected_player_display = st.sidebar.selectbox('Players', filteredNBA.index.map(lambda x: str(x)[:-5]))
selected_player = selected_player_display + " " + str(selected_year)


# Adding a column sidebars so you can look at graphs of your choice 
st.sidebar.title("Stat Selection")
st.title(selected_player_display + f" - Season {selected_year}")
column_selection = filteredNBA.columns
column_selection = column_selection[4:]
selected_column = st.sidebar.selectbox(label=str(selected_year)+" stats", options=column_selection)
filterNameNBA = nba[nba.index.str.contains(str(selected_player_display))]
pCapfiltered = filterNameNBA['pCap']


def get_player_headshot(name):
    suffix = get_player_suffix(name)
    jpg = suffix.split('/')[-1].replace('html', 'jpg')
    url = 'https://d2cwpp38twqe55.cloudfront.net/req/202006192/images/players/' + jpg
    return url

def pCap_chart(selected_player_display):
    '''
    This function returns pCap by season
    '''
    fig1 = Figure()
    ax = fig1.subplots()
    pCapfiltered = filterNameNBA[["Season", "pCap"]]   
    pCapfiltered = pCapfiltered.set_index("Season")
    pCapfiltered2 = filterNameNBA[["Season", "Predicted pCap"]]  
    pCapfiltered2 = pCapfiltered2.set_index("Season")
    sns.lineplot(data=pCapfiltered, ax=ax)
    sns.lineplot(data=pCapfiltered2, ax=ax, palette=["orange"])
    ax.legend() 
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('% of Cap Space Taken', fontsize=12)
    ax.set_axisbelow(True)
    row1_1.pyplot(fig1)


def graph_2000to2020(selected_player_display, selected_column, tuple, filter1, selected_filter):
    '''
    This function plots any stat vs. salary cap percentage from the 2000 to 2020 season
    '''

    fgp2 = filteredNBA[filteredNBA["Season"] == selected_year]
    fgp = filterNameNBA[filterNameNBA["Season"] == selected_year]
    position = fgp["Pos"][0]

    if filter1:
        fgp2 = fgp2[(fgp2[selected_filter] >= tuple[0]) & (fgp2[selected_filter] <= tuple[1])]

    fgp2 = fgp2[fgp2["Pos"] == position]
    fgp = fgp[[selected_column,'pCap']]
    fgp2 = fgp2[[selected_column, 'pCap']]

    fig4 = plt.figure()

    ax = fig4.subplots()
    plt.scatter(x=selected_column, y="pCap", data=fgp2, label = "NBA Players With Same Position")
    plt.scatter(x=selected_column, y="pCap", data=fgp, label = selected_player, marker="*", s=500.0)
    ax.set_xlabel(selected_column, fontsize=12)
    ax.set_ylabel('% of Cap Space Taken', fontsize=12)
    ax.legend() 
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)

    row1_2.pyplot(fig4)

def graph_2021(selected_player_display, selected_column2, tuple, filter2, selected_filter):
    '''
    This function plots any stat vs. salary cap percentage in 2021
    '''
    copy = pred

    fgp = copy[[selected_column2,'pCap', "sort.km.res.cluster."]]
    fgp2 = copy.loc[selected_for_pred][[selected_column2, 'pCap']]

    if filter2:
        fgp = fgp[(copy[selected_filter] >= tuple[0]) & (copy[selected_filter] <= tuple[1])]


    cluster_number = pred.loc[selected_for_pred]["sort.km.res.cluster."] 
    pred_cluster = fgp[fgp["sort.km.res.cluster."] == cluster_number]
    pred_cluster = pred_cluster[[selected_column2,'pCap']]
    c = pred_cluster[[selected_column2, 'pCap']]

    fig4 = plt.figure()
    ax = fig4.subplots()
    plt.scatter(x=selected_column2, y="pCap", data=fgp, label = "All NBA Players")
    plt.scatter(x=selected_column2, y="pCap", data=c, label = "Clusters of similar players")
    plt.scatter(x=selected_column2, y="pCap", data=fgp2, marker="*", s=500.0, label = selected_player_display)
    ax.set_xlabel(selected_column2, fontsize=12)
    ax.set_ylabel('% of Cap Space Taken', fontsize=12)
    ax.legend() 
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    row1_3.pyplot(fig4)



# Create columns and add player headshot 
col1, mid, col2, mid2, col3 = st.beta_columns([4, 0.5, 5, 1, 20])
col1.image(get_player_headshot(selected_player_display), use_column_width=True)



# Adding Table of Statistics
playerStat = nba[nba.index.str.contains(str(selected_player))]
col2.write(f"Age: {round(playerStat.iloc[0]['Age'])}")
col2.write(f"Position: {playerStat.iloc[0]['Pos']}")
col2.write(f"Team: {playerStat.iloc[0]['Tm']}")
col2.write(f"Actual Salary: ${'{:,.2f}'.format((round((playerStat.iloc[0]['Salary']))))}")
col2.write(f"Predicted Salary: ${'{:,.2f}'.format(((playerStat.iloc[0]['Predicted Salary'])))}")
options = col3.multiselect(
    'Select Player Stats To Display In Table',
    list(column_selection),
    ["Games Played (G)", "2 Pointers Made (2PM)", "3 Points Made (3PM)"])
col3.write("Player Statistics")
col3.dataframe(filterNameNBA[options])

# Adding graphs & 2021 Stats Selectbox
row1_1, row1_spacer2, row1_2, row1_spacer3, row1_3 = st.beta_columns(
    (5, 2, 5, 2, 5)
    )

column_selection2 = pred.columns
column_selection2 = column_selection2[4:]
selected_column2 = st.sidebar.selectbox(label="2021 Stats", options=column_selection2)

# Stat Filter & Select Stat Selectbox
st.sidebar.title("Stat Filter")
selected_filter = st.sidebar.selectbox(label="Select Stat", options=column_selection2)

# Find min and max of certain stats
max_selected_column = max(float(nba[selected_filter].max()), float(pred[selected_filter].max()))
min_selected_column = min(float(nba[selected_filter].min()), float(pred[selected_filter].min()))
values = st.sidebar.slider(selected_filter, min_value=min_selected_column, max_value=max_selected_column, value=(min_selected_column, max_selected_column))
filter1 = st.sidebar.checkbox(f"Apply filter to {selected_year} Stats")


with row1_1:
    st.subheader('Percent of Team Salary Cap by Season')
    pCap_chart(selected_player_display)
with row1_2:
    st.subheader(str(selected_year) + ' Salary vs ' + selected_column)
    graph_2000to2020(selected_player_display, selected_column, values, filter1, selected_filter)
with row1_3: 
    selected_for_pred = selected_player[0:len(selected_player) - 5] + " " + "2021"
    pred_player = pd.Series(pred.index).unique()

    if selected_for_pred in pred_player:
        st.subheader("2021 Salary vs " + selected_column2)
        filter2 = st.sidebar.checkbox("Apply filter to 2021 Stats")
        graph_2021(selected_player_display, selected_column2, values, filter2, selected_filter)


    else: 
        st.subheader("Player is retired; no 2021 statistics.")
    


# Adding link to definition of statistics
row2_1, row2_spacer3, row2_2, row2_spacer4, row2_3 = st.beta_columns(
    (5, 2, 5, 2, 5)
    )
with col2:
    selected_for_pred = selected_player[0:len(selected_player) - 5] + " " + "2021"
    pred_player = pd.Series(pred.index).unique()
    if selected_for_pred in pred_player:
        st.write("2022 Predicted Salary: $" + '{:,.2f}'.format(round(pred.loc[selected_for_pred]["2022 Predicted Salary"])))
with row2_1:
    st.subheader("Basketball Statistics Reference:")
with row2_2: 
    st.subheader("https://www.nba.com/stats/help/glossary/")
