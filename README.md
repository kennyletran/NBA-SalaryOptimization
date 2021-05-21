# NBA-SalaryOptimization

The NBA has a salary cap, which is a limit to the total amount of money they're allowed to spend on their players. As a result, it's imperative that every contract given matches the player's value (or potential value). There are countless instances where a "bad" contract (a contract given to a player who's production is much lower than their contract value) has negatively affected the team.

A motivating example is Timofey Mozgov who signed a 4 year, $64 million contract with the Los Angeles Lakers in 2016. Desite this large contract, Mozgov only averaged 7.4 points and 4.9 rebounds in 20.4 MPG, making his contract one of the worst in recent NBA history. In order to get rid of his salary, the Los Angeles Lakers had to trade away future all-star D'Angelo Russell.

This project seeks to use machine learning to find how valuable a player currently is. We will use basic and advanced statistics from the years 2000-2020 to predict player salaries in the form of the percentage of cap space they take up (player salary / NBA Salary Cap). In addition, we decided to create a model for each of the five nba positions to prevent playstyle biases.

Four models were considered and their average R-squared and RMSE are as listed: Lasso Regression (R2 = 0.5968, RMSE = 0.516), Recursive Feature Elimination + Ridge Regression (R2 = 0.5838, RMSE = 0.0533), Elastic-Net Regression (R2 = 0.5962, RMSE = 0.0517) , and Decision Tree Regression (R2 = 0.5045, RMSE = 0.0578).

The Lasso Regression model was the most successful and was used to create the final predictions for all players from 2000-2020. In addition, the model was used to predict salaries for the 2022 NBA season (from 2021 NBA data). All the predicted values were gathered and appended to the original dataset. A k-means clustering algorithm was used on the players from the 2021 NBA season to further split players into groups based on their style of play. These clusters allow us to visualize the most over or undervalued of each playstyle. This allows us to see which players can provide the most value in comparison to their salary.

We used Streamlit to create an interactive graphical user interface to allow users to see historical data as well as predictions for the 2022 NBA Season. Users can compare predicted salaries and statistics to observe a player’s value. This tool will help NBA managers determine how much money a player is worth given the stats they obtain. This can remove the problem of overpaying players.

There are two files for the modelling, one in R and one in Python. Users can download the .rmd or .ipynb alongside the attached csv (nba_data.csv) and run the code for identical results. Users that want to use the model to predict data in a future year (i.e 2027) can do so as well by reading in their own csv. The file for the GUI is also attached and users can download and run it on their computer through an IDE (streamlit run gui/NSOT_gui.py). This requires the sklearn, basketball_reference, matplotlib, seaborn, numpy, and pandas libary. Alternatively, this GUI can be accessed here: https://share.streamlit.io/basketkoo98/data_x_nsot/main/gui/NSOT_gui.py 

