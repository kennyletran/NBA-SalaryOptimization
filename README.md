# NBA-SalaryOptimization

The NBA has a salary cap, which is a limit to the total amount of money they're allowed to spend on their players. As a result, it's imperative that every contract given matches the player's value (or potential value). There are countless instances where a "bad" contract (a contract given to a player who's production is much lower than their contract value) has backfired and resulted in a loss of an asset. 

A motivating example is Timofey Mozgov who signed a 4 year, $64 million contract with the Los Angeles Lakers in 2016. Desite this large contract, Mozgov only averaged 7.4 PPG and 4.9 TRB in 20.4 MPG, making his contract one of the worst in NBA history. In order to get rid of his salary, the Los Angeles Lakers had to trade away future all-star D'Angelo Russell.

This project seeks to use machine learning to find how valuable a player is. Player statistics will be used to predict salary and this will help NBA front offices and fans determine how much money a player is worth given the stats they obtain. A GUI will be created to display the salary predictions for the 2021-2022 NBA season.


Demo 1: NBA per-game and advanced statistics from the 1990-1991 to the 2019-2020 season were used to predict salary (measured as a % of cap space). NBA players were split by position. The following models were used: Lasso Regression (R2 = 0.4712, RMSE = 0.0583), RFE + Ridge Regression (R2 = 0.4966, RMSE = 0.0578), and Elastic-Net Regression (R2 = 0.5148, RMSE = 0.0569).

Demo 2: In Progress

GUI: In progress

