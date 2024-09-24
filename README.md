MLB Player Stat Prediction
Overview
This project aims to predict the future performance of MLB players, specifically focusing on Manny Machado's statistics over the next eight years. The project utilizes historical player data, including age and various performance metrics, to create predictive models using polynomial regression.

Files
original_prediction.py: This script predicts Manny Machado's stats based on historical data from players who played past age 31, using a linear regression model.
polynomial_regression.py: This script predicts Manny Machado's stats using polynomial regression, allowing for a non-linear relationship between age and performance metrics.
Datasets

The project utilizes the following datasets:
MannyStats.csv: Contains Manny Machado's career statistics, including Year, Age, Team, Games, AB, Runs, Hits, 2B, 3B, HR, RBI, SB, BB, SO, BA, OBP, SLG, OPS, OPS+.
PaulGoldschmidtStats.csv: Contains statistics for Paul Goldschmidt.
BeltreStats.csv: Contains statistics for Adrian Beltre.
LongoriaStats.csv: Contains statistics for Evan Longoria.
AramisRamirezStats.csv: Contains statistics for Aramis Ramirez.
JoshDonaldsonStats.csv: Contains statistics for Josh Donaldson.
These datasets provide the historical context needed to predict Manny Machado's performance trends as he ages.

Getting Started
Prerequisites
Make sure you have Python installed along with the following libraries:
pandas
numpy
scikit-learn
matplotlib
You can install the required libraries using pip:

Results
The scripts will output Manny Machado's predicted statistics for the next eight years. The polynomial regression script also provides visualizations of the predicted stats against age.
