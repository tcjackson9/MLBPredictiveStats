import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load Manny Machado's stats
manny_machado = pd.read_csv("MannyStats.csv")

# Load the statistics of other players who played past age 31
files = ["PaulGoldschmidtStats.csv", "BeltreStats.csv", "LongoriaStats.csv", "AramisRamirezStats.csv", "JoshDonaldsonStats.csv"]
dataframes = []

for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all player data
combined_data = pd.concat(dataframes)

# Prepare the dataset for prediction
targets = ['BA', 'OBP', 'SLG', 'HR', 'OPS', 'RBI']

predictions = {target: [] for target in targets}

# Loop through the targets and train a polynomial regression model for each stat
for target in targets:
    # Prepare the data for training
    X = combined_data[['Age']]
    y = combined_data[target]

    # Create a polynomial regression model
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)

    # Predict the stat for Manny Machado for the next 8 years
    current_age = manny_machado['Age'].iloc[-1]
    for year in range(1, 9):
        next_year_age = current_age + year
        predicted_stat = model.predict([[next_year_age]])[0]
        predictions[target].append(predicted_stat)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions)
predictions_df['Age'] = range(current_age + 1, current_age + 9)

# Print predictions
print(predictions_df)

# Optionally plot the predictions for each stat
for target in targets:
    plt.figure()
    plt.title(f'Predicted {target} for Manny Machado Over the Next 8 Years')
    plt.plot(predictions_df['Age'], predictions_df[target], marker='o')
    plt.xlabel('Age')
    plt.ylabel(target)
    plt.grid()
    plt.show()