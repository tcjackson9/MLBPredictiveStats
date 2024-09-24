import pandas as pd
from sklearn.linear_model import LinearRegression

# Load each player's stats
paul_goldschmidt = pd.read_csv('PaulGoldschmidtStats.csv')
adrian_beltre = pd.read_csv('BeltreStats.csv')
evan_longoria = pd.read_csv('LongoriaStats.csv')
aramis_ramirez = pd.read_csv('AramisRamirezStats.csv')
josh_donaldson = pd.read_csv('JoshDonaldsonStats.csv')
manny_machado = pd.read_csv('MannyStats.csv')

# Combine all the players' stats into one DataFrame
combined_data = pd.concat([paul_goldschmidt, adrian_beltre, evan_longoria, aramis_ramirez, josh_donaldson])

# Strip any leading or trailing spaces from column names
combined_data.columns = combined_data.columns.str.strip()

# Define features and targets for prediction
features = ['Age']  # You can add more features if relevant
targets = ['BA', 'OBP', 'SLG', 'HR', 'OPS', 'RBI']  # Adding HR, OPS, and RBI to the list of stats to predict

# Create a dictionary to store the predictions
predictions = {target: [] for target in targets}

# Loop through the targets and train a model for each stat
for target in targets:
    # Prepare the data for training
    X = combined_data[['Age']]
    y = combined_data[target]

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the stat for Manny Machado for the next 8 years
    current_age = manny_machado['Age'].iloc[-1]
    for year in range(1, 9):  # Predict for the next 8 years
        next_year_age = current_age + year
        predicted_stat = model.predict([[next_year_age]])[0]
        predictions[target].append(predicted_stat)

# Output the predicted stats for Manny Machado for the next 8 years
print(f"Predicted stats for Manny Machado for the next 8 years (based on older players):")
for year in range(1, 9):
    print(f"\nYear {year} (Age: {current_age + year}):")
    for stat, value in predictions.items():
        print(f"{stat}: {value[year - 1]:.3f}")