import pandas as pd
import matplotlib.pyplot as plt

# Define seasons
seasons = ['winter', 'summer']

# Create a figure for the plots
plt.figure(figsize=(10, 7))

for i, season in enumerate(seasons):
    # Load the collected data for the season
    df = pd.read_csv(f"household_energy_data_{season}.csv")

    # Prepare the data for plotting
    house_sizes = df["Num People"]
    electricity_usage = df["Electricity Usage"]
    gas_usage = df["Gas Usage"]

    # Create subplots
    plt.subplot(1, 2, i + 1)  # Two subplots: one for each season (side by side)

    # Plot both electricity and gas usage on the same graph
    plt.scatter(house_sizes, electricity_usage, c='blue', label='Electricity Usage (kWh)', marker='o')
    plt.scatter(house_sizes, gas_usage, c='red', label='Gas Usage (kWh)', marker='x')

    # Add labels and title
    plt.xlabel('House Size (Number of People)')
    plt.ylabel('Usage (kWh)')
    plt.title(f'{season.capitalize()} - House Size vs Usage')
    plt.legend()

# Adjust layout for better spacing
plt.tight_layout()
# Display the plots
plt.show()
