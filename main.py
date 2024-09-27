import pandas as pd
from energy_model import EnergyModel
from energy_comparison_graph import generate_season_comparison_graphs

if __name__ == "__main__":
    num_households = 137
    num_rooms = 3
    seasons = ['winter', 'summer']

    # Initialize storage for results
    results = {
        'winter_trained': None,
        'winter_random': None,
        'summer_trained': None,
        'summer_random': None
    }

    for season in seasons:
        print(f"Running model for {season}...")
        model = EnergyModel(num_households, num_rooms, season=season)
        model.run()
        household_data = model.collect_data()

        # Save data to CSV file
        df = pd.DataFrame(household_data,
                          columns=["House Type", "Num People", "Electricity Usage", "Gas Usage", "Energy Saving"])
        df.to_csv(f"data/household_energy_data_{season}.csv", index=False)

        # Store results
        results[f'{season}_trained'] = {
            'cost': model.test_agent_exploitation()[2]
        }
        results[f'{season}_random'] = {
            'cost': model.test_random_policy()[2]
        }

    # Generate comparison graphs with all collected data
    generate_season_comparison_graphs([
        results['winter_trained'],
        results['winter_random'],
        results['summer_trained'],
        results['summer_random']
    ])
