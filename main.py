from models.agent import Household
from models.model import HouseholdEnergyModel
from q_learning_agent import QLearningAgent
from energy_environment import EnergyEnvironment
import numpy as np
from visualisations import generate_comparison_graphs

class EnergyModel:
    def __init__(self, num_households, num_rooms, season):
        self.num_households = num_households
        self.num_rooms = num_rooms
        self.season = season
        self.household_model = HouseholdEnergyModel(num_households, season)
        
        # Updated to include 5 actions (light, washing_machine, fridge, gas_heating, gas_cooking)
        self.q_learning_agent = QLearningAgent(state_size=[2, 2, 2, 2, 2], action_size=[2, 2, 2, 2, 2])

    def train_agent(self, episodes=2000):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)

        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.q_learning_agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.q_learning_agent.learn(state, action, reward, next_state)
                state = next_state

    def test_agent_exploitation(self):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_gas_used = 0
        total_cost = 0

        # Conversion factor from gas units to kWh
        gas_to_kwh_conversion_factor = 11.2

        # Multipliers for different heating levels and temperature impact
        gas_heating_multiplier = 2.5 if self.season == 'winter' else 1.0  # Increased multiplier for winter

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = self.q_learning_agent.choose_action(state)  # Exploit best known action
                state, reward, done, _ = env.step(action)
            
                # Calculate electricity usage and cost
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)

                # Simulate gas usage based on gas_heating and gas_cooking actions
                # Introduce temperature and peak heating periods to vary gas usage
                is_cold_hour = hour < 7 or hour >= 19  # Assume colder usage in early mornings and late evenings
                temperature_factor = 2.0 if is_cold_hour and self.season == 'winter' else 1.0  # Higher heating at colder hours

                # Gas heating usage based on time of day and temperature factor
                gas_heating = action[-2] * env.appliance_usage.get('gas_heating', 0) * gas_heating_multiplier * temperature_factor
                gas_cooking = action[-1] * env.appliance_usage.get('gas_cooking', 0)  # Cooking gas usage remains fixed
            
                # Total gas used
                gas_used_in_units = gas_heating + gas_cooking
                gas_used_in_kwh = gas_used_in_units * gas_to_kwh_conversion_factor

                # Update totals
                total_energy_used += energy_used
                total_gas_used += gas_used_in_kwh
                total_cost += cost + (gas_used_in_kwh * 0.04)  # Assuming 0.04 £ per unit of gas

        return total_energy_used, total_gas_used, total_cost


    def run(self):
        print(f"\nTraining agent for {self.season} season...")
        self.train_agent()
        print(f"Testing agent for {self.season} season...")

        # Test agent performance
        total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained = self.test_agent_exploitation()

        # Test random policy performance
        total_electricity_usage_random, total_gas_usage_random, total_cost_random = self.test_random_policy()

        # Print results
        print("\nResults with trained agent:")
        print(f"Electricity and Gas Usage and Cost for {self.season.capitalize()}")
        print(f"Total Electricity Usage: {total_electricity_usage_trained} kWh\n"
              f"Total Gas Usage: {total_gas_usage_trained} kWh\n"
              f"Total Cost: £{total_cost_trained:.2f}\n")

        print("\nResults with random policy:")
        print(f"{self.season.capitalize()} - Total Electricity Usage: {total_electricity_usage_random} kWh\n"
              f"Total Gas Usage: {total_gas_usage_random} kWh\n"
              f"Total Cost: £{total_cost_random:.2f}\n")

        print("\nPerformance improvement with trained agent:")
        print(f"{self.season.capitalize()} - Total Electricity Usage Reduction (vs Random): {total_electricity_usage_random - total_electricity_usage_trained} kWh\n"
              f"Total Gas Usage Reduction (vs Random): {total_gas_usage_random - total_gas_usage_trained} kWh\n"
              f"Total Cost Reduction (vs Random): £{total_cost_random - total_cost_trained:.2f}\n")
        
        # Generate and save comparison graphs
        generate_comparison_graphs(self.season, total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained,
                                    total_electricity_usage_random, total_gas_usage_random, total_cost_random)

    def test_random_policy(self):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)
        state = env.reset()
        total_energy_used = 0
        total_gas_used = 0
        total_cost = 0

        # Conversion factor from gas units to kWh
        gas_to_kwh_conversion_factor = 11.2

        for day in range(90):  # Simulate for 90 days
            for hour in range(24):
                # Randomly choose actions, but limit gas appliance usage
                # Light, washing machine, fridge can be uniformly random
                light_action = np.random.choice([0, 1])
                washing_machine_action = np.random.choice([0, 1])
                fridge_action = 1  # Assume fridge is always on

                # Gas heating: only likely to be on during colder hours (e.g., morning, evening) in winter
                if self.season == 'winter':
                    gas_heating_action = np.random.choice([0, 1], p=[0.7, 0.3] if hour < 7 or hour >= 19 else [0.9, 0.1])
                else:
                    gas_heating_action = 0  # No heating needed in summer

                # Gas cooking: Assume it is only used a few hours per day (e.g., morning and evening)
                if 7 <= hour < 9 or 17 <= hour < 19:  # Typical cooking hours
                    gas_cooking_action = np.random.choice([0, 1], p=[0.6, 0.4])
                else:
                    gas_cooking_action = 0

                action = [light_action, washing_machine_action, fridge_action, gas_heating_action, gas_cooking_action]
                state, reward, done, _ = env.step(action)

                # Calculate electricity usage and cost
                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)

                # Simulate gas usage
                gas_heating = gas_heating_action * env.appliance_usage.get('gas_heating', 0)
                gas_cooking = gas_cooking_action * env.appliance_usage.get('gas_cooking', 0)
                gas_used_in_units = gas_heating + gas_cooking

                # Convert gas usage to kWh
                gas_used_in_kwh = gas_used_in_units * gas_to_kwh_conversion_factor

                total_energy_used += energy_used
                total_gas_used += gas_used_in_kwh
                total_cost += cost + (gas_used_in_kwh * 0.04)  # Assuming 0.04 £ per kWh for gas

        return total_energy_used, total_gas_used, total_cost


if __name__ == "__main__":
    num_households = 100
    num_rooms = 3
    seasons = ['winter', 'summer']

    for season in seasons:
        print(f"Running model for {season}...")
        model = EnergyModel(num_households, num_rooms, season=season)
        model.run()
