import numpy as np

from q_learning_agent import QLearningAgent
from energy_environment import EnergyEnvironment
from models.model import HouseholdEnergyModel
from visualisations import generate_comparison_graphs

class EnergyModel:
    def __init__(self, num_households, num_rooms, season):
        self.num_households = num_households
        self.num_rooms = num_rooms
        self.season = season
        self.household_model = HouseholdEnergyModel(num_households, season)
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
        total_energy_used, total_gas_used, total_cost = 0, 0, 0
        gas_to_kwh_conversion_factor = 11.2
        gas_heating_multiplier = 3.5 if self.season == 'winter' else 1.0

        for _ in range(90):  # Simulate for 90 days
            for hour in range(24):
                action = self.q_learning_agent.choose_action(state)  # Exploit best known action
                state, reward, done, _ = env.step(action)

                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)

                is_cold_hour = hour < 7 or hour >= 19
                temperature_factor = 2.0 if is_cold_hour and self.season == 'winter' else 1.0

                gas_heating = action[-2] * env.appliance_usage.get('gas_heating', 0) * gas_heating_multiplier * temperature_factor
                gas_cooking = action[-1] * env.appliance_usage.get('gas_cooking', 0)
                gas_used_in_kwh = (gas_heating + gas_cooking) * gas_to_kwh_conversion_factor

                total_energy_used += energy_used
                total_gas_used += gas_used_in_kwh
                total_cost += cost + (gas_used_in_kwh * 0.04)

        return total_energy_used, total_gas_used, total_cost

    def test_random_policy(self):
        env = EnergyEnvironment(num_rooms=self.num_rooms, season=self.season)
        state = env.reset()
        total_energy_used, total_gas_used, total_cost = 0, 0, 0
        gas_to_kwh_conversion_factor = 11.2

        for day in range(90):
            for hour in range(24):
                action = [
                    np.random.choice([0, 1]),
                    np.random.choice([0, 1]),
                    1,  # Fridge is always on
                    np.random.choice([0, 1]) if self.season == 'winter' else 0,  # Heating
                    np.random.choice([0, 1], p=[0.8, 0.2]) if 7 <= hour < 9 or 17 <= hour < 19 else 0  # Cooking
                ]
                state, reward, done, _ = env.step(action)

                energy_used = -reward / (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)
                cost = energy_used * (env.peak_price / 100 if 7 <= hour < 17 or 19 <= hour < 23 else env.off_peak_price / 100)

                gas_heating = action[-2] * env.appliance_usage.get('gas_heating', 0)
                gas_cooking = action[-1] * env.appliance_usage.get('gas_cooking', 0)
                gas_used_in_kwh = (gas_heating + gas_cooking) * gas_to_kwh_conversion_factor

                total_energy_used += energy_used
                total_gas_used += gas_used_in_kwh
                total_cost += cost + (gas_used_in_kwh * 0.04)

        return total_energy_used, total_gas_used, total_cost

    def run(self):
        print(f"\nTraining agent for {self.season} season...")
        self.train_agent()
        print(f"Testing agent for {self.season} season...")

        total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained = self.test_agent_exploitation()
        total_electricity_usage_random, total_gas_usage_random, total_cost_random = self.test_random_policy()

        household_data = self.collect_data()
        avg_usage_by_household_size = self.calculate_average_usage(household_data)

        print(f"\nResults with trained agent for {self.num_rooms}-room house:")
        print(f"Electricity and Gas Usage for {self.season.capitalize()}:")
        print(f"Total Electricity Usage: {total_electricity_usage_trained} kWh\nTotal Gas Usage: {total_gas_usage_trained} kWh\nTotal Cost: £{total_cost_trained:.2f}\n")

        print(f"\nResults with random policy:")
        print(f"{self.season.capitalize()} - Total Electricity Usage: {total_electricity_usage_random} kWh\nTotal Gas Usage: {total_gas_usage_random} kWh\nTotal Cost: £{total_cost_random:.2f}\n")

        generate_comparison_graphs(self.season, total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained, total_electricity_usage_random, total_gas_usage_random, total_cost_random, avg_usage_by_household_size)

    def collect_data(self):
        return self.household_model.collect_data()

    def calculate_average_usage(self, household_data):
        num_people_data = {i: {'electricity_usage': 0, 'gas_usage': 0, 'count': 0} for i in range(1, 7)}
        for data in household_data:
            num_people = data[1]
            if num_people in num_people_data:
                num_people_data[num_people]['electricity_usage'] += data[2]
                num_people_data[num_people]['gas_usage'] += data[3]
                num_people_data[num_people]['count'] += 1

        avg_usage_by_household_size = {size: {'avg_electricity_usage': 0, 'avg_gas_usage': 0} for size in num_people_data}
        for size, values in num_people_data.items():
            if values['count'] > 0:
                avg_usage_by_household_size[size]['avg_electricity_usage'] = values['electricity_usage'] / values['count']
                avg_usage_by_household_size[size]['avg_gas_usage'] = values['gas_usage'] / values['count']

        return avg_usage_by_household_size
