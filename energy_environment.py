import gym
from gym import spaces
import numpy as np


class EnergyEnvironment(gym.Env):
    def __init__(self, num_rooms=1, season='winter'):
        super(EnergyEnvironment, self).__init__()

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2])  # Each can be 0 or 1
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2, 2])

        self.num_rooms = num_rooms
        self.season = season
        self.current_hour = 0
        self.current_day = 0
        self.state = [0, 0, 1, 0, 0]  # Initial state, fridge is always on

        self.set_seasonal_parameters()

    def set_seasonal_parameters(self):
        # Electricity prices (pence per kWh)
        self.peak_price = 24.50  # pence per kWh
        self.off_peak_price = 22.36  # pence per kWh
        self.gas_price = 6.24  # pence per kWh

        if self.season == 'winter':
            self.appliance_usage = {
                'washing_machine': 0.12,
                'fridge': 0.10,
                'lighting': 0.08,
                'heating': 0.20,
                'gas_heating': 0.5,
                'gas_cooking': 0.4,
            }
        elif self.season == 'summer':
            self.appliance_usage = {
                'washing_machine': 0.10,
                'fridge': 0.12,
                'lighting': 0.05,
                'cooling': 0.10,
                'gas_cooking': 0.6,
            }
        else:
            raise ValueError("Season must be 'winter' or 'summer'")

    def reset(self):
        self.state = [0, 0, 1, 0, 0]
        self.current_hour = 0
        self.current_day = 0
        return np.array(self.state)

    def step(self, action):
        light, washing_machine, fridge, gas_heating, gas_cooking = action
        fridge = 1

        electricity_used = (
                light * 0.5 * self.num_rooms * self.appliance_usage['lighting'] +
                washing_machine * 1.5 * self.appliance_usage['washing_machine'] +
                fridge * 1 * self.appliance_usage['fridge']
        )

        gas_used = (
                gas_heating * 3 * self.appliance_usage.get('gas_heating', 0) +
                gas_cooking * 3 * self.appliance_usage.get('gas_cooking', 0)
        )

        if 'heating' in self.appliance_usage:
            electricity_used += 2 * self.appliance_usage['heating']
        if 'cooling' in self.appliance_usage:
            electricity_used += 1 * self.appliance_usage['cooling']

        if 7 <= self.current_hour < 17 or 19 <= self.current_hour < 23:
            electricity_price = self.peak_price / 100
        else:
            electricity_price = self.off_peak_price / 100

        gas_price = self.gas_price / 100
        reward = -(electricity_used * electricity_price + gas_used * gas_price)

        self.state = [light, washing_machine, fridge, gas_heating, gas_cooking]
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        done = self.current_day >= 90

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        print(f"Day: {self.current_day}, Hour: {self.current_hour}, State: {self.state}")
