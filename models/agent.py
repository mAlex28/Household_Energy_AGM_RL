from mesa import Agent
import numpy as np


class Household(Agent):
    def __init__(self, unique_id, model, house_type, num_people):  # Household Attributes
        super().__init__(unique_id, model)
        self.house_type = house_type
        self.num_people = num_people
        self.energy_saving = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])  # 30% of the households engage in energy
        # savings (Randomly generated)
        self.season = model.season

        # Initial electricity and gas usage
        self.electricity_usage = self.calculate_energy_usage('electricity')
        self.gas_usage = self.calculate_energy_usage('gas')

    def calculate_energy_usage(self, energy_type):
        # Mean and standard deviation based on house type
        mean, std = self.model.energy_usage_params[self.house_type][energy_type]
        usage = np.random.normal(mean, std)  # Calculating the usage
        if self.season == 'winter':
            # Energy usage is 36% higher in winter
            usage *= 1.36
        if self.energy_saving == 'Yes':
            # If the household is engaged in energy savings, reduce the usage by 10%
            usage *= 0.9
        return max(usage, 0)

    '''
        Randomly choose a new possible position from all the available positions and move the agent there
    '''
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    '''
       Recalculate the energy and gas usage everytime agent is moved to a new position
    '''
    def step(self):
        self.move()
        self.electricity_usage = self.calculate_energy_usage('electricity')
        self.gas_usage = self.calculate_energy_usage('gas')
