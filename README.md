# Household Energy Simulation

This project simulates household energy usage using an agent-based model and reinforcemnt leanring.

## Project Structure

- `requirements.txt`: Lists project dependencies.
- `models/`: Contains the agent and model classes.
- `visualisation.py`:  Contains functions to generate comparison graphs for trained and random policy performances.
- `energy_environment.py`: The EnergyEnvironment class, defining the simulation environment using OpenAI's Gym framework.
- `q_learning_agent.py`: Implements the QLearningAgent class, which uses Q-learning to make decisions based on the environment's state.
- `energy_model.py`: Integrates the agent, environment, and reinforcement learning to run the simulation and evaluate different policies.
- `main.py`: Entry point to run the simulation.
- `data/`: Datasets of generated household energy usage and reductions
- `visualisations/`: Generated graphs

## How to Run

1. Install dependencies:
```python
pip install -r requirements.txt
```
2. Run the simulation:
```python
python main.py
```
