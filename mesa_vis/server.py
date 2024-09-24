from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider

from mesa_vis.portrayal import portrayAgent
from models.model import HouseholdEnergyModel


grid = CanvasGrid(portrayAgent, 10, 10, 300, 300)

chart = ChartModule(
    [{"Label": "Electricity Usage", "Color": "Black"}],
    data_collector_name="datacollector"
)

# Assuming default season is 'winter'
server = ModularServer(
    HouseholdEnergyModel,
    [grid, chart],
    "Household Energy Model",
    {"num_households": Slider('Number of households', 137, 10, 200, 1), 'season': 'winter'}
)



if __name__ == "__main__":
    # Set the season manually
    server.model_kwargs = {"num_households": 137, "season": 'winter'}

    # Run the server
    server.port = 8521
    server.launch()