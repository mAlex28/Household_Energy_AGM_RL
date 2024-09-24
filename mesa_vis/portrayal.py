def portrayAgent(agent):
    if agent.house_type == 'Flat/1-bedroom':
        portrayal = {"Shape": "rect", "Color": "blue", "Filled": "true", "Layer": 0, "w": 0.8, "h": 0.8}
    elif agent.house_type == 'Medium 2-3 bedroom':
        portrayal = {"Shape": "rect", "Color": "green", "Filled": "true", "Layer": 0, "w": 0.8, "h": 0.8}
    else:
        portrayal = {"Shape": "rect", "Color": "red", "Filled": "true", "Layer": 0, "w": 0.8, "h": 0.8}
    return portrayal