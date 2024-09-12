import matplotlib.pyplot as plt

def generate_season_comparison_graphs(data):
    seasons = ['Winter', 'Summer']
    policies = ['Trained Agent', 'Random Policy']

    # Data unpacking
    winter_trained_cost = data[0]['cost']
    winter_random_cost = data[1]['cost']
    summer_trained_cost = data[2]['cost']
    summer_random_cost = data[3]['cost']

    # Create bar chart for costs
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].bar(policies, [winter_trained_cost, winter_random_cost], color=['blue', 'orange'])
    ax[0].set_title('Energy Cost during Winter')
    ax[0].set_ylabel('£')

    ax[1].bar(policies, [summer_trained_cost, summer_random_cost], color=['blue', 'orange'])
    ax[1].set_title('Energy Cost during Summer')
    ax[1].set_ylabel('£')

    plt.suptitle('Comparison of Energy Costs')
    plt.tight_layout()
    plt.savefig('energy_cost_comparison.png')
    plt.show()
