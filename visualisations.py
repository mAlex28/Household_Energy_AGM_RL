import matplotlib.pyplot as plt
import numpy as np


def generate_comparison_graphs(season, total_electricity_usage_trained, total_gas_usage_trained, total_cost_trained,
                               total_electricity_usage_random, total_gas_usage_random, total_cost_random,
                               avg_usage_by_household_size):
    # Trained Agent vs Random Policy

    # Create labels for the x-axis
    labels = ['Trained Agent', 'Random Policy']
    x = range(2)

    # Create a figure and axis objects for plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot electricity usage on the primary y-axis (left side)
    ax1.plot(x, [total_electricity_usage_trained, total_electricity_usage_random], marker='o', color='blue',
             label='Electricity Usage (kWh)', linewidth=2)
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Electricity Usage (kWh)', color='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Set limits for electricity usage
    ax1.set_ylim(min(total_electricity_usage_trained, total_electricity_usage_random) * 0.9,
                 max(total_electricity_usage_trained, total_electricity_usage_random) * 1.1)

    # Create a secondary y-axis for gas usage (right side)
    ax2 = ax1.twinx()
    ax2.plot(x, [total_gas_usage_trained, total_gas_usage_random], marker='s', color='red', label='Gas Usage (kWh)',
             linewidth=2)
    ax2.set_ylabel('Gas Usage (kWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set limits for gas usage
    ax2.set_ylim(min(total_gas_usage_trained, total_gas_usage_random) * 0.9,
                 max(total_gas_usage_trained, total_gas_usage_random) * 1.1)

    # Create a third axis for cost
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis to avoid overlap
    ax3.plot(x, [total_cost_trained, total_cost_random], marker='^', color='green', label='Total Cost (£)', linewidth=2)
    ax3.set_ylabel('Total Cost (£)', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # Set limits for cost
    ax3.set_ylim(min(total_cost_trained, total_cost_random) * 0.9,
                 max(total_cost_trained, total_cost_random) * 1.1)

    # Set title and layout
    plt.title(f'{season.capitalize()} - Electricity, Gas, and Cost Comparison')

    # Add legends for all y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='center right')

    # Save the graph
    plt.tight_layout()
    plt.savefig(f'visualisations/{season}_electricity_gas_cost_comparison_graph.png')
    plt.show()

    # ABM vs Trained Agent vs Random Policy
    policies = ['ABM', 'Trained Agent', 'Random Policy']

    # Calculate the overall average electricity and gas usage from ABM data
    total_abm_electricity_usage = 0
    total_abm_gas_usage = 0
    abm_count = 0

    for size, values in avg_usage_by_household_size.items():
        total_abm_electricity_usage += values['avg_electricity_usage']
        total_abm_gas_usage += values['avg_gas_usage']
        abm_count += 1

    avg_electricity_usage_abm = total_abm_electricity_usage / abm_count
    avg_gas_usage_abm = total_abm_gas_usage / abm_count

    electricity_usage = [avg_electricity_usage_abm, total_electricity_usage_trained, total_electricity_usage_random]
    gas_usage = [avg_gas_usage_abm, total_electricity_usage_random, total_gas_usage_random]

    x = np.arange(len(policies))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, electricity_usage, width, label='Electricity Usage')
    rects2 = ax.bar(x + width / 2, gas_usage, width, label='Gas Usage')

    ax.set_xlabel('Policy')
    ax.set_ylabel('Energy Usage (kWh)')
    ax.set_title(f'Energy Usage Comparison during {season.capitalize()}')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.legend()

    fig.tight_layout()
    plt.savefig(f'visualisations/energy_usage_comparison_{season}.png')
    plt.show()

