"""
Visualization utilities for Reverse Stress Testing results.

Provides functions to create visualizations similar to those in the paper:
- Network topology diagrams
- Probability density functions for predicted losses
- Vulnerability heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import networkx as nx
from scipy.stats import gaussian_kde
import pandas as pd


def plot_network_topology(network, figsize=(14, 10), title="Supply Chain Network Topology"):
    """
    Visualize the supply chain network topology.
    
    Creates a layered visualization similar to Figure 3 (right panel) in the paper.
    
    Args:
        network: SupplyChainNetwork object
        figsize: Figure size tuple
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create position layout based on layers
    pos = {}
    for layer in sorted(network.layers.keys()):
        nodes_in_layer = network.layers[layer]
        n_nodes = len(nodes_in_layer)
        
        # Spread nodes horizontally within each layer
        for i, node_id in enumerate(nodes_in_layer):
            x = (i - n_nodes/2) * 2
            y = layer * 3
            pos[node_id] = (x, y)
    
    # Color nodes by country
    countries = list(set(node.country for node in network.nodes.values()))
    color_map = plt.cm.tab20(np.linspace(0, 1, len(countries)))
    country_colors = {country: color_map[i] for i, country in enumerate(countries)}
    
    node_colors = [country_colors[network.nodes[node_id].country] for node_id in network.graph.nodes()]
    
    # Size nodes by baseline quantity
    node_sizes = [network.nodes[node_id].baseline_quantity / 100 for node_id in network.graph.nodes()]
    
    # Draw network
    nx.draw_networkx_nodes(network.graph, pos, node_color=node_colors, 
                           node_size=node_sizes, alpha=0.7, ax=ax)
    
    # Draw edges with width proportional to flow
    edges = network.graph.edges()
    edge_widths = [network.graph[u][v]['quantity_flow'] / 5000 for u, v in edges]
    nx.draw_networkx_edges(network.graph, pos, width=edge_widths, 
                           alpha=0.3, arrows=True, arrowsize=10, ax=ax)
    
    # Draw labels
    labels = {node_id: network.nodes[node_id].country for node_id in network.graph.nodes()}
    nx.draw_networkx_labels(network.graph, pos, labels, font_size=8, ax=ax)
    
    # Add layer labels
    for layer in sorted(network.layers.keys()):
        good_type = network.nodes[network.layers[layer][0]].good_type
        ax.text(-10, layer * 3, f"Layer {layer}\n{good_type}", 
               fontsize=10, fontweight='bold', va='center')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax


def plot_layer_quantities(network, figsize=(12, 6)):
    """
    Plot quantities flowing through each layer (Figure 3, left panel).
    
    Args:
        network: SupplyChainNetwork object
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate quantities by layer and country
    layer_data = {}
    for layer in sorted(network.layers.keys()):
        layer_data[layer] = {}
        for node_id in network.layers[layer]:
            node = network.nodes[node_id]
            country = node.country
            quantity = node.baseline_quantity
            
            if country in layer_data[layer]:
                layer_data[layer][country] += quantity
            else:
                layer_data[layer][country] = quantity
    
    # Create stacked bar chart
    countries = list(set(country for layer_dict in layer_data.values() for country in layer_dict.keys()))
    
    bottom = np.zeros(len(layer_data))
    for country in countries:
        quantities = [layer_data[layer].get(country, 0) for layer in sorted(layer_data.keys())]
        ax.bar(range(len(layer_data)), quantities, bottom=bottom, label=country, alpha=0.8)
        bottom += quantities
    
    # Format plot
    layer_names = [network.nodes[network.layers[layer][0]].good_type.replace('_', ' ').title() 
                   for layer in sorted(layer_data.keys())]
    ax.set_xticks(range(len(layer_data)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('Quantity (kg)', fontsize=12)
    ax.set_title('Quantities Flowing Through Each Layer', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_scenario_pdfs(rst_results, network, loss_level=0.05, figsize=(16, 12)):
    """
    Plot probability density functions for predicted losses across the supply chain.
    
    Similar to Figure 4 in the paper.
    
    Args:
        rst_results: Results from ReverseStressTester.run_full_rst()
        network: SupplyChainNetwork object
        loss_level: Which loss level to plot (e.g., 0.05 for 5%)
        figsize: Figure size tuple
    """
    scenarios = rst_results[loss_level]
    
    # Get all layers
    layers = sorted(scenarios.keys())
    n_layers = len(layers)
    
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=False)
    if n_layers == 1:
        axes = [axes]
    
    for idx, layer in enumerate(layers):
        ax = axes[idx]
        layer_scenarios = scenarios[layer]
        
        if not layer_scenarios:
            continue
        
        # Aggregate losses by country
        country_losses = {}
        for scenario in layer_scenarios:
            for node_id, loss_qty in scenario.loss_predictions.items():
                country = network.nodes[node_id].country
                if country not in country_losses:
                    country_losses[country] = []
                country_losses[country].append((loss_qty, scenario.probability))
        
        # Plot PDFs for top 5 countries
        sorted_countries = sorted(country_losses.items(), 
                                 key=lambda x: max(loss for loss, _ in x[1]), 
                                 reverse=True)[:5]
        
        for country, losses in sorted_countries:
            if not losses:
                continue
            
            # Create weighted KDE
            quantities = np.array([loss for loss, _ in losses])
            weights = np.array([prob for _, prob in losses])
            
            if len(quantities) > 1 and np.std(quantities) > 0:
                # Use KDE for smooth distribution
                kde = gaussian_kde(quantities, weights=weights)
                x_range = np.linspace(max(0, quantities.min() - quantities.std()), 
                                    quantities.max() + quantities.std(), 200)
                y = kde(x_range)
                ax.plot(x_range, y, label=country, linewidth=2)
                ax.fill_between(x_range, y, alpha=0.2)
            else:
                # Single point or no variation - plot as vertical line
                ax.axvline(quantities[0], label=country, linewidth=2)
        
        # Format subplot
        good_type = network.nodes[network.layers[layer][0]].good_type.replace('_', ' ').title()
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'Layer {layer}: {good_type}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Predicted Loss (kg)', fontsize=12)
    fig.suptitle(f'Probability Distributions for {loss_level*100}% Loss Scenario', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, axes


def plot_vulnerability_heatmap(rst_results, network, figsize=(12, 8)):
    """
    Create a heatmap showing vulnerability scores across loss levels.
    
    Args:
        rst_results: Results from ReverseStressTester.run_full_rst()
        network: SupplyChainNetwork object
        figsize: Figure size tuple
    """
    from reverse_stress_testing import ReverseStressTester
    
    # Compute vulnerability scores for each loss level
    loss_levels = sorted(rst_results.keys())
    all_countries = set()
    vulnerability_matrix = {}
    
    # Create dummy RST object to use get_vulnerability_scores method
    rst = ReverseStressTester(network)
    
    for loss_level in loss_levels:
        vuln_df = rst.get_vulnerability_scores(rst_results[loss_level])
        vulnerability_matrix[loss_level] = {}
        
        for _, row in vuln_df.iterrows():
            country = row['country']
            all_countries.add(country)
            vulnerability_matrix[loss_level][country] = row['vulnerability_score']
    
    # Create matrix for heatmap
    countries = sorted(all_countries)
    matrix = np.zeros((len(countries), len(loss_levels)))
    
    for i, country in enumerate(countries):
        for j, loss_level in enumerate(loss_levels):
            matrix[i, j] = vulnerability_matrix[loss_level].get(country, 0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                xticklabels=[f'{l*100}%' for l in loss_levels],
                yticklabels=countries, ax=ax, cbar_kws={'label': 'Vulnerability Score'})
    
    ax.set_xlabel('Loss Scenario', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Vulnerability Scores Across Loss Scenarios', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_comparison_table(rst_results, network, figsize=(14, 10)):
    """
    Create a comparison table similar to Table 1 in the paper.
    
    Shows top 5 contributors at each layer for each loss scenario.
    
    Args:
        rst_results: Results from ReverseStressTester.run_full_rst()
        network: SupplyChainNetwork object
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    loss_levels = sorted(rst_results.keys())
    layers = sorted(set(layer for scenarios in rst_results.values() for layer in scenarios.keys()))
    
    # Create table data
    table_data = []
    headers = ['Loss\nScenario'] + [f'Layer {l}\n{network.nodes[network.layers[l][0]].good_type.replace("_", " ").title()}' 
                                     for l in layers]
    
    for loss_level in loss_levels:
        row = [f'{loss_level*100}%']
        scenarios = rst_results[loss_level]
        
        for layer in layers:
            if layer not in scenarios or not scenarios[layer]:
                row.append('')
                continue
            
            # Get top 5 contributors
            most_likely = scenarios[layer][0]
            sorted_losses = sorted(
                most_likely.loss_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Format as text
            text = '\n'.join([f"{network.nodes[nid].country}: {loss:.1f}" 
                             for nid, loss in sorted_losses])
            row.append(text)
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    colWidths=[0.1] + [0.22] * len(layers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)
    
    # Style headers
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
    for i, row in enumerate(table_data):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i % len(colors)])
    
    plt.title('Top 5 Contributors Across Supply Chain Tiers', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig, ax


def create_full_report(network, rst_results, output_dir='./rst_results'):
    """
    Generate a complete report with all visualizations.
    
    Args:
        network: SupplyChainNetwork object
        rst_results: Results from ReverseStressTester.run_full_rst()
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Network topology
    print("  - Network topology...")
    fig, _ = plot_network_topology(network)
    fig.savefig(f'{output_dir}/network_topology.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Layer quantities
    print("  - Layer quantities...")
    fig, _ = plot_layer_quantities(network)
    fig.savefig(f'{output_dir}/layer_quantities.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. PDFs for each loss scenario
    for loss_level in sorted(rst_results.keys()):
        print(f"  - PDFs for {loss_level*100}% loss...")
        fig, _ = plot_scenario_pdfs(rst_results, network, loss_level)
        fig.savefig(f'{output_dir}/pdfs_{int(loss_level*100)}pct.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 4. Vulnerability heatmap
    print("  - Vulnerability heatmap...")
    fig, _ = plot_vulnerability_heatmap(rst_results, network)
    fig.savefig(f'{output_dir}/vulnerability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Comparison table
    print("  - Comparison table...")
    fig, _ = plot_comparison_table(rst_results, network)
    fig.savefig(f'{output_dir}/comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nAll visualizations saved to {output_dir}/")
