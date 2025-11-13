"""
Complete demonstration of Reverse Stress Testing for Supply Chain Resilience.

This script demonstrates the full workflow from the paper:
1. Network construction
2. Single-layer reverse stress test
3. Backpropagation of losses
4. Multiple scenario generation
5. Probabilistic analysis
6. Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reverse_stress_testing import (
    SupplyChainNetwork, SupplyChainNode, SupplyChainEdge,
    ReverseStressTester, create_example_copper_network
)
from rst_data_processing import ComtradeDataProcessor, create_synthetic_comtrade_data
from rst_visualization import (
    plot_network_topology, plot_layer_quantities,
    plot_scenario_pdfs, plot_vulnerability_heatmap,
    plot_comparison_table, create_full_report
)


def run_basic_example():
    """Run basic example with pre-built network"""
    print("\n" + "="*70)
    print("BASIC EXAMPLE: Pre-built Copper Supply Chain Network")
    print("="*70)
    
    # Create example network
    print("\n1. Creating copper supply chain network...")
    network = create_example_copper_network()
    print(f"   Created network with {len(network.nodes)} nodes and {len(network.edges)} edges")
    print(f"   Layers: {sorted(network.layers.keys())}")
    
    # Initialize RST
    print("\n2. Initializing Reverse Stress Tester...")
    rst = ReverseStressTester(network, q=0.5)
    
    # Add synthetic historical data
    print("\n3. Adding synthetic historical data...")
    np.random.seed(42)
    for node_id in network.nodes:
        suppliers = network.get_suppliers(node_id)
        if suppliers:
            n_months = 24
            data = {}
            for supplier in suppliers:
                edge = network.graph[supplier][node_id]
                baseline = edge['quantity_flow']
                # Add realistic variation
                quantities = baseline * (1 + np.random.randn(n_months) * 0.1)
                quantities = np.maximum(quantities, 0)  # No negative quantities
                data[supplier] = quantities
            rst.add_historical_data(node_id, pd.DataFrame(data))
    print(f"   Added historical data for {len(rst.historical_data)} nodes")
    
    # Run RST for multiple loss scenarios
    print("\n4. Running Reverse Stress Testing...")
    loss_levels = [0.05, 0.20, 0.50]  # 5%, 20%, 50%
    print(f"   Loss scenarios: {[f'{l*100}%' for l in loss_levels]}")
    
    results = rst.run_full_rst(loss_levels, num_samples=100)
    
    # Display results
    print("\n5. Results Summary:")
    print("   " + "-"*66)
    
    for loss_level in loss_levels:
        print(f"\n   {loss_level*100}% Loss Scenario:")
        scenarios = results[loss_level]
        
        for layer in sorted(scenarios.keys()):
            layer_scenarios = scenarios[layer]
            if not layer_scenarios:
                continue
            
            good_type = network.nodes[network.layers[layer][0]].good_type
            print(f"\n   Layer {layer} ({good_type}):")
            print(f"   {'Country':<15} {'Predicted Loss (kg)':>20}")
            print(f"   {'-'*35}")
            
            # Show most likely scenario
            most_likely = layer_scenarios[0]
            sorted_losses = sorted(
                most_likely.loss_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for node_id, loss_qty in sorted_losses:
                node = network.nodes[node_id]
                print(f"   {node.country:<15} {loss_qty:>20,.2f}")
        
        # Vulnerability scores
        print(f"\n   Top 5 Most Vulnerable Nodes:")
        print(f"   {'Country':<15} {'Good Type':<20} {'Score':>15}")
        print(f"   {'-'*50}")
        vuln_scores = rst.get_vulnerability_scores(scenarios)
        for _, row in vuln_scores.head(5).iterrows():
            print(f"   {row['country']:<15} {row['good_type']:<20} {row['vulnerability_score']:>15,.2f}")
    
    return network, results


def run_data_driven_example():
    """Run example with synthetic Comtrade data"""
    print("\n" + "="*70)
    print("DATA-DRIVEN EXAMPLE: Building Network from Synthetic Trade Data")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic UN Comtrade data...")
    df = create_synthetic_comtrade_data(n_months=24)
    print(f"   Generated {len(df)} trade records")
    print(f"   Countries: {df['reporter'].unique()}")
    print(f"   HS Codes: {df['hs_code'].unique()}")
    
    # Initialize processor
    print("\n2. Processing trade data...")
    processor = ComtradeDataProcessor()
    processor.raw_data = df
    
    # Set good type mapping
    hs_mapping = {
        'HS2603': 'copper_ore',
        'HS7402': 'unrefined_copper',
        'HS7403': 'refined_copper',
        'HS7408': 'copper_wire'
    }
    processor.set_good_type_mapping(hs_mapping)
    
    # Process data
    df = processor.aggregate_monthly_flows(df)
    df = processor.handle_missing_values(df)
    df = processor.remove_transient_suppliers(df, min_periods=3)
    print(f"   Processed data: {len(df)} records")
    
    # Build network
    print("\n3. Building supply chain network...")
    good_hierarchy = ['copper_ore', 'unrefined_copper', 'refined_copper', 'copper_wire']
    network = processor.build_network_from_data(df, 'USA', good_hierarchy)
    
    # Create time series
    print("\n4. Creating historical time series...")
    time_series = processor.create_historical_time_series(df, network)
    print(f"   Created time series for {len(time_series)} nodes")
    
    # Run RST
    print("\n5. Running Reverse Stress Testing...")
    rst = ReverseStressTester(network, q=0.5)
    
    # Add time series data
    for node_id, ts_df in time_series.items():
        rst.add_historical_data(node_id, ts_df)
    
    loss_levels = [0.05, 0.20, 0.50]
    results = rst.run_full_rst(loss_levels, num_samples=50)
    
    print("\n6. Results Summary:")
    for loss_level in loss_levels:
        print(f"\n   {loss_level*100}% Loss Scenario:")
        vuln_scores = rst.get_vulnerability_scores(results[loss_level])
        print(f"   Top 5 Vulnerable Countries:")
        print(vuln_scores[['country', 'good_type', 'vulnerability_score']].head().to_string(index=False))
    
    return network, results


def create_visualizations(network, results):
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    create_full_report(network, results, output_dir='./rst_results')
    
    print("\n✓ All visualizations saved to ./rst_results/")
    print("\nGenerated files:")
    print("  - network_topology.png")
    print("  - layer_quantities.png")
    print("  - pdfs_5pct.png")
    print("  - pdfs_20pct.png")
    print("  - pdfs_50pct.png")
    print("  - vulnerability_heatmap.png")
    print("  - comparison_table.png")


def demonstrate_single_layer_rst():
    """Demonstrate single-layer RST calculation"""
    print("\n" + "="*70)
    print("DEMONSTRATION: Single-Layer Reverse Stress Test")
    print("="*70)
    
    print("\nThis demonstrates the core RST equation from the paper:")
    print("  a_j = (L / sum(D_j)) * D_j * 1_vector")
    print("\nwhere:")
    print("  L = target loss")
    print("  D_j = covariance matrix of supplier percentage changes")
    print("  a_j = predicted percentage losses for each supplier")
    
    # Create simple 3-supplier example
    print("\n" + "-"*70)
    print("Example: Node with 3 suppliers")
    print("-"*70)
    
    # Baseline quantities
    print("\nSupplier baseline quantities:")
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C']
    baselines = np.array([10000, 15000, 8000])
    for supplier, qty in zip(suppliers, baselines):
        print(f"  {supplier}: {qty:,} kg")
    
    total_baseline = baselines.sum()
    print(f"  Total: {total_baseline:,} kg")
    
    # Target loss
    target_loss_pct = 0.20  # 20%
    target_loss_abs = target_loss_pct * total_baseline
    print(f"\nTarget loss: {target_loss_pct*100}% = {target_loss_abs:,.0f} kg")
    
    # Covariance matrix (synthetic)
    print("\nCovariance matrix D_j (percentage changes):")
    cov_matrix = np.array([
        [0.01, 0.003, 0.002],
        [0.003, 0.015, 0.004],
        [0.002, 0.004, 0.008]
    ])
    print(cov_matrix)
    
    # Apply RST equation
    print("\nApplying RST equation:")
    ones = np.ones(3)
    cov_sum = np.sum(cov_matrix)
    print(f"  sum(D_j) = {cov_sum:.6f}")
    
    a_j = (target_loss_abs / cov_sum) * (cov_matrix @ ones)
    print(f"\n  a_j (predicted % losses) = {a_j}")
    
    # Convert to absolute quantities
    predicted_losses = a_j * baselines
    print(f"\nPredicted absolute losses:")
    for supplier, loss in zip(suppliers, predicted_losses):
        pct = (loss / baselines[suppliers.index(supplier)]) * 100
        print(f"  {supplier}: {loss:,.2f} kg ({pct:.1f}%)")
    
    print(f"\nTotal predicted loss: {predicted_losses.sum():,.2f} kg")
    print(f"Target loss: {target_loss_abs:,.2f} kg")
    print(f"Match: {'✓' if abs(predicted_losses.sum() - target_loss_abs) < 1 else '✗'}")


def main():
    """Run complete demonstration"""
    print("\n" + "="*70)
    print("REVERSE STRESS TESTING FOR SUPPLY CHAIN RESILIENCE")
    print("Implementation of arXiv:2511.07289 (Smith et al., 2025)")
    print("="*70)
    
    # Demonstrate single-layer RST
    demonstrate_single_layer_rst()
    
    # Run basic example
    network, results = run_basic_example()
    
    # Generate visualizations
    create_visualizations(network, results)
    
    # Optional: Run data-driven example
    print("\n" + "="*70)
    print("Would you like to run the data-driven example? (uses synthetic Comtrade data)")
    print("This will take a few more minutes...")
    print("="*70)
    
    # For automated execution, we'll run it
    # In interactive mode, you could ask for user input
    print("\nRunning data-driven example...")
    network2, results2 = run_data_driven_example()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("  1. RST identifies most likely disruption sources without assuming")
    print("     specific threat scenarios (threat-agnostic approach)")
    print("  2. Different loss levels reveal different vulnerabilities:")
    print("     - Some countries are critical across all scenarios")
    print("     - Others only matter for specific disruption magnitudes")
    print("  3. Probabilistic analysis captures uncertainty in predictions")
    print("  4. Method enables proactive supply chain risk management")
    print("\nNext steps:")
    print("  - Review generated visualizations in ./rst_results/")
    print("  - Modify network structure or add real trade data")
    print("  - Adjust loss scenarios to match your use case")
    print("  - Integrate with existing risk management frameworks")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
