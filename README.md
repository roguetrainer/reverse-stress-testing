# Reverse Stress Testing for Supply Chain Resilience

Python implementation of the methodology from **"Reverse Stress Testing for Supply Chain Resilience"** (arXiv:2511.07289) by Smith et al., 2025.

## Overview

This implementation provides a complete framework for **reverse stress testing (RST)** of supply chains. Unlike traditional forward stress testing that starts with a threat scenario and measures its impact, RST works backwards from a specified disruption level to identify the most likely causes.

### Key Features

- **Threat-Agnostic Analysis**: Identifies vulnerabilities without assuming specific disruption scenarios
- **Multi-Layer Network Modeling**: Handles complex supply chains with multiple transformation stages
- **Probabilistic Predictions**: Uses Bayesian methods to quantify uncertainty
- **Backpropagation Algorithm**: Traces disruptions through entire supply chain tiers
- **Rich Visualizations**: Generates publication-quality figures similar to the paper

## Methodology

The implementation follows the 5-step process from the paper:

1. **Network Construction**: Build layered supply chain network from trade data
2. **Single-Layer RST**: Apply reverse stress testing equation to find most likely losses
3. **Backpropagation**: Propagate predicted losses upstream through supply chain
4. **Scenario Generation**: Create alternative scenarios along principal components
5. **Probabilistic Analysis**: Sample covariance matrices to capture uncertainty

### Core Equation

The most likely disruption scenario is computed using:

```
a_j = (L / sum(D_j)) * D_j * 1_vector
```

where:
- `L` = target loss at consumer node
- `D_j` = covariance matrix of historical supplier changes
- `a_j` = predicted percentage losses for each supplier

## Installation

```bash
# Clone repository
git clone <repository-url>
cd reverse-stress-testing

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- NetworkX >= 2.6.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## Quick Start

### Basic Example

```python
from reverse_stress_testing import create_example_copper_network, ReverseStressTester
import numpy as np

# Create example network
network = create_example_copper_network()

# Initialize RST
rst = ReverseStressTester(network, q=0.5)

# Add historical data (or use synthetic data)
# ... add data for each node ...

# Run RST for multiple loss scenarios
loss_levels = [0.05, 0.20, 0.50]  # 5%, 20%, 50%
results = rst.run_full_rst(loss_levels, num_samples=1000)

# Get vulnerability scores
for loss_level in loss_levels:
    vuln_scores = rst.get_vulnerability_scores(results[loss_level])
    print(f"\n{loss_level*100}% Loss Scenario:")
    print(vuln_scores.head(10))
```

### Complete Demonstration

Run the full demonstration:

```bash
python demo_rst.py
```

This will:
1. Create an example copper supply chain network
2. Run reverse stress testing for 5%, 20%, and 50% loss scenarios
3. Generate all visualizations
4. Display vulnerability analysis

Output will be saved to `./rst_results/`.

## Usage Examples

### 1. Building Network from Trade Data

```python
from rst_data_processing import ComtradeDataProcessor

# Initialize processor
processor = ComtradeDataProcessor()

# Load Comtrade data
df = processor.load_comtrade_data('trade_data.csv', 
                                   hs_codes=['HS2603', 'HS7402', 'HS7403', 'HS7408'])

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
df = processor.remove_transient_suppliers(df, min_periods=6)

# Build network
good_hierarchy = ['copper_ore', 'unrefined_copper', 'refined_copper', 'copper_wire']
network = processor.build_network_from_data(df, 'USA', good_hierarchy)

# Create time series for covariance matrices
time_series = processor.create_historical_time_series(df, network)
```

### 2. Running RST Analysis

```python
from reverse_stress_testing import ReverseStressTester

# Initialize
rst = ReverseStressTester(network, q=0.5)

# Add historical data
for node_id, ts_data in time_series.items():
    rst.add_historical_data(node_id, ts_data)

# Run analysis
results = rst.run_full_rst([0.05, 0.20, 0.50], num_samples=1000)

# Analyze results
for loss_level, scenarios in results.items():
    print(f"\n{loss_level*100}% Loss Scenario:")
    
    for layer, layer_scenarios in scenarios.items():
        most_likely = layer_scenarios[0]
        print(f"\nLayer {layer}:")
        for node_id, loss_qty in sorted(most_likely.loss_predictions.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
            country = network.nodes[node_id].country
            print(f"  {country}: {loss_qty:,.0f} kg")
```

### 3. Creating Visualizations

```python
from rst_visualization import (plot_network_topology, plot_scenario_pdfs, 
                               plot_vulnerability_heatmap, create_full_report)

# Plot network topology
fig, ax = plot_network_topology(network)
fig.savefig('network_topology.png', dpi=300, bbox_inches='tight')

# Plot PDFs for loss scenarios
fig, axes = plot_scenario_pdfs(results, network, loss_level=0.20)
fig.savefig('pdfs_20pct.png', dpi=300, bbox_inches='tight')

# Plot vulnerability heatmap
fig, ax = plot_vulnerability_heatmap(results, network)
fig.savefig('vulnerability_heatmap.png', dpi=300, bbox_inches='tight')

# Generate complete report
create_full_report(network, results, output_dir='./results')
```

## File Structure

```
.
├── reverse_stress_testing.py    # Core RST implementation
├── rst_data_processing.py       # Data loading and network construction
├── rst_visualization.py         # Visualization utilities
├── demo_rst.py                  # Complete demonstration script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Module Documentation

### `reverse_stress_testing.py`

Core implementation of the RST methodology.

**Classes:**
- `SupplyChainNode`: Represents a node (entity) in the supply chain
- `SupplyChainEdge`: Represents a transaction between entities
- `SupplyChainNetwork`: Network structure with layered organization
- `ReverseStressTester`: Main class for running RST analysis
- `RSTScenario`: Contains results for a single scenario

**Key Methods:**
- `ReverseStressTester.single_layer_rst()`: Single-layer reverse stress test
- `ReverseStressTester.backpropagate_losses()`: Backpropagate through all layers
- `ReverseStressTester.run_full_rst()`: Complete RST for multiple loss levels

### `rst_data_processing.py`

Utilities for loading and processing trade data.

**Classes:**
- `ComtradeDataProcessor`: Process UN Comtrade data

**Key Methods:**
- `load_comtrade_data()`: Load trade data from CSV
- `build_network_from_data()`: Construct network from trade records
- `create_historical_time_series()`: Extract time series for covariance matrices
- `handle_missing_values()`: Impute missing data
- `remove_transient_suppliers()`: Filter out unreliable suppliers

### `rst_visualization.py`

Visualization functions for RST results.

**Functions:**
- `plot_network_topology()`: Visualize supply chain network structure
- `plot_layer_quantities()`: Show quantities flowing through each layer
- `plot_scenario_pdfs()`: Plot probability distributions (Figure 4 from paper)
- `plot_vulnerability_heatmap()`: Heatmap of vulnerability scores
- `plot_comparison_table()`: Comparison table (Table 1 from paper)
- `create_full_report()`: Generate all visualizations

## Use Case: Copper Supply Chain

The paper demonstrates RST on the global copper supply chain, analyzing imports to the United States. The supply chain has 4 layers:

1. **Layer 0**: Copper ore (HS Code 2603)
2. **Layer 1**: Unrefined/blister copper (HS Code 7402)
3. **Layer 2**: Refined copper (HS Code 7403)
4. **Layer 3**: Copper wire (HS Code 7408)

### Key Findings from Paper

- **Canada, Chile, and Mexico** are critical across all loss scenarios
- **Some countries** (e.g., Papua New Guinea) matter for small disruptions but not catastrophic ones
- **Other countries** (e.g., Peru) only become critical at high disruption levels
- **Probabilistic approach** reveals uncertainty and alternative scenarios

## Extending the Implementation

### Adding New Supply Chains

1. Define your good type hierarchy
2. Map HS codes or product codes to good types
3. Load trade/transaction data
4. Build network using `ComtradeDataProcessor`
5. Run RST analysis

### Customizing Analysis

```python
# Adjust scenario distance parameter
rst = ReverseStressTester(network, q=0.3)  # Default is 0.5

# Change number of uncertainty samples
results = rst.run_full_rst(loss_levels, num_samples=5000)

# Filter transient suppliers more aggressively
df = processor.remove_transient_suppliers(df, min_periods=12)
```

### Adding Reserve Mechanism

The paper mentions a "reserve" system to handle mismatches between input and output weights. This is partially implemented in the `SupplyChainNode.reserve` attribute and can be extended for your use case.

## Comparison: Forward vs. Reverse Stress Testing

| Aspect | Forward Stress Testing | Reverse Stress Testing |
|--------|----------------------|----------------------|
| **Starting Point** | Threat scenario | Disruption outcome |
| **Question** | "What happens if X occurs?" | "What could cause Y loss?" |
| **Scenarios** | Limited by imagination | Discovers probable scenarios |
| **Residual Risk** | Cannot capture | Systematically identifies |
| **Approach** | Threat-specific | Threat-agnostic |

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{smith2025reverse,
  title={Reverse Stress Testing for Supply Chain Resilience},
  author={Smith, Madison and Gaiewski, Michael and Dulin, Sam and Williams, Laurel and Keisler, Jeffrey and Jin, Andrew and Linkov, Igor},
  journal={arXiv preprint arXiv:2511.07289},
  year={2025}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for methodology details and proper attribution.

## Contributing

Contributions are welcome! Areas for improvement:

- Enhanced covariance matrix estimation
- Integration with additional data sources
- Performance optimization for large networks
- Additional visualization options
- Real-time monitoring capabilities

## Contact

For questions about the implementation, please open an issue on GitHub.

For questions about the methodology, refer to the original paper or contact the authors.

## Acknowledgments

This implementation is based on the methodology developed by Smith et al. (2025) and published on arXiv. The code structure follows the 5-step process described in Section 2 of the paper.

Special thanks to the authors for their clear exposition of the methodology and the detailed copper supply chain use case that enabled this implementation.
