# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python demo_rst.py
```

This will:
- Create an example copper supply chain network
- Run reverse stress testing for 5%, 20%, and 50% loss scenarios  
- Generate visualizations in `./rst_results/`
- Display vulnerability analysis results

### 3. View Results

Check the `./rst_results/` directory for:
- Network topology visualization
- Probability density functions for each scenario
- Vulnerability heatmaps
- Comparison tables

## Quick Code Example

```python
from reverse_stress_testing import create_example_copper_network, ReverseStressTester
import numpy as np
import pandas as pd

# 1. Create network
network = create_example_copper_network()

# 2. Initialize RST
rst = ReverseStressTester(network, q=0.5)

# 3. Add historical data (synthetic example)
np.random.seed(42)
for node_id in network.nodes:
    suppliers = network.get_suppliers(node_id)
    if suppliers:
        # Create 24 months of data
        data = {}
        for supplier in suppliers:
            edge = network.graph[supplier][node_id]
            baseline = edge['quantity_flow']
            quantities = baseline * (1 + np.random.randn(24) * 0.1)
            data[supplier] = quantities
        rst.add_historical_data(node_id, pd.DataFrame(data))

# 4. Run RST
results = rst.run_full_rst([0.05, 0.20, 0.50], num_samples=100)

# 5. Get vulnerability scores
for loss_level in [0.05, 0.20, 0.50]:
    print(f"\n{loss_level*100}% Loss Scenario:")
    vuln = rst.get_vulnerability_scores(results[loss_level])
    print(vuln.head(5))
```

## Key Concepts

### What is Reverse Stress Testing?

**Traditional (Forward) Stress Testing:**
- Start with: "What if there's an earthquake in Chile?"
- Find: Impact on copper wire imports

**Reverse Stress Testing:**
- Start with: "What if US copper wire imports drop 20%?"
- Find: Most likely causes across the supply chain

### Why RST?

1. **Threat-Agnostic**: Don't need to imagine every possible disruption
2. **Probabilistic**: Quantifies uncertainty in predictions
3. **Comprehensive**: Identifies vulnerabilities across entire supply chain
4. **Actionable**: Shows where to focus risk mitigation efforts

## Understanding the Output

### Vulnerability Scores

Higher scores = greater contribution to potential disruptions

```
Country         Good Type            Vulnerability Score
Canada          refined_copper       42,000
Chile           unrefined_copper     36,000
Zambia          unrefined_copper     30,000
```

### Probability Distributions

The PDFs show the range of possible losses for each country, accounting for:
- Uncertainty in historical relationships
- Variability in supplier behavior
- Alternative scenarios

### Layer Analysis

Each layer represents a transformation stage:
- **Layer 0**: Raw materials (copper ore)
- **Layer 1**: Intermediate processing (unrefined copper)
- **Layer 2**: Final processing (refined copper)
- **Layer 3**: End product (copper wire)

RST identifies critical points at each stage.

## Next Steps

1. **Customize Network**: Modify `create_example_copper_network()` for your supply chain
2. **Add Real Data**: Use `ComtradeDataProcessor` to load actual trade data
3. **Adjust Parameters**: 
   - `q` parameter controls scenario diversity
   - `num_samples` affects probabilistic accuracy
   - `min_periods` filters unreliable suppliers
4. **Export Results**: All visualizations are saved as high-res PNG files

## Troubleshooting

### "No historical data"
Make sure to call `rst.add_historical_data()` for nodes with suppliers.

### "Covariance matrix not positive definite"
This is handled automatically. If you see warnings, try:
- Increasing `min_periods` when filtering transient suppliers
- Adding more historical data points
- Checking for data quality issues

### "Results don't match target loss"
This is expected! RST finds the most likely scenario, which may involve:
- Non-linear relationships
- Constraints (can't have negative production)
- Uncertainty in the covariance matrix

## Support

- **README.md**: Full documentation
- **demo_rst.py**: Complete working example
- **Paper**: arXiv:2511.07289 for methodology details

## Files Overview

| File | Purpose |
|------|---------|
| `reverse_stress_testing.py` | Core RST implementation |
| `rst_data_processing.py` | Data loading and network building |
| `rst_visualization.py` | Plotting and visualization |
| `demo_rst.py` | Complete demonstration |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `QUICKSTART.md` | This file |
