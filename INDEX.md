# Index: Reverse Stress Testing Implementation

## 📚 Start Here

| File | Purpose | Read Time |
|------|---------|-----------|
| **QUICKSTART.md** | Get running in 5 minutes | 5 min |
| **README.md** | Full documentation and examples | 15 min |
| **IMPLEMENTATION_SUMMARY.md** | Technical details and validation | 10 min |

## 💻 Code Files

### Core Implementation
| File | Lines | Description |
|------|-------|-------------|
| **reverse_stress_testing.py** | 780 | Main RST algorithm implementation |
| **rst_data_processing.py** | 450 | Data loading and network building |
| **rst_visualization.py** | 380 | Visualization and plotting functions |
| **demo_rst.py** | 360 | Complete working demonstration |

### Configuration
| File | Description |
|------|-------------|
| **requirements.txt** | Python package dependencies |

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete demonstration
python demo_rst.py

# Run quick test
python reverse_stress_testing.py
```

## 📖 Learning Path

### Beginner
1. Read **QUICKSTART.md**
2. Run `python demo_rst.py`
3. Look at generated visualizations in `./rst_results/`
4. Read the basic example in **README.md**

### Intermediate
1. Read full **README.md**
2. Examine `reverse_stress_testing.py` - start with `ReverseStressTester` class
3. Try modifying the example network in `create_example_copper_network()`
4. Experiment with different loss scenarios

### Advanced
1. Read **IMPLEMENTATION_SUMMARY.md** for technical details
2. Study the backpropagation algorithm
3. Implement your own supply chain network
4. Load real Comtrade data using `ComtradeDataProcessor`
5. Extend the methodology with custom features

## 🔍 Code Structure

```
reverse_stress_testing.py
├── SupplyChainNode          # Data: Node in network
├── SupplyChainEdge          # Data: Edge/transaction
├── SupplyChainNetwork       # Structure: Layered network
├── RSTScenario              # Data: Scenario result
└── ReverseStressTester      # Algorithm: Main RST class
    ├── compute_covariance_matrix()
    ├── single_layer_rst()         # Equation 1 from paper
    ├── backpropagate_losses()     # Steps 2-3 from paper
    └── run_full_rst()             # Complete workflow

rst_data_processing.py
├── ComtradeDataProcessor    # Load and process trade data
│   ├── load_comtrade_data()
│   ├── build_network_from_data()
│   └── create_historical_time_series()
└── create_synthetic_comtrade_data()  # Generate test data

rst_visualization.py
├── plot_network_topology()      # Figure 3 (right)
├── plot_layer_quantities()      # Figure 3 (left)
├── plot_scenario_pdfs()         # Figure 4
├── plot_vulnerability_heatmap()
└── create_full_report()         # Generate all visualizations

demo_rst.py
├── demonstrate_single_layer_rst()   # Show core equation
├── run_basic_example()              # Pre-built network
├── run_data_driven_example()        # Build from data
└── create_visualizations()          # Generate plots
```

## 📊 Paper Mapping

| Paper Section | Implementation |
|---------------|----------------|
| 2.1.1 Network Construction | `SupplyChainNetwork`, `build_network_from_data()` |
| 2.1.2 Single Layer RST | `single_layer_rst()`, Equation 1 |
| 2.1.3 Backpropagation | `backpropagate_losses()` |
| 2.1.4 Probability Distributions | `_generate_probabilistic_scenarios()` |
| 2.2 Copper Use Case | `create_example_copper_network()` |
| Figure 3 | `plot_network_topology()`, `plot_layer_quantities()` |
| Figure 4 | `plot_scenario_pdfs()` |
| Table 1 | `plot_comparison_table()` |

## 🎯 Key Concepts

### Reverse Stress Testing
- **Forward**: Scenario → Impact
- **Reverse**: Impact → Scenarios
- **Implemented in**: `single_layer_rst()`, `backpropagate_losses()`

### Core Equation
```
a_j = (L / sum(D_j)) * D_j * 1_vector
```
- **Location**: `single_layer_rst()` method, line ~200
- **Inputs**: Loss L, covariance matrix D_j
- **Output**: Predicted losses a_j for each supplier

### Backpropagation
- Traces disruptions through supply chain tiers
- **Location**: `backpropagate_losses()` method
- **Process**: Layer M → Layer M-1 → ... → Layer 0

### Uncertainty Quantification
- Inverse Wishart sampling
- Monte Carlo simulation
- **Location**: `_generate_probabilistic_scenarios()` method

## 🧪 Testing

```python
# Quick functionality test
python -c "from reverse_stress_testing import create_example_copper_network; 
           n = create_example_copper_network(); 
           print(f'✓ Network: {len(n.nodes)} nodes')"

# Full demonstration
python demo_rst.py
```

## 📈 Example Outputs

Running `demo_rst.py` generates:

**Console Output:**
- Network statistics
- RST progress
- Vulnerability rankings
- Top contributors by layer

**Files in `./rst_results/`:**
- `network_topology.png` - Supply chain structure
- `layer_quantities.png` - Flow analysis
- `pdfs_5pct.png` - 5% loss scenario distributions
- `pdfs_20pct.png` - 20% loss scenario distributions
- `pdfs_50pct.png` - 50% loss scenario distributions
- `vulnerability_heatmap.png` - Vulnerability analysis
- `comparison_table.png` - Side-by-side comparison

## 🔧 Customization Points

### Network Structure
```python
# Modify in reverse_stress_testing.py
def create_example_copper_network():
    # Add/remove nodes
    # Change connections
    # Adjust quantities
```

### RST Parameters
```python
rst = ReverseStressTester(network, 
    q=0.5  # Scenario distance parameter (0.1-1.0)
)

results = rst.run_full_rst(
    [0.05, 0.20, 0.50],  # Loss levels
    num_samples=1000      # Uncertainty samples
)
```

### Data Processing
```python
processor.remove_transient_suppliers(
    min_periods=6  # Minimum data points required
)
```

## 🐛 Troubleshooting

| Issue | Solution | Location |
|-------|----------|----------|
| Import errors | Install requirements | `requirements.txt` |
| No historical data | Add data with `add_historical_data()` | `demo_rst.py` line 40 |
| Singular covariance | Increase min_periods | `rst_data_processing.py` line 180 |
| Negative losses | Check baseline quantities | Network construction |

## 📞 Getting Help

1. Check **QUICKSTART.md** for common tasks
2. Read **README.md** troubleshooting section
3. Review **IMPLEMENTATION_SUMMARY.md** for technical details
4. Examine code comments in source files
5. Run `demo_rst.py` to see working example

## 🎓 Citation

If using this implementation:

```bibtex
@article{smith2025reverse,
  title={Reverse Stress Testing for Supply Chain Resilience},
  author={Smith, Madison and Gaiewski, Michael and Dulin, Sam and 
          Williams, Laurel and Keisler, Jeffrey and Jin, Andrew and Linkov, Igor},
  journal={arXiv preprint arXiv:2511.07289},
  year={2025}
}
```

## 📝 Version Info

- **Implementation Date**: November 2025
- **Paper**: arXiv:2511.07289
- **Python**: 3.8+
- **Total Code**: ~2,000 lines
- **Files**: 8 (4 code, 4 documentation)

---

**Quick Links:**
- Paper: https://arxiv.org/abs/2511.07289
- Start: QUICKSTART.md
- Docs: README.md
- Code: reverse_stress_testing.py
