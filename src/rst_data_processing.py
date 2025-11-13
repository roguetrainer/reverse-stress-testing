"""
Data processing utilities for supply chain data.

Provides functions to:
- Load and process UN Comtrade data
- Build supply chain networks from trade data
- Handle data quality issues (missing values, outliers, transient suppliers)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from reverse_stress_testing import SupplyChainNetwork, SupplyChainNode, SupplyChainEdge


class ComtradeDataProcessor:
    """
    Process UN Comtrade data for reverse stress testing.
    
    Following the data processing approach described in Section 2.2 of the paper.
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.good_types = {}  # HS Code -> good type mapping
        
    def load_comtrade_data(self, filepath: str, hs_codes: List[str]) -> pd.DataFrame:
        """
        Load UN Comtrade data from CSV file.
        
        Expected columns:
        - reporter: Reporting country
        - partner: Partner country
        - trade_flow: Import or Export
        - hs_code: HS commodity code
        - year: Year of trade
        - month: Month of trade (optional)
        - trade_value_usd: Trade value in USD
        - netweight_kg: Net weight in kg (optional)
        - quantity: Quantity in appropriate units
        
        Args:
            filepath: Path to Comtrade CSV file
            hs_codes: List of HS codes to filter for
            
        Returns:
            Filtered and processed DataFrame
        """
        print(f"Loading Comtrade data from {filepath}...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Filter for relevant HS codes
        df = df[df['hs_code'].isin(hs_codes)]
        
        # Filter for time period (2018-2024 as in paper)
        if 'year' in df.columns:
            df = df[(df['year'] >= 2018) & (df['year'] <= 2024)]
        
        self.raw_data = df
        print(f"Loaded {len(df)} records")
        
        return df
    
    def set_good_type_mapping(self, hs_code_mapping: Dict[str, str]):
        """
        Set mapping from HS codes to good types.
        
        Example for copper (from paper Section 2.2):
        {
            'HS2603': 'copper_ore',
            'HS7402': 'unrefined_copper',
            'HS7403': 'refined_copper',
            'HS7408': 'copper_wire'
        }
        
        Args:
            hs_code_mapping: Dictionary mapping HS codes to good type names
        """
        self.good_types = hs_code_mapping
    
    def compute_quantity_from_value(self, df: pd.DataFrame, 
                                   commodity_prices: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compute quantity in kg from trade value when weight data is missing.
        
        Uses spot prices as described in the paper (Section 2.2).
        
        Args:
            df: DataFrame with trade data
            commodity_prices: Dict of {hs_code: {year_month: price_per_kg}}
            
        Returns:
            DataFrame with computed quantities
        """
        df = df.copy()
        
        # Create year-month column if needed
        if 'year_month' not in df.columns and 'year' in df.columns:
            df['year_month'] = df['year'].astype(str)
            if 'month' in df.columns:
                df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        
        # Compute quantity where missing
        for hs_code in df['hs_code'].unique():
            if hs_code not in commodity_prices:
                continue
            
            mask = (df['hs_code'] == hs_code) & (df['quantity_kg'].isna())
            
            for year_month, price in commodity_prices[hs_code].items():
                mask_ym = mask & (df['year_month'] == year_month)
                if mask_ym.any():
                    df.loc[mask_ym, 'quantity_kg'] = df.loc[mask_ym, 'trade_value_usd'] / price
        
        return df
    
    def aggregate_monthly_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trade flows by reporter-partner-good type-month.
        
        Args:
            df: Raw trade data
            
        Returns:
            Aggregated DataFrame
        """
        # Add good type
        df = df.copy()
        df['good_type'] = df['hs_code'].map(self.good_types)
        
        # Aggregate
        agg_cols = {
            'quantity_kg': 'sum',
            'trade_value_usd': 'sum'
        }
        
        groupby_cols = ['reporter', 'partner', 'good_type', 'trade_flow', 'year_month']
        result = df.groupby(groupby_cols).agg(agg_cols).reset_index()
        
        return result
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in time series data.
        
        Addresses data quality issues mentioned in the paper.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'interpolate', 'forward_fill', or 'zero'
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if strategy == 'interpolate':
            df['quantity_kg'] = df.groupby(['reporter', 'partner', 'good_type'])['quantity_kg'].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        elif strategy == 'forward_fill':
            df['quantity_kg'] = df.groupby(['reporter', 'partner', 'good_type'])['quantity_kg'].ffill()
        elif strategy == 'zero':
            df['quantity_kg'] = df['quantity_kg'].fillna(0)
        
        return df
    
    def remove_transient_suppliers(self, df: pd.DataFrame, 
                                  min_periods: int = 6) -> pd.DataFrame:
        """
        Remove highly transient suppliers who appear infrequently.
        
        As mentioned in Section 2.1.2 of the paper, transient suppliers
        can affect covariance matrix quality.
        
        Args:
            df: Trade data
            min_periods: Minimum number of time periods a supplier must appear
            
        Returns:
            Filtered DataFrame
        """
        # Count appearances per supplier-customer pair
        counts = df.groupby(['reporter', 'partner', 'good_type']).size().reset_index(name='n_periods')
        
        # Keep only those with sufficient history
        valid = counts[counts['n_periods'] >= min_periods]
        
        # Merge back
        result = df.merge(valid[['reporter', 'partner', 'good_type']], 
                         on=['reporter', 'partner', 'good_type'],
                         how='inner')
        
        print(f"Removed {len(df) - len(result)} records from transient suppliers")
        return result
    
    def build_network_from_data(self, df: pd.DataFrame, 
                               end_country: str,
                               good_hierarchy: List[str]) -> SupplyChainNetwork:
        """
        Build supply chain network from trade data.
        
        Implements network construction from Section 2.1.1:
        - Start at end node (final product in end_country)
        - Add suppliers layer by layer, working backwards
        - Only include entities that export to downstream layers
        
        Args:
            df: Processed trade data
            end_country: Final consumer country (e.g., 'USA')
            good_hierarchy: Ordered list of goods from raw to final
                           (e.g., ['copper_ore', 'unrefined_copper', 'refined_copper', 'copper_wire'])
            
        Returns:
            SupplyChainNetwork object
        """
        # Create network with end node
        final_good = good_hierarchy[-1]
        end_node_id = f"{end_country}_{final_good}"
        network = SupplyChainNetwork(end_node_id=end_node_id)
        
        # Calculate baseline quantities (average over time period)
        baseline_quantities = df.groupby(['reporter', 'good_type'])['quantity_kg'].mean().to_dict()
        
        # Build network layer by layer, starting from end
        layer_nodes = {len(good_hierarchy) - 1: [end_node_id]}
        
        # Add end node
        end_baseline = df[
            (df['partner'] == end_country) & 
            (df['good_type'] == final_good) &
            (df['trade_flow'] == 'Import')
        ]['quantity_kg'].sum()
        
        network.add_node(SupplyChainNode(
            node_id=end_node_id,
            layer=len(good_hierarchy) - 1,
            good_type=final_good,
            country=end_country,
            baseline_quantity=end_baseline
        ))
        
        # Work backwards through good hierarchy
        for layer in range(len(good_hierarchy) - 1, 0, -1):
            current_good = good_hierarchy[layer]
            previous_good = good_hierarchy[layer - 1]
            
            # Find all suppliers of current layer
            current_layer_countries = set()
            for node_id in layer_nodes.get(layer, []):
                country = node_id.split('_')[0]
                current_layer_countries.add(country)
            
            # Find exporters of current good to current layer countries
            suppliers = df[
                (df['partner'].isin(current_layer_countries)) &
                (df['good_type'] == current_good) &
                (df['trade_flow'] == 'Export')
            ]['reporter'].unique()
            
            layer_nodes[layer - 1] = []
            
            # Add supplier nodes and edges
            for supplier in suppliers:
                node_id = f"{supplier}_{previous_good}"
                
                if node_id not in network.nodes:
                    baseline = baseline_quantities.get((supplier, previous_good), 0)
                    
                    network.add_node(SupplyChainNode(
                        node_id=node_id,
                        layer=layer - 1,
                        good_type=previous_good,
                        country=supplier,
                        baseline_quantity=baseline
                    ))
                    
                    layer_nodes[layer - 1].append(node_id)
                
                # Add edges to customers in next layer
                for customer_country in current_layer_countries:
                    customer_node_id = f"{customer_country}_{current_good}"
                    
                    if customer_node_id in network.nodes:
                        # Calculate flow quantity
                        flow = df[
                            (df['reporter'] == supplier) &
                            (df['partner'] == customer_country) &
                            (df['good_type'] == current_good) &
                            (df['trade_flow'] == 'Export')
                        ]['quantity_kg'].sum()
                        
                        if flow > 0:
                            network.add_edge(SupplyChainEdge(
                                source=node_id,
                                target=customer_node_id,
                                quantity_flow=flow
                            ))
        
        print(f"Built network with {len(network.nodes)} nodes and {len(network.edges)} edges")
        return network
    
    def create_historical_time_series(self, df: pd.DataFrame, 
                                     network: SupplyChainNetwork) -> Dict[str, pd.DataFrame]:
        """
        Create historical time series for each node's suppliers.
        
        This data is used to compute covariance matrices.
        
        Args:
            df: Trade data with time series
            network: Supply chain network
            
        Returns:
            Dictionary mapping node_id -> DataFrame of supplier quantities over time
        """
        time_series_data = {}
        
        for node_id in network.nodes:
            suppliers = network.get_suppliers(node_id)
            if not suppliers:
                continue
            
            # Get country and good type for this node
            country = network.nodes[node_id].country
            good_type = network.nodes[node_id].good_type
            
            # Extract time series for each supplier
            supplier_data = {}
            for supplier_id in suppliers:
                supplier_country = network.nodes[supplier_id].country
                supplier_good = network.nodes[supplier_id].good_type
                
                # Filter data
                supplier_ts = df[
                    (df['reporter'] == supplier_country) &
                    (df['partner'] == country) &
                    (df['good_type'] == good_type) &
                    (df['trade_flow'] == 'Export')
                ][['year_month', 'quantity_kg']].copy()
                
                if not supplier_ts.empty:
                    supplier_ts = supplier_ts.set_index('year_month')['quantity_kg']
                    supplier_data[supplier_id] = supplier_ts
            
            if supplier_data:
                time_series_data[node_id] = pd.DataFrame(supplier_data)
        
        return time_series_data


def create_synthetic_comtrade_data(n_months: int = 24, 
                                  hs_codes: List[str] = ['HS2603', 'HS7402', 'HS7403', 'HS7408'],
                                  countries: List[str] = ['USA', 'CAN', 'MEX', 'CHL', 'PER', 'BRA']) -> pd.DataFrame:
    """
    Create synthetic Comtrade-like data for testing.
    
    Args:
        n_months: Number of months to generate
        hs_codes: HS codes to include
        countries: Countries to include
        
    Returns:
        DataFrame with synthetic trade data
    """
    np.random.seed(42)
    
    records = []
    
    # Good type hierarchy
    good_hierarchy = {
        'HS2603': ('copper_ore', 0),
        'HS7402': ('unrefined_copper', 1),
        'HS7403': ('refined_copper', 2),
        'HS7408': ('copper_wire', 3)
    }
    
    # Generate trade flows
    for month in range(1, n_months + 1):
        year = 2023 + (month // 12)
        m = (month % 12) + 1
        year_month = f"{year}-{str(m).zfill(2)}"
        
        for hs_code in hs_codes:
            good_type, layer = good_hierarchy[hs_code]
            
            # Generate exports from each country to others
            for exporter in countries:
                for importer in countries:
                    if exporter == importer:
                        continue
                    
                    # Random quantity with some structure
                    base_qty = np.random.uniform(1000, 50000)
                    seasonal = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
                    noise = np.random.normal(1, 0.1)
                    quantity = base_qty * seasonal * noise
                    
                    # Trade value (quantity * random price)
                    price = np.random.uniform(5, 15)
                    value = quantity * price
                    
                    records.append({
                        'reporter': exporter,
                        'partner': importer,
                        'trade_flow': 'Export',
                        'hs_code': hs_code,
                        'year': year,
                        'month': m,
                        'year_month': year_month,
                        'quantity_kg': quantity,
                        'trade_value_usd': value
                    })
    
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("Data Processing Utilities for Reverse Stress Testing")
    print("=" * 60)
    
    # Create synthetic data
    print("\nGenerating synthetic Comtrade data...")
    df = create_synthetic_comtrade_data()
    print(f"Generated {len(df)} trade records")
    
    # Initialize processor
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
    print("\nProcessing data...")
    df = processor.aggregate_monthly_flows(df)
    df = processor.handle_missing_values(df)
    df = processor.remove_transient_suppliers(df, min_periods=3)
    
    # Build network
    print("\nBuilding supply chain network...")
    good_hierarchy = ['copper_ore', 'unrefined_copper', 'refined_copper', 'copper_wire']
    network = processor.build_network_from_data(df, 'USA', good_hierarchy)
    
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Edges: {len(network.edges)}")
    print(f"  Layers: {list(network.layers.keys())}")
    
    # Create time series
    print("\nCreating historical time series...")
    time_series = processor.create_historical_time_series(df, network)
    print(f"Created time series for {len(time_series)} nodes")
    
    print("\nData processing complete!")
