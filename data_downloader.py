"""
Amazon dataset downloader with configuration support and parquet output.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Set, Tuple
import pandas as pd
from datasets import load_dataset
import yaml


class AmazonDataDownloader:
    """Downloads and processes Amazon review datasets with metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the downloader with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.base_path = Path(config['output']['base_path'])
        self.dataset_path = Path(config['output']['base_path']) / config['dataset']['subset']
        
        # Create all required directories
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.test_path = self.dataset_path / "test"
        self.meta_path = self.dataset_path / "meta"
        self.embeddings_path = self.dataset_path / "embeddings"
        
        for path in [self.train_path, self.val_path, self.test_path, 
                    self.meta_path, self.embeddings_path]:
            path.mkdir(parents=True, exist_ok=True)
        
    def _remove_missing_titles(self, metadata_df: pd.DataFrame, dataset_splits: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Remove items with missing titles from metadata and all dataset splits.
        
        Args:
            metadata_df: DataFrame containing metadata
            dataset_splits: Dictionary of dataset splits
            
        Returns:
            Tuple of (cleaned_metadata_df, cleaned_dataset_splits)
        """
        # Find items with missing titles
        missing_titles_mask = (
            metadata_df['title'].isnull() | 
            (metadata_df['title'].astype(str).str.strip() == '')
        )
        missing_asins = metadata_df[missing_titles_mask]['parent_asin'].tolist()
        
        if missing_asins:
            self.logger.warning(f"Found {len(missing_asins)} items with missing or empty titles:")
            for asin in missing_asins:
                self.logger.warning(f"  - ASIN: {asin}")
            
            # Remove from metadata
            metadata_df = metadata_df[~missing_titles_mask].copy()
            
            # Remove from all splits
            cleaned_splits = {}
            for split_name, df in dataset_splits.items():
                cleaned_df = df[~df['parent_asin'].isin(missing_asins)].copy()
                removed_count = len(df) - len(cleaned_df)
                if removed_count > 0:
                    self.logger.warning(f"Removed {removed_count} interactions with missing titles from {split_name} split")
                cleaned_splits[split_name] = cleaned_df
            
            return metadata_df, cleaned_splits
        
        return metadata_df, dataset_splits
        
    def download_and_process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and process the Amazon dataset.
        
        Returns:
            Tuple of (combined_dataset, metadata_dataset)
        """
        self.logger.info("Starting Amazon dataset download and processing")
        
        # Step 1: Load main dataset
        dataset_splits = self._load_main_dataset()
        
        # Step 2: Collect parent ASINs
        parent_asins_needed = self._collect_parent_asins(dataset_splits)
        
        # Step 3: Load and filter metadata
        metadata_df = self._load_metadata(parent_asins_needed)
        
        # Step 4: Remove items with missing titles
        metadata_df, dataset_splits = self._remove_missing_titles(metadata_df, dataset_splits)
        
        # Step 5: Validate and log statistics
        self._validate_and_log_stats(dataset_splits, metadata_df, parent_asins_needed)
        
        # Step 6: Save datasets
        combined_df = self._save_datasets(dataset_splits, metadata_df)
        
        self.logger.info("Amazon dataset processing completed successfully")
        return combined_df, metadata_df
    
    def _load_main_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load the main Amazon dataset splits."""
        self.logger.info(f"Loading dataset: {self.config['dataset']['name']}")
        self.logger.info(f"Subset: {self.config['dataset']['subset']}")
        
        ds = load_dataset(
            self.config['dataset']['name'], 
            self.config['dataset']['subset']
        )
        
        dataset_splits = {}
        for split in self.config['dataset']['splits']:
            if split == 'valid':
                # Dataset uses 'valid' but we map it to 'val' for consistency
                dataset_splits['val'] = pd.DataFrame(ds['valid'])
                self.logger.info(f"Loaded {split} split: {len(dataset_splits['val'])} records")
            else:
                dataset_splits[split] = pd.DataFrame(ds[split])
                self.logger.info(f"Loaded {split} split: {len(dataset_splits[split])} records")
        
        return dataset_splits
    
    def _collect_parent_asins(self, dataset_splits: Dict[str, pd.DataFrame]) -> Set[str]:
        """Collect all unique parent ASINs from dataset splits."""
        self.logger.info("Collecting unique parent ASINs from all splits")
        
        all_parent_asins = []
        for split_name, df in dataset_splits.items():
            all_parent_asins.append(df['parent_asin'])
            self.logger.info(f"Found {df['parent_asin'].nunique()} unique parent ASINs in {split_name}")
        
        parent_asins_needed = set(pd.concat(all_parent_asins).unique())
        self.logger.info(f"Total unique parent ASINs needed: {len(parent_asins_needed)}")
        
        return parent_asins_needed
    
    def _load_metadata(self, parent_asins_needed: Set[str]) -> pd.DataFrame:
        """Load and filter metadata for the needed parent ASINs."""
        self.logger.info(f"Loading metadata: {self.config['dataset']['metadata_subset']}")
        
        # Load metadata
        meta = load_dataset(
            self.config['dataset']['name'],
            self.config['dataset']['metadata_subset'], 
            split=None
        )
        
        # Reduce columns
        columns_to_keep = self.config['metadata']['columns_to_keep']
        meta_reduced = meta['full'].remove_columns([
            col for col in meta['full'].column_names 
            if col not in columns_to_keep
        ])
        
        self.logger.info(f"Reduced metadata columns to: {columns_to_keep}")
        
        # Filter for needed parent ASINs
        self.logger.info("Filtering metadata for needed parent ASINs...")
        meta_filtered = meta_reduced.filter(
            lambda x: x['parent_asin'] in parent_asins_needed
        )
        
        # Convert to pandas
        df_meta = pd.DataFrame(meta_filtered)
        self.logger.info(f"Filtered metadata contains {len(df_meta)} records")
        
        return df_meta
    
    def _validate_and_log_stats(
        self, 
        dataset_splits: Dict[str, pd.DataFrame], 
        metadata_df: pd.DataFrame, 
        parent_asins_needed: Set[str]
    ) -> None:
        """Validate data and log comprehensive statistics."""
        self.logger.info("Validating data and logging statistics")
        
        # Dataset statistics
        total_records = sum(len(df) for df in dataset_splits.values())
        self.logger.info(f"Total dataset records: {total_records}")
        
        for split_name, df in dataset_splits.items():
            self.logger.info(f"{split_name.upper()} split statistics:")
            self.logger.info(f"  - Records: {len(df)}")
            self.logger.info(f"  - Unique users: {df['user_id'].nunique()}")
            self.logger.info(f"  - Unique items: {df['parent_asin'].nunique()}")
            # Convert rating to numeric, coercing errors to NaN
            ratings = pd.to_numeric(df['rating'], errors='coerce')
            avg_rating = ratings.mean()
            self.logger.info(f"  - Average rating: {avg_rating:.2f}")
        
        # Metadata validation
        missing_titles = (
            metadata_df['title'].isnull().sum() + 
            (metadata_df['title'].astype(str).str.strip() == '').sum()
        )
        
        self.logger.info(f"Metadata statistics:")
        self.logger.info(f"  - Total metadata records: {len(metadata_df)}")
        
        # We don't need to assert no missing titles anymore since we've removed them
        if missing_titles > 0:
            self.logger.warning(f"Found {missing_titles} remaining items with missing or empty titles after cleanup")
        else:
            self.logger.info("✓ All titles are present and non-empty")
        
        # Coverage validation
        covered_asins = set(metadata_df['parent_asin'])
        missing_ids = parent_asins_needed - covered_asins
        
        if self.config['validation']['log_missing_metadata']:
            self.logger.info(f"Parent ASINs from dataset not found in metadata: {len(missing_ids)}")
            if missing_ids:
                self.logger.warning(f"Missing ASINs: {list(missing_ids)[:10]}...")  # Log first 10
        
        coverage_percent = (len(covered_asins) / len(parent_asins_needed)) * 100
        self.logger.info(f"Metadata coverage: {coverage_percent:.2f}%")
    
    def _save_datasets(
        self, 
        dataset_splits: Dict[str, pd.DataFrame], 
        metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Save all datasets to parquet files."""
        self.logger.info("Saving datasets to parquet files")
        
        # Save individual splits
        split_paths = {
            'train': self.train_path,
            'val': self.val_path,
            'test': self.test_path
        }
        
        for split_name, df in dataset_splits.items():
            filepath = split_paths[split_name] / self.config['output'][f'{split_name}_file']
            df.to_parquet(filepath, index=False)
            self.logger.info(f"Saved {split_name} split to {filepath}")
        
        # Save metadata
        metadata_filepath = self.meta_path / self.config['output']['metadata_file']
        metadata_df.to_parquet(metadata_filepath, index=False)
        self.logger.info(f"Saved metadata to {metadata_filepath}")
        
        # Create combined dataset for convenience
        combined_df = pd.concat(dataset_splits.values(), ignore_index=True)
        combined_df['split'] = pd.concat([
            pd.Series([split_name] * len(df), name='split') 
            for split_name, df in dataset_splits.items()
        ], ignore_index=True)
        
        self.logger.info(f"Created combined dataset with {len(combined_df)} records")
        return combined_df


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    ) 