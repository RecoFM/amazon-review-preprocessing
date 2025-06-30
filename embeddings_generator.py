"""
Generate embeddings for product titles using OpenAI's text-embedding-3-large model.
"""

import logging
import os
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import time


def check_openai_api_key() -> None:
    """Check if OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )


class TitleEmbeddingsGenerator:
    """Generates embeddings for product titles using OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the generator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI()
        
        # Setup paths
        self.base_path = Path(config['output']['base_path'])
        self.dataset_path = self.base_path / config['dataset']['subset']
        self.embeddings_path = self.dataset_path / "embeddings"
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.embeddings_path / "embeddings_progress.json"
        self.completed_asins = self._load_progress()
        
    def _load_progress(self) -> set:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self) -> None:
        """Save progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.completed_asins), f)
    
    def _get_embeddings_batch(self, texts: List[str], retries: int = 0) -> Optional[List[List[float]]]:
        """Get embeddings for a batch of texts using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model=self.config['embeddings']['model'],
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            if retries < self.config['embeddings']['max_retries']:
                self.logger.warning(f"Retry {retries + 1} after error: {str(e)}")
                time.sleep(2 ** retries)  # Exponential backoff
                return self._get_embeddings_batch(texts, retries + 1)
            else:
                self.logger.error(f"Failed to get embeddings after {retries} retries: {str(e)}")
                return None
    
    def generate_embeddings(self, metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate embeddings for product titles.
        
        Args:
            metadata_df: Optional metadata DataFrame. If not provided, will load from file.
            
        Returns:
            DataFrame with embeddings
        """
        self.logger.info("Starting embeddings generation")
        
        # Load metadata if not provided
        if metadata_df is None:
            metadata_path = self.dataset_path / "meta" / self.config['output']['metadata_file']
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            metadata_df = pd.read_parquet(metadata_path)
        
        # Filter for titles that need embeddings
        titles_to_process = metadata_df[
            ~metadata_df['parent_asin'].isin(self.completed_asins)
        ]
        
        if len(titles_to_process) == 0:
            self.logger.info("No new titles to process")
            embeddings_path = self.embeddings_path / self.config['output']['embeddings_file']
            if embeddings_path.exists():
                return pd.read_parquet(embeddings_path)
            else:
                return pd.DataFrame(columns=['parent_asin', 'title', 'embedding'])
        
        self.logger.info(f"Generating embeddings for {len(titles_to_process)} titles")
        
        # Process in batches
        batch_size = self.config['embeddings']['batch_size']
        delay = self.config['embeddings']['delay_between_batches']
        
        all_embeddings = []
        all_asins = []
        all_titles = []
        
        for i in tqdm(range(0, len(titles_to_process), batch_size)):
            batch = titles_to_process.iloc[i:i + batch_size]
            batch_titles = batch['title'].tolist()
            batch_embeddings = self._get_embeddings_batch(batch_titles)
            
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
                all_asins.extend(batch['parent_asin'].tolist())
                all_titles.extend(batch_titles)
                self.completed_asins.update(batch['parent_asin'].tolist())
                
                # Save progress after each batch
                self._save_progress()
            
            if i + batch_size < len(titles_to_process):
                time.sleep(delay)
        
        # Create embeddings DataFrame
        embeddings_df = pd.DataFrame({
            'parent_asin': all_asins,
            'title': all_titles,
            'embedding': all_embeddings
        })
        
        # Load existing embeddings if any
        embeddings_path = self.embeddings_path / self.config['output']['embeddings_file']
        if embeddings_path.exists():
            existing_df = pd.read_parquet(embeddings_path)
            embeddings_df = pd.concat([existing_df, embeddings_df], ignore_index=True)
        
        # Save embeddings
        embeddings_df.to_parquet(embeddings_path, index=False)
        self.logger.info(f"Saved {len(embeddings_df)} embeddings to {embeddings_path}")
        
        return embeddings_df
    
    def find_similar_titles(
        self, 
        query: str, 
        embeddings_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find similar titles using embeddings.
        
        Args:
            query: Query text to find similar titles for
            embeddings_df: Optional embeddings DataFrame. If not provided, will load from file.
            metadata_df: Optional metadata DataFrame. If not provided, will load from file.
            top_k: Number of similar titles to return
            
        Returns:
            DataFrame with similar titles and their scores
        """
        # Get query embedding
        query_embedding = self._get_embeddings_batch([query])
        if not query_embedding:
            raise ValueError("Failed to get embedding for query")
        query_embedding = query_embedding[0]
        
        # Load embeddings if not provided
        if embeddings_df is None:
            embeddings_path = self.embeddings_path / self.config['output']['embeddings_file']
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
            embeddings_df = pd.read_parquet(embeddings_path)
        
        # Load metadata if not provided
        if metadata_df is None:
            metadata_path = self.dataset_path / "meta" / self.config['output']['metadata_file']
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            metadata_df = pd.read_parquet(metadata_path)
        
        # Convert embeddings to numpy array for fast computation
        embeddings = np.array(embeddings_df['embedding'].tolist())
        query_embedding = np.array(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k similar titles
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get titles and scores
        results = []
        for idx in top_indices:
            results.append({
                'parent_asin': embeddings_df.iloc[idx]['parent_asin'],
                'title': embeddings_df.iloc[idx]['title'],
                'similarity_score': similarities[idx]
            })
        
        return pd.DataFrame(results) 