"""
Title embeddings generator using OpenAI's API.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import json
from dotenv import load_dotenv
import time


class TitleEmbeddingsGenerator:
    """Generates embeddings for product titles using OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the generator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        self.client = OpenAI(api_key=api_key)
        
        # Create output directory
        self.output_path = Path(config['output']['base_path'])
        self.output_path.mkdir(exist_ok=True)
        
        # Progress tracking file
        self.progress_file = self.output_path / "embeddings_progress.json"
        
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from previous runs."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed_batches': [],
            'successful_embeddings': {},
            'total_processed': 0,
            'failed_batches': []
        }
        
    def save_progress(self, progress: Dict[str, Any]):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)
            
    def generate_embeddings(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for product titles.
        
        Args:
            metadata_df: DataFrame containing product titles
            
        Returns:
            DataFrame with title embeddings
        """
        self.logger.info("Starting embeddings generation")
        self.logger.info(f"Generating embeddings for {len(metadata_df)} titles")
        
        # Load previous progress
        progress = self.load_progress()
        
        # Create batches
        batch_size = self.config['embeddings']['batch_size']
        titles = metadata_df['title'].tolist()
        parent_asins = metadata_df['parent_asin'].tolist()
        
        # Skip already processed titles
        if progress['successful_embeddings']:
            processed_asins = set(progress['successful_embeddings'].keys())
            titles_to_process = []
            asins_to_process = []
            for title, asin in zip(titles, parent_asins):
                if asin not in processed_asins:
                    titles_to_process.append(title)
                    asins_to_process.append(asin)
            titles = titles_to_process
            parent_asins = asins_to_process
            self.logger.info(f"Skipping {len(processed_asins)} already processed titles")
        
        batches = [
            (titles[i:i + batch_size], parent_asins[i:i + batch_size])
            for i in range(0, len(titles), batch_size)
        ]
        
        # Process batches
        embeddings_dict = progress['successful_embeddings'].copy()
        failed_batches = progress.get('failed_batches', [])
        
        for batch_idx, (title_batch, asin_batch) in enumerate(tqdm(batches, desc="Generating embeddings")):
            if str(batch_idx) in progress['completed_batches']:
                continue
                
            try:
                batch_embeddings = self._generate_batch_embeddings(title_batch, asin_batch)
                embeddings_dict.update(batch_embeddings)
                progress['completed_batches'].append(str(batch_idx))
                progress['successful_embeddings'] = embeddings_dict
                progress['total_processed'] = len(embeddings_dict)
                self.save_progress(progress)
                
                # Add delay between successful batches
                time.sleep(self.config['embeddings']['delay_between_batches'])
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Failed to process batch {batch_idx}: {error_msg}")
                
                # Check for rate limit or quota errors
                if "insufficient_quota" in error_msg or "rate_limit" in error_msg:
                    failed_batches.append(batch_idx)
                    progress['failed_batches'] = failed_batches
                    self.save_progress(progress)
                    
                    # Create partial DataFrame with successful embeddings
                    if embeddings_dict:
                        partial_df = pd.DataFrame([
                            {
                                'parent_asin': asin,
                                'title': metadata_df[metadata_df['parent_asin'] == asin]['title'].iloc[0],
                                'embedding': embedding
                            }
                            for asin, embedding in embeddings_dict.items()
                        ])
                        
                        # Save partial results
                        output_file = self.output_path / self.config['output']['embeddings_file']
                        partial_df.to_parquet(output_file, index=False)
                        self.logger.info(f"Saved partial embeddings to {output_file}")
                        
                        # Log statistics for partial completion
                        self.logger.info("Partial embedding generation statistics:")
                        self.logger.info(f"  - Total titles: {len(metadata_df)}")
                        self.logger.info(f"  - Successfully embedded: {len(embeddings_dict)}")
                        self.logger.info(f"  - Failed embeddings: {len(metadata_df) - len(embeddings_dict)}")
                        self.logger.info(f"  - Success rate: {(len(embeddings_dict) / len(metadata_df)) * 100:.2f}%")
                        self.logger.info(f"  - Failed batches: {failed_batches}")
                    
                    raise RuntimeError(
                        f"API quota exceeded. Successfully processed {len(embeddings_dict)} titles. "
                        f"Failed batches: {failed_batches}. Please check your OpenAI account quota "
                        "and billing details, then run again to continue from where we left off."
                    )
                else:
                    # For other errors, just add to failed batches and continue
                    failed_batches.append(batch_idx)
                    progress['failed_batches'] = failed_batches
                    self.save_progress(progress)
        
        # Create final DataFrame
        embeddings_df = pd.DataFrame([
            {
                'parent_asin': asin,
                'title': metadata_df[metadata_df['parent_asin'] == asin]['title'].iloc[0],
                'embedding': embedding
            }
            for asin, embedding in embeddings_dict.items()
        ])
        
        # Log statistics
        self.logger.info("Embedding generation statistics:")
        self.logger.info(f"  - Total titles: {len(metadata_df)}")
        self.logger.info(f"  - Successfully embedded: {len(embeddings_dict)}")
        self.logger.info(f"  - Failed embeddings: {len(metadata_df) - len(embeddings_dict)}")
        self.logger.info(f"  - Success rate: {(len(embeddings_dict) / len(metadata_df)) * 100:.2f}%")
        
        # Only raise error if there are actual failed batches
        if failed_batches and len(embeddings_dict) < len(metadata_df):
            failed_titles = sum(len(batches[idx][0]) for idx in failed_batches)
            error_msg = f"Failed to generate embeddings for {failed_titles} titles in batches: {failed_batches}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Save embeddings
        output_file = self.output_path / self.config['output']['embeddings_file']
        embeddings_df.to_parquet(output_file, index=False)
        self.logger.info(f"Saved embeddings to {output_file}")
        
        self.logger.info("Embeddings generation completed successfully")
        return embeddings_df
    
    def _generate_batch_embeddings(
        self, titles: List[str], asins: List[str]
    ) -> Dict[str, List[float]]:
        """Generate embeddings for a batch of titles."""
        max_retries = self.config['embeddings']['max_retries']
        delay = self.config['embeddings']['delay_between_batches']
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config['embeddings']['model'],
                    input=titles
                )
                
                # Create dictionary mapping ASINs to embeddings
                embeddings_dict = {
                    asin: data.embedding
                    for asin, data in zip(asins, response.data)
                }
                
                return embeddings_dict
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: Error code: {getattr(e, 'status_code', 'unknown')} - {error_msg}"
                )
                
                # Check for rate limit or quota errors
                if "insufficient_quota" in error_msg or "rate_limit" in error_msg:
                    raise  # Let the caller handle quota errors
                
                if attempt < max_retries - 1:
                    retry_delay = delay * (attempt + 1)
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
    
    def load_embeddings(self) -> pd.DataFrame:
        """Load existing embeddings from file."""
        embeddings_file = self.output_path / self.config['output']['embeddings_file']
        return pd.read_parquet(embeddings_file)
    
    def find_similar_titles(
        self, query: str, embeddings_df: pd.DataFrame, top_k: int = 5
    ) -> pd.DataFrame:
        """
        Find titles similar to a query using cosine similarity.
        
        Args:
            query: Query title to find similar titles for
            embeddings_df: DataFrame containing title embeddings
            top_k: Number of similar titles to return
            
        Returns:
            DataFrame with similar titles and similarity scores
        """
        # Generate embedding for query
        query_response = self.client.embeddings.create(
            model=self.config['embeddings']['model'],
            input=[query]
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate cosine similarity
        similarities = []
        for _, row in embeddings_df.iterrows():
            similarity = self._cosine_similarity(query_embedding, row['embedding'])
            similarities.append(similarity)
        
        # Add similarities to DataFrame
        results_df = embeddings_df.copy()
        results_df['similarity'] = similarities
        
        # Sort by similarity and return top k
        return results_df.sort_values('similarity', ascending=False).head(top_k)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def check_openai_api_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(os.getenv('OPENAI_API_KEY')) 