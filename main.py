"""
Main orchestrator for Amazon dataset preprocessing with embeddings generation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import typer
from typing_extensions import Annotated
from dotenv import load_dotenv

from data_downloader import AmazonDataDownloader, load_config, setup_logging
from embeddings_generator import TitleEmbeddingsGenerator, check_openai_api_key

app = typer.Typer(help="Amazon dataset preprocessing with embeddings generation")

# Load environment variables at startup
load_dotenv()


def use_yaml_config(param_name: str = "config"):
    """Decorator to load YAML configuration."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Load config if config parameter is provided
            if param_name in kwargs and kwargs[param_name]:
                config = load_config(kwargs[param_name])
                kwargs['config'] = config
            return func(*args, **kwargs)
        return wrapper
    return decorator


@app.command(name="download")
def download_data(
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Download and preprocess Amazon dataset with metadata.
    
    Args:
        config: Path to YAML configuration file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        logger.info("=" * 60)
        logger.info("STARTING AMAZON DATASET DOWNLOAD")
        logger.info("=" * 60)
        
        # Initialize downloader
        downloader = AmazonDataDownloader(config_dict)
        
        # Download and process data
        combined_df, metadata_df = downloader.download_and_process()
        
        logger.info("=" * 60)
        logger.info("DATASET DOWNLOAD COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        typer.echo(f"✅ Successfully downloaded and processed Amazon dataset")
        typer.echo(f"📊 Total records: {len(combined_df)}")
        typer.echo(f"🏷️  Metadata records: {len(metadata_df)}")
        typer.echo(f"💾 Data saved to: {config_dict['output']['base_path']}")
        
    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="embeddings")
def generate_embeddings(
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml",
    skip_download: Annotated[bool, typer.Option(help="Skip data download if files already exist")] = False
) -> None:
    """
    Generate embeddings for product titles using OpenAI's text-embedding-large model.
    
    Args:
        config: Path to YAML configuration file
        skip_download: Skip data download if files already exist
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        logger.info("=" * 60)
        logger.info("STARTING EMBEDDINGS GENERATION")
        logger.info("=" * 60)
        
        # Check OpenAI API key
        if not check_openai_api_key():
            error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            logger.error(error_msg)
            typer.echo(f"❌ {error_msg}", err=True)
            raise typer.Exit(1)
        
        # Check if metadata file exists or download data
        metadata_file = Path(config_dict['output']['base_path']) / config_dict['output']['metadata_file']
        
        if not metadata_file.exists() and not skip_download:
            logger.info("Metadata file not found. Downloading data first...")
            downloader = AmazonDataDownloader(config_dict)
            _, metadata_df = downloader.download_and_process()
        elif metadata_file.exists():
            logger.info(f"Loading existing metadata from {metadata_file}")
            import pandas as pd
            metadata_df = pd.read_parquet(metadata_file)
        else:
            error_msg = f"Metadata file not found at {metadata_file} and skip_download=True"
            logger.error(error_msg)
            typer.echo(f"❌ {error_msg}", err=True)
            raise typer.Exit(1)
        
        # Initialize embeddings generator
        embeddings_generator = TitleEmbeddingsGenerator(config_dict)
        
        # Generate embeddings
        embeddings_df = embeddings_generator.generate_embeddings(metadata_df)
        
        logger.info("=" * 60)
        logger.info("EMBEDDINGS GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        typer.echo(f"✅ Successfully generated embeddings for product titles")
        typer.echo(f"🧮 Embeddings generated: {len(embeddings_df)}")
        typer.echo(f"📐 Embedding dimension: {len(embeddings_df['embedding'].iloc[0])}")
        typer.echo(f"💾 Embeddings saved to: {config_dict['output']['base_path']}/{config_dict['output']['embeddings_file']}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="pipeline")
def full_pipeline(
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Run the complete pipeline: download data and generate embeddings.
    
    Args:
        config: Path to YAML configuration file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        logger.info("=" * 80)
        logger.info("STARTING FULL AMAZON DATASET PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Check OpenAI API key
        if not check_openai_api_key():
            error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            logger.error(error_msg)
            typer.echo(f"❌ {error_msg}", err=True)
            raise typer.Exit(1)
        
        # Step 1: Download and process data
        logger.info("STEP 1: DOWNLOADING AND PROCESSING DATA")
        logger.info("-" * 50)
        
        downloader = AmazonDataDownloader(config_dict)
        combined_df, metadata_df = downloader.download_and_process()
        
        typer.echo(f"✅ Step 1 completed: Downloaded {len(combined_df)} records with {len(metadata_df)} metadata entries")
        
        # Step 2: Generate embeddings
        logger.info("STEP 2: GENERATING EMBEDDINGS")
        logger.info("-" * 50)
        
        embeddings_generator = TitleEmbeddingsGenerator(config_dict)
        embeddings_df = embeddings_generator.generate_embeddings(metadata_df)
        
        typer.echo(f"✅ Step 2 completed: Generated embeddings for {len(embeddings_df)} titles")
        
        # Final summary
        logger.info("=" * 80)
        logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        typer.echo("\n🎉 Full pipeline completed successfully!")
        typer.echo(f"📊 Dataset records: {len(combined_df)}")
        typer.echo(f"🏷️  Metadata records: {len(metadata_df)}")
        typer.echo(f"🧮 Embeddings generated: {len(embeddings_df)}")
        typer.echo(f"📐 Embedding dimension: {len(embeddings_df['embedding'].iloc[0])}")
        typer.echo(f"💾 All data saved to: {config_dict['output']['base_path']}")
        
        # Log file locations
        output_files = [
            config_dict['output']['train_file'],
            config_dict['output']['val_file'], 
            config_dict['output']['test_file'],
            config_dict['output']['metadata_file'],
            config_dict['output']['embeddings_file']
        ]
        
        typer.echo("\n📁 Generated files:")
        for file in output_files:
            file_path = Path(config_dict['output']['base_path']) / file
            if file_path.exists():
                typer.echo(f"  ✓ {file}")
            else:
                typer.echo(f"  ❌ {file} (not found)")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        typer.echo(f"❌ Pipeline failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="similar")
def find_similar(
    query: Annotated[str, typer.Argument(help="Query title to find similar titles for")],
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml",
    top_k: Annotated[int, typer.Option(help="Number of similar titles to return")] = 5
) -> None:
    """
    Find titles similar to a query title using embeddings.
    
    Args:
        query: Query title to find similar titles for
        config: Path to YAML configuration file
        top_k: Number of similar titles to return
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        logger.info(f"Finding titles similar to: '{query}'")
        
        # Check if embeddings file exists
        embeddings_file = Path(config_dict['output']['base_path']) / config_dict['output']['embeddings_file']
        
        if not embeddings_file.exists():
            error_msg = f"Embeddings file not found at {embeddings_file}. Please run 'generate-embeddings' first."
            logger.error(error_msg)
            typer.echo(f"❌ {error_msg}", err=True)
            raise typer.Exit(1)
        
        # Initialize embeddings generator and load embeddings
        embeddings_generator = TitleEmbeddingsGenerator(config_dict)
        embeddings_df = embeddings_generator.load_embeddings()
        
        # Find similar titles
        similar_titles = embeddings_generator.find_similar_titles(query, embeddings_df, top_k)
        
        # Display results
        typer.echo(f"\n🔍 Top {top_k} titles similar to: '{query}'\n")
        typer.echo("Rank | Similarity | ASIN | Title")
        typer.echo("-" * 80)
        
        for idx, (_, row) in enumerate(similar_titles.iterrows(), 1):
            similarity = row['similarity']
            asin = row['parent_asin']
            title = row['title']
            
            # Truncate long titles
            if len(title) > 50:
                title = title[:47] + "..."
            
            typer.echo(f"{idx:4d} | {similarity:10.4f} | {asin} | {title}")
        
        logger.info(f"Found {len(similar_titles)} similar titles")
        
    except Exception as e:
        logger.error(f"Failed to find similar titles: {str(e)}")
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="status")
def status(
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Check the status of generated files and provide information about the dataset.
    
    Args:
        config: Path to YAML configuration file
    """
    try:
        # Load configuration
        config_dict = load_config(config)
        
        output_path = Path(config_dict['output']['base_path'])
        
        typer.echo("📋 Amazon Dataset Preprocessing Status\n")
        
        # Check file existence
        files_to_check = {
            "Training data": config_dict['output']['train_file'],
            "Validation data": config_dict['output']['val_file'],
            "Test data": config_dict['output']['test_file'],
            "Metadata": config_dict['output']['metadata_file'],
            "Embeddings": config_dict['output']['embeddings_file']
        }
        
        existing_files = []
        
        for description, filename in files_to_check.items():
            file_path = output_path / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                typer.echo(f"✅ {description}: {filename} ({file_size:.1f} MB)")
                existing_files.append(file_path)
            else:
                typer.echo(f"❌ {description}: {filename} (not found)")
        
        # Provide recommendations
        typer.echo(f"\n💡 Recommendations:")
        
        if len(existing_files) == 0:
            typer.echo("  - Run 'python main.py pipeline' to start from scratch")
        elif len(existing_files) < 4:
            typer.echo("  - Run 'python main.py download' to complete data download")
        elif len(existing_files) == 4:
            typer.echo("  - Run 'python main.py embeddings' to create embeddings")
        else:
            typer.echo("  - All files present! Use 'similar' to search for similar titles")
        
        # Check API key
        if check_openai_api_key():
            typer.echo("  ✅ OpenAI API key is configured")
        else:
            typer.echo("  ❌ OpenAI API key not found (required for embeddings)")
            typer.echo("     Set OPENAI_API_KEY environment variable")
        
    except Exception as e:
        typer.echo(f"❌ Error checking status: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 