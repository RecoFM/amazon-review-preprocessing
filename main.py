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
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Generate embeddings for product titles.
    
    Args:
        config: Path to YAML configuration file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check OpenAI API key
        check_openai_api_key()
        
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        logger.info("=" * 60)
        logger.info("STARTING EMBEDDINGS GENERATION")
        logger.info("=" * 60)
        
        # Initialize generator
        generator = TitleEmbeddingsGenerator(config_dict)
        
        # Generate embeddings
        embeddings_df = generator.generate_embeddings()
        
        logger.info("=" * 60)
        logger.info("EMBEDDINGS GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        typer.echo(f"✅ Successfully generated embeddings")
        typer.echo(f"📊 Total embeddings: {len(embeddings_df)}")
        typer.echo(f"💾 Embeddings saved to: {config_dict['output']['embeddings_file']}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="pipeline")
def run_pipeline(
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Run the complete pipeline: download data and generate embeddings.
    
    Args:
        config: Path to YAML configuration file
    """
    try:
        # Download data
        download_data(config=config)
        
        # Generate embeddings
        generate_embeddings(config=config)
        
        typer.echo("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Pipeline failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command(name="similar")
def find_similar(
    query: Annotated[str, typer.Argument(help="Query text to find similar titles for")],
    top_k: Annotated[int, typer.Option(help="Number of similar titles to return")] = 5,
    config: Annotated[Optional[str], typer.Option(help="Path to YAML configuration file")] = "config.yaml"
) -> None:
    """
    Find similar product titles using embeddings.
    
    Args:
        query: Query text to find similar titles for
        top_k: Number of similar titles to return
        config: Path to YAML configuration file
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check OpenAI API key
        check_openai_api_key()
        
        # Load configuration
        config_dict = load_config(config)
        setup_logging(config_dict)
        
        # Initialize generator
        generator = TitleEmbeddingsGenerator(config_dict)
        
        # Find similar titles
        similar_df = generator.find_similar_titles(query, top_k=top_k)
        
        # Display results
        typer.echo(f"\n🔍 Similar titles to: {query}\n")
        for _, row in similar_df.iterrows():
            typer.echo(f"📦 {row['title']}")
            typer.echo(f"   Score: {row['score']:.3f}")
            typer.echo(f"   ASIN: {row['parent_asin']}\n")
        
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
        
        # Setup paths
        base_path = Path(config_dict['output']['base_path'])
        dataset_path = base_path / config_dict['dataset']['subset']
        
        typer.echo("📋 Amazon Dataset Preprocessing Status\n")
        
        # Check directory structure
        directories = {
            "Train": dataset_path / "train",
            "Validation": dataset_path / "val",
            "Test": dataset_path / "test",
            "Metadata": dataset_path / "meta",
            "Embeddings": dataset_path / "embeddings"
        }
        
        for description, dir_path in directories.items():
            if dir_path.exists():
                typer.echo(f"✅ {description} directory: {dir_path.relative_to(base_path)}")
            else:
                typer.echo(f"❌ {description} directory: {dir_path.relative_to(base_path)} (not found)")
        
        typer.echo("\n📦 Data Files:")
        
        # Check file existence
        files_to_check = {
            "Training data": dataset_path / "train" / config_dict['output']['train_file'],
            "Validation data": dataset_path / "val" / config_dict['output']['val_file'],
            "Test data": dataset_path / "test" / config_dict['output']['test_file'],
            "Metadata": dataset_path / "meta" / config_dict['output']['metadata_file'],
            "Embeddings": dataset_path / "embeddings" / config_dict['output']['embeddings_file']
        }
        
        existing_files = []
        missing_files = []
        
        for description, file_path in files_to_check.items():
            if file_path.exists():
                existing_files.append(description)
                typer.echo(f"✅ {description}: {file_path.relative_to(base_path)}")
            else:
                missing_files.append(description)
                typer.echo(f"❌ {description}: {file_path.relative_to(base_path)} (not found)")
        
        # Provide recommendations
        if missing_files:
            typer.echo("\n⚠️  Recommendations:")
            if all(f in missing_files for f in ["Training data", "Validation data", "Test data", "Metadata"]):
                typer.echo("   • Run 'python main.py download' to download and process the dataset")
            if "Embeddings" in missing_files:
                typer.echo("   • Run 'python main.py embeddings' to generate embeddings")
            if len(missing_files) == len(files_to_check):
                typer.echo("   • Run 'python main.py pipeline' to run the complete pipeline")
        else:
            typer.echo("\n✨ All files are present and ready to use!")
        
    except Exception as e:
        typer.echo(f"❌ Error checking status: {str(e)}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 