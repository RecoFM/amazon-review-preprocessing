# Amazon Review Dataset Preprocessing

A comprehensive tool for downloading Amazon review datasets, processing metadata, and generating embeddings for product titles using OpenAI's text-embedding-3-large model. The processed dataset can be uploaded to Hugging Face Hub for easy sharing and distribution.

## Features

- 🔽 **Dataset Download**: Automatically downloads Amazon review datasets from Hugging Face
- 📊 **Data Processing**: Processes train/validation/test splits with comprehensive statistics
- 🏷️ **Metadata Management**: Fetches and filters product metadata
- 🧮 **Embeddings Generation**: Creates embeddings for product titles using OpenAI's text-embedding-3-large
- 💾 **Parquet Storage**: Efficient data storage in parquet format with organized directory structure
- 📝 **Comprehensive Logging**: Detailed logging of all operations
- ⚙️ **YAML Configuration**: Flexible configuration management
- 🔄 **Progress Tracking**: Saves progress during embeddings generation
- ✅ **Data Verification**: Includes notebook for verifying data integrity
- 🚀 **Hugging Face Upload**: Direct upload to Hugging Face Hub for dataset sharing

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings generation)
- HuggingFace account (for dataset access)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd amazon-review-preprocessing
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Required for embeddings generation
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Required for Hugging Face Hub upload
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

## Configuration

The system uses a YAML configuration file (`config.yaml`) to manage all settings:

```yaml
dataset:
  name: "McAuley-Lab/Amazon-Reviews-2023"
  subset: "5core_last_out_w_his_All_Beauty"
  metadata_subset: "raw_meta_All_Beauty"
  splits: ["train", "valid", "test"]

metadata:
  columns_to_keep: ["title", "parent_asin"]
  
output:
  base_path: "data"
  dataset_path: "{base_path}/{dataset.subset}"
  train_path: "{dataset_path}/train"
  val_path: "{dataset_path}/val"
  test_path: "{dataset_path}/test"
  meta_path: "{dataset_path}/meta"
  embeddings_path: "{dataset_path}/embeddings"
  train_file: "train_data.parquet"
  val_file: "val_data.parquet"
  test_file: "test_data.parquet"
  metadata_file: "metadata.parquet"
  embeddings_file: "title_embeddings.parquet"

embeddings:
  model: "text-embedding-3-large"
  batch_size: 100
  max_retries: 3
  delay_between_batches: 1.0

huggingface:
  repository_id: "username/dataset-name"  # Set this to your desired repo name
  private: false
  token: null  # Will be loaded from HUGGINGFACE_TOKEN env variable
  create_repo: true
  commit_message: "Upload preprocessed Amazon review dataset with embeddings"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "amazon_preprocessing.log"
```

## Usage

### Command Line Interface

The system provides several commands through `main.py`:

#### 1. Check Status
```bash
python main.py status
```
Shows the current status of all files and directories, and provides recommendations.

#### 2. Download Data
```bash
python main.py download
```
Downloads and processes the Amazon dataset with metadata.

#### 3. Generate Embeddings
```bash
python main.py embeddings
```
Generates embeddings for product titles (requires data to be downloaded first).

#### 4. Full Pipeline
```bash
python main.py pipeline
```
Runs the complete pipeline: downloads data and generates embeddings.

#### 5. Find Similar Titles
```bash
python main.py similar "wireless bluetooth headphones" --top-k 10
```
Finds similar product titles using embeddings.

#### 6. Upload to Hugging Face Hub
```bash
python main.py upload
```
Uploads the processed dataset to Hugging Face Hub.

### Custom Configuration
You can use a custom configuration file:

```bash
python main.py pipeline --config my_config.yaml
```

### Uploading to HuggingFace Hub
To upload the processed dataset to HuggingFace Hub, use the following command:
```bash
huggingface-cli upload ChernovAndrei/reco-fm-data . --repo-type=dataset
```

## Directory Structure

The system organizes data in the following structure:

```
data/
└── {dataset.subset}/
    ├── train/
    │   └── train_data.parquet
    ├── val/
    │   └── val_data.parquet
    ├── test/
    │   └── test_data.parquet
    ├── meta/
    │   └── metadata.parquet
    └── embeddings/
        ├── title_embeddings.parquet
        └── embeddings_progress.json
```

## API Usage

You can also use the components programmatically:

```python
from data_downloader import AmazonDataDownloader, load_config, setup_logging
from embeddings_generator import TitleEmbeddingsGenerator
from huggingface_uploader import HuggingFaceUploader

# Load configuration
config = load_config("config.yaml")
setup_logging(config)

# Download data
downloader = AmazonDataDownloader(config)
combined_df, metadata_df = downloader.download_and_process()

# Generate embeddings
embeddings_generator = TitleEmbeddingsGenerator(config)
embeddings_df = embeddings_generator.generate_embeddings(metadata_df)

# Upload to Hugging Face Hub
uploader = HuggingFaceUploader(config)
uploader.upload_dataset()

# Find similar titles
similar = embeddings_generator.find_similar_titles(
    "wireless headphones", 
    embeddings_df, 
    metadata_df,
    top_k=5
)
```

## Dataset Information

The system works with the Amazon Reviews 2023 dataset from Hugging Face:
- **Dataset**: McAuley-Lab/Amazon-Reviews-2023
- **Default Subset**: 5core_last_out_w_his_All_Beauty (Beauty products)
- **Metadata**: Raw product metadata with titles and ASINs

### Dataset Statistics (Beauty subset example)
- **Train**: 2,029 reviews
- **Validation**: 253 reviews
- **Test**: 253 reviews
- **Unique Products**: 356 items with metadata

## Embeddings

- **Model**: OpenAI's text-embedding-3-large
- **Dimension**: 3072
- **Use Cases**: 
  - Product similarity search
  - Recommendation systems
  - Content-based filtering
  - Clustering and categorization

## Error Handling

The system includes comprehensive error handling:
- API rate limiting and retries for OpenAI requests
- Progress tracking and resumption for embeddings generation
- Graceful handling of missing data
- Detailed error logging
- Validation of data integrity

## Logging

All operations are logged with different levels:
- **INFO**: General progress and statistics
- **WARNING**: Non-critical issues (e.g., missing data)
- **ERROR**: Critical errors that stop execution
- **DEBUG**: Detailed technical information

Logs are written to both console and file (`amazon_preprocessing.log`).

## Data Verification

The project includes a Jupyter notebook (`verify_data.ipynb`) for verifying:
- Existence of all required files and directories
- Data split statistics and integrity
- Metadata coverage
- Embeddings dimensions and coverage
- Missing data detection

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   ```
   Error: OpenAI API key not found
   ```
   Solution: Set the `OPENAI_API_KEY` in `.env` file.

2. **Hugging Face Token Not Found**
   ```
   Error: Hugging Face token not found
   ```
   Solution: Set the `HUGGINGFACE_TOKEN` in `.env` file.

3. **Dataset Download Failures**
   ```
   Error: Failed to download dataset
   ```
   Solution: Check internet connection and Hugging Face dataset availability.

4. **Memory Issues**
   ```
   Error: Out of memory
   ```
   Solution: Reduce batch size in config or use a machine with more RAM.

5. **Rate Limiting**
   ```
   Warning: Rate limited by OpenAI API
   ```
   Solution: The system automatically retries with exponential backoff and saves progress.

6. **API Quota Exceeded**
   ```
   Error: API quota exceeded
   ```
   Solution: Check your OpenAI account quota and billing. The system saves progress and can resume later.

### Getting Help

- Check the log file for detailed error information
- Use `python main.py status` to diagnose issues
- Run `verify_data.ipynb` to check data integrity
- Ensure all dependencies are installed correctly
- Verify your API keys and tokens are set correctly

## Development

### Project Structure
```
amazon-review-preprocessing/
├── main.py                    # Main CLI orchestrator
├── data_downloader.py         # Dataset download and processing
├── embeddings_generator.py    # Embeddings generation with OpenAI
├── huggingface_uploader.py    # Hugging Face Hub upload functionality
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── verify_data.ipynb        # Data verification notebook
├── README.md                # This file
└── data/                    # Output directory (created automatically)
```

### Adding New Features

1. **New Dataset**: Modify `config.yaml` to specify different dataset/subset
2. **Custom Embeddings**: Extend `TitleEmbeddingsGenerator` class
3. **Additional Processing**: Add methods to `AmazonDataDownloader` class
4. **New Commands**: Add commands to `main.py` with `@app.command()` decorator
5. **Upload Options**: Modify `HuggingFaceUploader` class for custom upload behavior

## License

This project is open source. Please check the license file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Amazon Reviews 2023 dataset by McAuley Lab
- Hugging Face Datasets library
- OpenAI for embeddings API
- LangChain for embeddings integration
