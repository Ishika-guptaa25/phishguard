# PhishGuard

A machine learning-based phishing detection system designed to identify and classify suspicious websites and URLs with high accuracy.

## Overview

PhishGuard leverages advanced machine learning algorithms to detect phishing websites in real-time. The system analyzes URL characteristics and website features to classify webpages as legitimate or phishing attempts, providing robust protection against online fraud.

## Features

- **URL-Based Detection**: Analyzes URL structure and characteristics to identify phishing patterns
- **Feature Extraction**: Extracts comprehensive features from URLs for accurate classification
- **Machine Learning Models**: Utilizes trained models for robust phishing detection
- **Real-Time Analysis**: Quick and efficient classification of URLs
- **Docker Support**: Containerized deployment for easy scaling and integration

## Tech Stack

- **Language**: Python 3.x
- **ML Framework**: Scikit-learn / XGBoost
- **Containerization**: Docker & Docker Compose
- **Data Processing**: Pandas, NumPy
- **Web Framework**: Flask

## Project Structure

```
phishguard/
├── app.py                        # Flask web application
├── model.py                      # Model training and inference
├── feature_extractor.py          # URL feature extraction
├── main.py                       # Main execution script
├── convert_kaggle_dataset.py     # Dataset preprocessing utility
├── inspect_csv.py                # Data inspection tools
├── simple_inspect.py             # Additional inspection utilities
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup configuration
├── config.toml                   # Configuration settings
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Multi-container orchestration
└── .gitignore                    # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Docker & Docker Compose (optional)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ishika-guptaa25/phishguard.git
   cd phishguard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**
   ```bash
   docker build -t phishguard .
   docker run -p 5000:5000 phishguard
   ```

## Usage

### Web Application

Start the Flask app and access the web interface:
```bash
python app.py
```

### Command Line

Analyze URLs directly:
```bash
python main.py <url>
```

### Feature Extraction

Extract features from URLs for analysis:
```bash
python feature_extractor.py
```

### Dataset Processing

Convert Kaggle datasets to the required format:
```bash
python convert_kaggle_dataset.py
```

## Configuration

Edit `config.toml` to customize:
- Model parameters
- Feature extraction settings
- Application configuration
- Database connections

## Model Training

To train or retrain the phishing detection model:

1. Prepare your dataset
2. Run the feature extractor on your data
3. Execute model training:
   ```bash
   python model.py --train --dataset <path_to_dataset>
   ```

## Performance

PhishGuard achieves high accuracy in detecting phishing websites through:
- Comprehensive URL feature analysis
- Machine learning model ensemble techniques
- Continuous model optimization

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

## Disclaimer

While PhishGuard strives for high accuracy, no phishing detection system is 100% foolproof. Always exercise caution when browsing the internet and never share sensitive information on unverified websites.

---
