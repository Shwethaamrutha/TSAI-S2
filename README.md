# LLM Visualizer

A comprehensive web-based tool for visualizing how Large Language Models (LLMs) like BERT process and understand text. This project provides interactive 3D embeddings visualization, attention matrix analysis, and step-by-step tokenization process visualization.

## 🚀 Features

### 3D Embedding Visualization
- **Interactive 3D scatter plot** of token embeddings
- **Color-coded tokens** by position in sequence (purple to yellow gradient)
- **Text labels** on each point showing token names
- **Hover information** with detailed token data
- **Download functionality** for embeddings in CSV and JSON formats
- **Responsive design** with zoom, rotate, and pan controls

### Attention Matrix Analysis
- **Interactive heatmap** showing attention scores between tokens
- **Layer selection** (first, middle, last attention layers)
- **Numerical annotations** showing exact attention values
- **Toggle controls** for showing/hiding attention values
- **Enhanced color scale** for better pattern visualization
- **Self-attention highlighting** on diagonal elements

### Tokenization Process Visualization
- **Step-by-step breakdown** of the tokenization process
- **Interactive navigation** through 4 detailed steps:
  1. Input Text Analysis
  2. Tokenization Results
  3. Token ID Mapping
  4. Vector Embeddings
- **Color-coded tokens** by type (full, subword, special)
- **Statistical information** for each step

### Technical Information Panel
- **Model details** (BERT-base-uncased)
- **Input analysis** (character count, word count, token count)
- **Embedding details** (dimensions, perplexity, reduction method)
- **Attention details** (matrix size, score ranges)

## 🛠️ Technology Stack

### Backend
- **Python 3.12+**
- **Flask** - Web framework
- **Transformers** - Hugging Face library for BERT model
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **scikit-learn** - TSNE and PCA for dimensionality reduction
- **Plotly** - Data visualization

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling and responsive design
- **JavaScript (ES6+)** - Interactive functionality
- **Plotly.js** - 3D and 2D visualizations
- **D3.js** - Data manipulation

### Development Tools
- **uv** - Python package manager
- **pyproject.toml** - Project configuration and dependencies
- **Virtual environment** - Isolated dependencies

## 📋 Prerequisites

- Python 3.12 or higher
- Node.js (optional, for development)
- Modern web browser with JavaScript enabled

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LLMVisualizer.git
cd LLMVisualizer
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

### 4. Run the Application
```bash
# Start the Flask development server
uv run python app.py

# Or alternatively, you can use:
python app.py
```

The application will be available at `http://localhost:5000`

## 📖 Usage

### Basic Usage
1. **Open the application** in your web browser
2. **Enter text** in the input field (e.g., "Cats and dogs, humanity's two most popular domestic companions...")
3. **Click "Process"** to generate visualizations
4. **Explore the results**:
   - 3D embedding visualization
   - Attention matrix heatmap
   - Step-by-step tokenization process

### Advanced Features

#### 3D Visualization Controls
- **Toggle text labels**: Show/hide token names on 3D points
- **Download embeddings**: Get CSV or JSON files with complete embedding data
- **Interactive controls**: Zoom, rotate, and pan the 3D space

#### Attention Matrix Controls
- **Layer selection**: Choose which attention layer to visualize
- **Toggle annotations**: Show/hide numerical attention values
- **Hover for details**: Get exact attention scores

#### Tokenization Process
- **Navigate through steps**: Use Previous/Next buttons
- **View detailed breakdown**: See how text is processed at each stage
- **Understand tokenization**: Learn about subword tokenization and embeddings

## 🏗️ Project Structure

```
LLMVisualizer/
├── app.py                 # Main Flask application
├── main.py               # Entry point
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock              # Locked dependency versions
├── README.md            # Project documentation
├── static/
│   ├── css/
│   │   └── style.css    # Stylesheets
│   └── js/
│       └── visualization.js  # Frontend JavaScript
└── templates/
    └── index.html       # Main HTML template
```

## 🔧 Implementation Details

### Backend Architecture

#### Model Loading
```python
# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
```

#### Text Processing Pipeline
1. **Tokenization**: Convert text to tokens using BERT tokenizer
2. **Embedding Generation**: Generate 768-dimensional embeddings
3. **Dimensionality Reduction**: Use TSNE/PCA to reduce to 3D
4. **Attention Calculation**: Extract attention scores from model layers
5. **Data Preparation**: Format data for frontend visualization

#### Key Algorithms
- **TSNE (t-SNE)**: Non-linear dimensionality reduction for embeddings
- **PCA**: Linear dimensionality reduction for small datasets
- **Softmax Normalization**: Enhance attention score visualization
- **Coordinate Normalization**: Scale coordinates for better visualization

### Frontend Architecture

#### Visualization Components
- **3D Scatter Plot**: Plotly.js for interactive 3D visualization
- **Attention Heatmap**: 2D heatmap with custom color scales
- **Step-by-step Process**: Dynamic content updates with JavaScript

#### Interactive Features
- **Real-time Updates**: Dynamic visualization updates
- **Download Functionality**: Client-side file generation
- **Responsive Design**: Mobile-friendly interface
- **Error Handling**: Graceful error handling and user feedback

## 🎯 Key Features Explained

### 3D Embedding Visualization
The 3D plot shows how the model represents tokens in semantic space:
- **Proximity**: Similar tokens appear closer together
- **Color Coding**: Position in sequence (purple = beginning, yellow = end)
- **Contextual Embeddings**: Same word can have different positions based on context

### Attention Matrix Analysis
The attention heatmap reveals how the model "pays attention" to different tokens:
- **Self-Attention**: Diagonal elements show tokens attending to themselves
- **Cross-Attention**: Off-diagonal elements show relationships between tokens
- **Function Words**: Words like "and", "with", "while" act as attention hubs

### Tokenization Process
Step-by-step breakdown of how text is processed:
1. **Input Analysis**: Character count, word count, text statistics
2. **Tokenization**: Breaking text into tokens and subword units
3. **ID Mapping**: Converting tokens to vocabulary IDs
4. **Embeddings**: Generating vector representations

## 🔍 Example Analysis

### Sample Text
"Cats and dogs, humanity's two most popular domestic companions, offer distinct types of companionship, with dogs known for their social nature, loyalty to their owners, and need for regular exercise and training, while cats are celebrated for their independence, self-sufficiency, and lower-maintenance nature"

### Key Insights
- **Semantic Clustering**: "self" and "sufficiency" cluster together
- **Contextual Differences**: "cats" and "dogs" are positioned separately based on their contrasting characteristics
- **Attention Patterns**: Function words like "and", "with", "while" show strong horizontal attention lines
- **Self-Attention**: "cats" has stronger self-attention than "dogs" due to more complex context

## 🚀 Deployment

### Local Development
```bash
# Using uv (recommended)
uv run python app.py

# Or using Python directly
python app.py
```

### Production Deployment
For production deployment, consider:
- **WSGI Server**: Use Gunicorn or uWSGI
- **Environment Variables**: Configure model paths and settings
- **Caching**: Implement caching for model loading
- **Load Balancing**: For high-traffic scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and BERT model
- **Plotly** for the interactive visualization library
- **scikit-learn** for dimensionality reduction algorithms
- **Flask** for the web framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation

---

**Built with ❤️ for AI interpretability and education**
