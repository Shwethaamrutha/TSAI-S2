import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        text = data.get('text', '')
        logger.debug(f"Received text: {text}")

        if not text:
            raise ValueError("No text provided")

        # Tokenization
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        logger.debug(f"Tokenized into {len(tokens)} tokens")

        # Embeddings
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        embeddings = outputs.last_hidden_state.squeeze().numpy()
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")

        # Calculate appropriate perplexity
        n_samples = len(tokens)
        perplexity = min(30, max(5, n_samples - 1))
        logger.debug(f"Using perplexity: {perplexity}")

        # Calculate technical details
        technical_info = {
            'input_text_length': len(text),
            'num_tokens': len(tokens),
            'embedding_shape': embeddings.shape,
            'perplexity': perplexity,
            'model_name': model_name,
            'vocab_size': tokenizer.vocab_size,
            'hidden_size': embeddings.shape[-1],
            'num_attention_heads': model.config.num_attention_heads,
            'num_hidden_layers': model.config.num_hidden_layers
        }

        # 3D dimensionality reduction
        if n_samples >= 4:
            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=perplexity,
                n_iter_without_progress=300,
                min_grad_norm=1e-7,
            )
            embeddings_3d = tsne.fit_transform(embeddings)
        else:
            pca = PCA(n_components=min(3, n_samples))
            embeddings_3d = pca.fit_transform(embeddings)
            if embeddings_3d.shape[1] < 3:
                padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
                embeddings_3d = np.hstack((embeddings_3d, padding))
        
        logger.debug(f"3D embeddings shape: {embeddings_3d.shape}")

        # Create 3D scatter plot data
        scatter_data = {
            'data': [{
                'type': 'scatter3d',
                'x': embeddings_3d[:, 0].tolist(),
                'y': embeddings_3d[:, 1].tolist(),
                'z': embeddings_3d[:, 2].tolist(),
                'mode': 'markers+text',
                'text': tokens,
                'hoverinfo': 'text',
                'marker': {
                    'size': 8,
                    'color': list(range(len(tokens))),
                    'colorscale': 'Viridis',
                    'opacity': 0.8
                }
            }],
            'layout': {
                'scene': {
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'zaxis': {'title': 'Z'},
                    'camera': {
                        'up': {'x': 0, 'y': 0, 'z': 1},
                        'center': {'x': 0, 'y': 0, 'z': 0},
                        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    }
                },
                'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'showlegend': False,
                'title': f'Token Embeddings ({n_samples} tokens)'
            }
        }

        # Get attention scores
        attention_scores = outputs.attentions[0].mean(dim=1)[0].numpy()
        logger.debug(f"Attention scores shape: {attention_scores.shape}")

        response_data = {
            'tokens': tokens,
            'token_ids': token_ids,
            'embeddings': embeddings_3d.tolist(),
            'attention_scores': attention_scores.tolist(),
            'plot_data': scatter_data,
            'technical_info': technical_info
        }
        
        logger.debug("Successfully prepared response data")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
