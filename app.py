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
import json

AVAILABLE_COLORSCALES = [
    'Viridis', 'Plasma', 'Inferno', 'Magma',  # Sequential
    'RdBu', 'RdYlBu', 'PiYG', 'PRGn',        # Diverging
    'Rainbow', 'Jet', 'Turbo',                # Spectral
    'Blues', 'Greens', 'Reds', 'Purples'      # Single color
]


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

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
@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        text = data.get('text', '')
        logger.debug(f"Received text: {text}")

        if not text:
            raise ValueError("No text provided")

        # Get full tokenization details
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        # Get tokens with special tokens
        full_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Get embeddings and attention
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True)
        
        embeddings = outputs.last_hidden_state.squeeze().numpy()
        
        # Calculate appropriate perplexity
        n_samples = len(full_tokens)
        perplexity = min(30, max(5, n_samples - 1))
        logger.debug(f"Using perplexity: {perplexity}")

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
            logger.debug(f"TSNE reduction completed. Shape: {embeddings_3d.shape}")
        else:
            pca = PCA(n_components=min(3, n_samples))
            embeddings_3d = pca.fit_transform(embeddings)
            if embeddings_3d.shape[1] < 3:
                padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
                embeddings_3d = np.hstack((embeddings_3d, padding))
            logger.debug(f"PCA reduction completed. Shape: {embeddings_3d.shape}")
        
        # Validate embeddings_3d
        logger.debug(f"Final embeddings_3d shape: {embeddings_3d.shape}")
        logger.debug(f"Embeddings_3d contains NaN: {np.isnan(embeddings_3d).any()}")
        logger.debug(f"Embeddings_3d contains Inf: {np.isinf(embeddings_3d).any()}")
        logger.debug(f"Embeddings_3d range: [{embeddings_3d.min():.3f}, {embeddings_3d.max():.3f}]")

        # Prepare token types and markers
        token_types = [
            'Special Token' if t in ['[CLS]', '[SEP]'] else
            'Subword' if t.startswith('##') else
            'Word' for t in full_tokens
        ]

        marker_symbols = [
            'circle' if not t.startswith('##') else 'diamond'
            for t in full_tokens
        ]

        # Normalize coordinates to a reasonable range for better visualization
        coords = embeddings_3d.astype(np.float64)  # Convert to float64 for better JSON serialization
        coord_range = max(abs(coords.min()), abs(coords.max()))
        if coord_range > 10:  # Only normalize if coordinates are too large
            coords = coords / coord_range * 10
            logger.debug(f"Normalized coordinates to range [-10, 10]")
        
        # Convert to regular Python lists with float values
        x_coords = [float(x) for x in coords[:, 0]]
        y_coords = [float(y) for y in coords[:, 1]]
        z_coords = [float(z) for z in coords[:, 2]]
        
        # Create scatter data for 3D visualization
        scatter_data = {
            'data': [{
                'type': 'scatter3d',
                'x': x_coords,
                'y': y_coords,
                'z': z_coords,
                'mode': 'markers+text',  # Add text labels to the markers
                'text': full_tokens,  # Token names for labels
                'textposition': 'top center',  # Position text above the points
                'textfont': {
                    'size': 8,  # Smaller font size to avoid overlap
                    'color': 'black'
                },
                'textangle': 0,  # No rotation for better readability
                'marker': {
                    'size': 8,  # Increased size for better visibility
                    'color': list(range(len(full_tokens))),
                    'colorscale': 'Viridis',
                    'opacity': 0.9,
                    'showscale': True,
                    'colorbar': {
                        'title': 'Token Position',
                        'thickness': 15,
                        'len': 0.5,
                        'x': 0.85
                    },
                    'line': {
                        'color': 'black',
                        'width': 1
                    }
                },
                'hovertemplate': 
                    '<b>Token:</b> %{text}<br>' +
                    '<b>Position:</b> %{marker.color}<br>' +
                    '<b>X:</b> %{x:.3f}<br>' +
                    '<b>Y:</b> %{y:.3f}<br>' +
                    '<b>Z:</b> %{z:.3f}<br>' +
                    '<extra></extra>'
            }],
            'layout': {
                'title': {
                    'text': '3D Token Embeddings',
                    'font': {'size': 16}
                },
                'scene': {
                    'xaxis': {
                        'title': 'X',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 0].min() - 1), float(coords[:, 0].max() + 1)]
                    },
                    'yaxis': {
                        'title': 'Y',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 1].min() - 1), float(coords[:, 1].max() + 1)]
                    },
                    'zaxis': {
                        'title': 'Z',
                        'showgrid': True,
                        'zeroline': True,
                        'range': [float(coords[:, 2].min() - 1), float(coords[:, 2].max() + 1)]
                    },
                    'camera': {
                        'up': {'x': 0, 'y': 0, 'z': 1},
                        'center': {'x': 0, 'y': 0, 'z': 0},
                        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
                    },
                    'aspectmode': 'cube'
                },
                'margin': {'l': 0, 'r': 0, 'b': 0, 't': 50},
                'showlegend': False,
                'width': 800,
                'height': 600
            }
        }



        # Get attention scores - use the last layer for better attention patterns
        attention_scores = outputs.attentions[-1].mean(dim=1)[0].numpy()
        
        # Normalize attention scores for better visualization
        # Apply softmax to make the attention distribution more pronounced
        attention_scores_softmax = np.exp(attention_scores * 10) / np.sum(np.exp(attention_scores * 10), axis=-1, keepdims=True)
        
        # Use a combination of raw and softmax scores for better visualization
        attention_scores_final = 0.7 * attention_scores + 0.3 * attention_scores_softmax
        
        # Debug logging
        logger.debug(f"Scatter data structure: {scatter_data}")
        logger.debug(f"Embeddings 3D shape: {embeddings_3d.shape}")
        logger.debug(f"Number of tokens: {len(full_tokens)}")
        logger.debug(f"X coordinates: {embeddings_3d[:, 0].tolist()[:5]}...")  # First 5 values
        logger.debug(f"Y coordinates: {embeddings_3d[:, 1].tolist()[:5]}...")  # First 5 values
        logger.debug(f"Z coordinates: {embeddings_3d[:, 2].tolist()[:5]}...")  # First 5 values
        
        # Attention scores debugging
        logger.debug(f"Attention scores shape: {attention_scores.shape}")
        logger.debug(f"Raw attention scores range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
        logger.debug(f"Final attention scores range: [{attention_scores_final.min():.4f}, {attention_scores_final.max():.4f}]")
        logger.debug(f"Self-attention diagonal: {np.diag(attention_scores_final).tolist()}")
        
        # Prepare response data
        response_data = {
            'tokens': full_tokens[1:-1],
            'token_ids': encoded['input_ids'][0].tolist(),
            'embeddings': [[float(val) for val in row] for row in embeddings_3d.tolist()],
            'original_embeddings': [[float(val) for val in row] for row in embeddings.tolist()],
            'attention_scores': [[float(val) for val in row] for row in attention_scores_final.tolist()],
            'plot_data': scatter_data,
            'technical_info': {
                'input_text_length': len(text),
                'num_tokens': len(full_tokens),
                'embedding_shape': embeddings.shape,
                'perplexity': perplexity,
                'model_name': model_name,
                'vocab_size': tokenizer.vocab_size,
                'hidden_size': embeddings.shape[-1],
                'num_attention_heads': model.config.num_attention_heads,
                'num_hidden_layers': model.config.num_hidden_layers
            }
        }
        response_data['available_colorscales'] = AVAILABLE_COLORSCALES
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
