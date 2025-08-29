document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('process-btn');
    const inputText = document.getElementById('input-text');
    
    processBtn.addEventListener('click', function() {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        
        const text = inputText.value.trim();
        if (text.length === 0) {
            alert('Please enter some text');
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
            return;
        }
        
        console.log('Processing text:', text);
        
        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({text: text}),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            
            // Update all visualizations
            updateTokenList(data.tokens, data.token_ids);
            updateTechnicalInfo(data);  // Pass the entire data object
            updateEmbeddingPlot(data.plot_data);
            updateAttentionHeatmap(data.attention_scores, data.tokens);
        })
        .catch(error => {
            console.error('Error details:', error);
            alert(`Error: ${error.message}`);
        })
        .finally(() => {
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
        });
    });

    function updateTokenList(tokens, tokenIds) {
        const tokenList = document.getElementById('token-list');
        tokenList.innerHTML = '';
        tokens.forEach((token, index) => {
            const li = document.createElement('li');
            li.className = token.startsWith('##') ? 'subword-token' : 'full-token';
            const displayInfo = `
                <div class="token-content">
                    <span class="token-text">${token}</span>
                    <span class="token-id">ID: ${tokenIds[index]}</span>
                    ${token.startsWith('##') ? 
                        '<span class="token-type">Subword continuation</span>' : 
                        '<span class="token-type">Full/Start token</span>'}
                </div>
            `;
            li.innerHTML = displayInfo;
            tokenList.appendChild(li);
        });
    }

    function updateTechnicalInfo(data) {
        const info = data.technical_info;

        function createInfoRow(label, value) {
            return `
                <div class="info-detail">
                    <span class="info-label">${label}:</span>
                    <span class="info-value">${value}</span>
                </div>
            `;
        }

        // Model Details
        document.getElementById('model-details').innerHTML = `
            ${createInfoRow('Model', info.model_name)}
            ${createInfoRow('Vocabulary Size', info.vocab_size.toLocaleString())}
            ${createInfoRow('Hidden Layers', info.num_hidden_layers)}
            ${createInfoRow('Attention Heads', info.num_attention_heads)}
        `;

        // Input Analysis
        document.getElementById('input-details').innerHTML = `
            ${createInfoRow('Input Length', info.input_text_length)}
            ${createInfoRow('Tokens', info.num_tokens)}
            ${createInfoRow('Avg. Tokens/Word', (info.num_tokens / info.input_text_length).toFixed(2))}
        `;

        // Embedding Details
        document.getElementById('embedding-details').innerHTML = `
            ${createInfoRow('Original Dimensions', info.embedding_shape[1])}
            ${createInfoRow('Reduced Dimensions', '3D')}
            ${createInfoRow('Perplexity', info.perplexity)}
            ${createInfoRow('Hidden Size', info.hidden_size)}
        `;

        // Attention Details
        document.getElementById('attention-details').innerHTML = `
            ${createInfoRow('Attention Matrix', `${info.num_tokens} × ${info.num_tokens}`)}
            ${createInfoRow('Max Attention', Math.max(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Min Attention', Math.min(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Avg Attention', (data.attention_scores.flat().reduce((a, b) => a + b) / data.attention_scores.flat().length).toFixed(4))}
        `;
    }

    function updateEmbeddingPlot(plotData) {
        try {
            const container = document.getElementById('embedding-plot');
            const containerWidth = container.offsetWidth;
            const width = Math.min(containerWidth, 1000);
            const height = Math.min(width * 0.75, 600);
    
            // Create a deep copy of the plot data to avoid modifying the original
            const data = JSON.parse(JSON.stringify(plotData.data));
            
            // Ensure marker properties are set correctly
            data[0].marker = {
                ...data[0].marker,
                size: 6,
                opacity: 0.8,
                line: {
                    color: 'white',
                    width: 0.5
                }
            };
    
            const layout = {
                width: width,
                height: height,
                scene: {
                    aspectmode: 'cube',
                    camera: {
                        up: {x: 0, y: 0, z: 1},
                        center: {x: 0, y: 0, z: 0},
                        eye: {x: 1.5, y: 1.5, z: 1.5}
                    },
                    xaxis: {
                        title: 'X',
                        range: [-1.5, 1.5]
                    },
                    yaxis: {
                        title: 'Y',
                        range: [-1.5, 1.5]
                    },
                    zaxis: {
                        title: 'Z',
                        range: [-1.5, 1.5]
                    }
                },
                margin: {
                    l: 0,
                    r: 0,
                    t: 30,
                    b: 0,
                    pad: 10
                },
                showlegend: false
            };
    
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'token_embeddings',
                    height: height,
                    width: width,
                    scale: 2
                }
            };
    
            Plotly.newPlot('embedding-plot', data, layout, config);
    
        } catch (error) {
            console.error('Error creating 3D plot:', error);
        }
    }
    
    function createColorScaleSelector(colorscales) {
        // Remove existing selector if it exists
        const existingSelector = document.querySelector('.color-scale-selector');
        if (existingSelector) {
            existingSelector.remove();
        }
    
        // Find the controls container
        const container = document.querySelector('.color-scale-selector');
        if (!container) return;
        
        // Create color scale selector
        container.innerHTML = `
            <label for="colorscale">Color Scale:</label>
            <select id="colorscale">
                ${colorscales.map(scale => `
                    <option value="${scale}" ${scale === 'Viridis' ? 'selected' : ''}>${scale}</option>
                `).join('')}
            </select>
        `;
    
        // Add event listener
        document.getElementById('colorscale').addEventListener('change', function(e) {
            Plotly.restyle('embedding-plot', {
                'marker.colorscale': e.target.value
            });
        });
    }
    
    // Add CSS
    const style = document.createElement('style');
    style.textContent = `
        .viz-container {
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
        }
    
        #embedding-plot {
            width: 100%;
            min-height: 400px;
            background-color: white;
        }
    
        .color-scale-selector {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
    
        .color-scale-selector label {
            margin-right: 10px;
            font-weight: 500;
        }
    
        .color-scale-selector select {
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
    
        /* Add responsive design */
        @media (max-width: 768px) {
            .viz-container {
                padding: 10px;
            }
            
            #embedding-plot {
                min-height: 300px;
            }
        }
    `;
    document.head.appendChild(style);
    
    
    

    function updateAttentionHeatmap(attentionScores, tokens) {
        // Get container width
        const container = document.getElementById('attention-heatmap');
        const containerWidth = container.offsetWidth;
        
        // Calculate responsive dimensions
        const maxSize = Math.min(containerWidth, 800); // Limit maximum size
        const fontSize = Math.max(8, Math.min(12, 200 / tokens.length));
        
        const data = [{
            z: attentionScores,
            x: tokens,
            y: tokens,
            type: 'heatmap',
            colorscale: [
                [0, 'rgb(0,0,100)'],
                [0.5, 'rgb(0,255,255)'],
                [1, 'rgb(255,255,0)']
            ],
            hoverongaps: false,
            hoverinfo: 'all',
            hovertemplate: 
                'Source: %{y}<br>' +
                'Target: %{x}<br>' +
                'Attention: %{z:.4f}<br>' +
                '<extra></extra>',
        }];
    
        const layout = {
            title: {
                text: 'Attention Scores<br><sub>Hover over cells to see attention values</sub>',
                font: { size: 16 },
                y: 0.95
            },
            xaxis: {
                title: 'Target Tokens',
                side: 'bottom',
                tickfont: { size: fontSize },
                tickangle: -45
            },
            yaxis: {
                title: 'Source Tokens',
                tickfont: { size: fontSize }
            },
            width: maxSize,
            height: maxSize,
            margin: {
                l: 80,  // Reduced left margin
                r: 20,  // Reduced right margin
                t: 60,  // Reduced top margin
                b: 80   // Reduced bottom margin
            },
            annotations: [],
            coloraxis: {
                colorbar: {
                    title: 'Attention Score',
                    titleside: 'right',
                    thickness: 15,
                    len: 0.5,
                    x: -0.15  // Move colorbar to the left
                }
            }
        };
    
        // Add annotations for strong attention values
        const threshold = 0.15;
        attentionScores.forEach((row, i) => {
            row.forEach((score, j) => {
                if (score > threshold) {
                    layout.annotations.push({
                        x: j,
                        y: i,
                        text: score.toFixed(2),
                        showarrow: false,
                        font: {
                            color: score > 0.3 ? 'white' : 'black',
                            size: fontSize
                        }
                    });
                }
            });
        });
    
        Plotly.newPlot('attention-heatmap', data, layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'attention_heatmap',
                height: maxSize,
                width: maxSize,
                scale: 2
            }
        });
    }
    

    // Add CSS styles
    
});

// Update Plotly CDN warning
window.addEventListener('load', function() {
    console.log('Note: Using Plotly version 1.58.5. This is expected and won\'t affect functionality.');
});
