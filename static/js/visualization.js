document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('process-btn');
    const inputText = document.getElementById('input-text');
    
    processBtn.addEventListener('click', async function() {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        
        try {
            const text = inputText.value.trim();
            if (text.length === 0) {
                alert('Please enter some text');
                return;
            }
            
            console.log('Processing text:', text);
            
            const response = await fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text}),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Received data:', data);

            if (data.error) {
                throw new Error(data.error);
            }

            // Update all visualizations
            updateTokenList(data.tokens, data.token_ids);
            updateTechnicalInfo(data);
            updateEmbeddingPlot(data.plot_data);
            updateAttentionHeatmap(data.attention_scores, data.tokens);
            
        } catch (error) {
            console.error('Error details:', error);
            alert(`Error: ${error.message}`);
        } finally {
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
        }
    });

    function updateTokenList(tokens, tokenIds) {
        const tokenList = document.getElementById('token-list');
        tokenList.innerHTML = '';
        tokens.forEach((token, index) => {
            const li = document.createElement('li');
            li.textContent = `${token} (ID: ${tokenIds[index]})`;
            tokenList.appendChild(li);
        });
    }

    function updateTechnicalInfo(data) {
        const modelDetails = document.getElementById('model-details');
        const inputDetails = document.getElementById('input-details');
        const embeddingDetails = document.getElementById('embedding-details');
        const attentionDetails = document.getElementById('attention-details');

        function createInfoRow(label, value) {
            return `
                <div class="info-detail">
                    <span class="info-label">${label}:</span>
                    <span class="info-value">${value}</span>
                </div>
            `;
        }

        modelDetails.innerHTML = `
            ${createInfoRow('Model', data.technical_info.model_name)}
            ${createInfoRow('Vocabulary Size', data.technical_info.vocab_size.toLocaleString())}
            ${createInfoRow('Hidden Layers', data.technical_info.num_hidden_layers)}
            ${createInfoRow('Attention Heads', data.technical_info.num_attention_heads)}
        `;

        inputDetails.innerHTML = `
            ${createInfoRow('Input Length', data.technical_info.input_text_length)}
            ${createInfoRow('Tokens', data.technical_info.num_tokens)}
            ${createInfoRow('Avg. Tokens/Word', (data.technical_info.num_tokens / data.technical_info.input_text_length).toFixed(2))}
        `;

        embeddingDetails.innerHTML = `
            ${createInfoRow('Original Dimensions', `${data.technical_info.embedding_shape[1]}`)}
            ${createInfoRow('Reduced Dimensions', '3D')}
            ${createInfoRow('Perplexity', data.technical_info.perplexity)}
            ${createInfoRow('Hidden Size', data.technical_info.hidden_size)}
        `;

        const attentionShape = `${data.attention_scores.length} × ${data.attention_scores[0].length}`;
        attentionDetails.innerHTML = `
            ${createInfoRow('Attention Matrix', attentionShape)}
            ${createInfoRow('Max Attention', Math.max(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Min Attention', Math.min(...data.attention_scores.flat()).toFixed(4))}
            ${createInfoRow('Avg Attention', (data.attention_scores.flat().reduce((a, b) => a + b) / data.attention_scores.flat().length).toFixed(4))}
        `;
    }

    function updateEmbeddingPlot(plotData) {
        Plotly.newPlot('embedding-plot', plotData.data, plotData.layout);
    }

    function updateAttentionHeatmap(attentionScores, tokens) {
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

        const size = Math.max(500, tokens.length * 40);

        const layout = {
            title: {
                text: 'Attention Scores<br><sub>Hover over cells to see attention values</sub>',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Target Tokens',
                side: 'bottom',
                tickfont: { size: fontSize }
            },
            yaxis: {
                title: 'Source Tokens',
                tickfont: { size: fontSize }
            },
            width: size,
            height: size,
            annotations: []
        };

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
                            color: 'white',
                            size: fontSize
                        }
                    });
                }
            });
        });

        Plotly.newPlot('attention-heatmap', data, layout);
    }
});
