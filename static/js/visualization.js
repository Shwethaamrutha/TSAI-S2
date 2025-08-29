document.addEventListener('DOMContentLoaded', function() {
    const processBtn = document.getElementById('process-btn');
    const inputText = document.getElementById('input-text');
    
    // Verify 3D plot container exists
    const embeddingPlot = document.getElementById('embedding-plot');
    if (!embeddingPlot) {
        console.error('3D plot container not found!');
    } else {
        console.log('3D plot container found and ready');
    }
    
    // Verify Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded!');
    } else {
        console.log('Plotly is loaded successfully, version:', Plotly.version);
    }
    
    // Add event listener for text labels toggle
    const textLabelsCheckbox = document.getElementById('show-text-labels');
    if (textLabelsCheckbox) {
        textLabelsCheckbox.addEventListener('change', function() {
            const showLabels = this.checked;
            const plotElement = document.getElementById('embedding-plot');
            
            if (plotElement && plotElement.data) {
                Plotly.restyle('embedding-plot', {
                    'mode': showLabels ? 'markers+text' : 'markers'
                });
            }
        });
    }
    
    // Add event listener for attention values toggle
    const attentionValuesCheckbox = document.getElementById('show-attention-values');
    if (attentionValuesCheckbox) {
        attentionValuesCheckbox.addEventListener('change', function() {
            // Recreate the attention heatmap with updated annotation settings
            if (processData && processData.attention_scores && processData.tokens) {
                updateAttentionHeatmap(processData.attention_scores, processData.tokens);
            }
        });
    }
    
    // Download functionality
    const downloadEmbeddingsBtn = document.getElementById('download-embeddings');
    const downloadEmbeddingsJsonBtn = document.getElementById('download-embeddings-json');
    
    if (downloadEmbeddingsBtn) {
        downloadEmbeddingsBtn.addEventListener('click', function() {
            if (processData && processData.embeddings && processData.tokens) {
                downloadEmbeddingsCSV(processData.tokens, processData.embeddings);
            } else {
                alert('No embedding data available. Please process some text first.');
            }
        });
    }
    
    if (downloadEmbeddingsJsonBtn) {
        downloadEmbeddingsJsonBtn.addEventListener('click', function() {
            if (processData && processData.embeddings && processData.tokens) {
                downloadEmbeddingsJSON(processData.tokens, processData.embeddings, processData);
            } else {
                alert('No embedding data available. Please process some text first.');
            }
        });
    }
    
    function downloadEmbeddingsCSV(tokens, embeddings) {
        // Create CSV content
        let csvContent = 'Token,Position,X,Y,Z\n';
        
        tokens.forEach((token, index) => {
            const embedding = embeddings[index];
            if (embedding && embedding.length >= 3) {
                csvContent += `"${token}",${index},${embedding[0]},${embedding[1]},${embedding[2]}\n`;
            }
        });
        
        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'token_embeddings_3d.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    function downloadEmbeddingsJSON(tokens, embeddings, fullData) {
        // Create comprehensive JSON data
        const jsonData = {
            metadata: {
                timestamp: new Date().toISOString(),
                model: fullData.technical_info?.model_name || 'bert-base-uncased',
                total_tokens: tokens.length,
                original_dimensions: fullData.technical_info?.embedding_shape?.[1] || 'unknown',
                reduced_dimensions: 3
            },
            embeddings: tokens.map((token, index) => ({
                token: token,
                position: index,
                token_id: fullData.token_ids?.[index] || null,
                embedding_3d: embeddings[index] || [],
                embedding_full: fullData.original_embeddings?.[index] || null
            }))
        };
        
        // Create and download file
        const jsonContent = JSON.stringify(jsonData, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'token_embeddings_complete.json');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    // Tokenization process step controls
    let currentStep = 1;
    const totalSteps = 4;
    let processData = null;
    
    const prevBtn = document.getElementById('prev-step');
    const nextBtn = document.getElementById('next-step');
    const currentStepSpan = document.getElementById('current-step');
    const totalStepsSpan = document.getElementById('total-steps');
    
    if (prevBtn && nextBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 1) {
                currentStep--;
                updateStepDisplay();
            }
        });
        
        nextBtn.addEventListener('click', () => {
            if (currentStep < totalSteps) {
                currentStep++;
                updateStepDisplay();
            }
        });
    }
    
    function updateStepDisplay() {
        // Update step indicator
        currentStepSpan.textContent = currentStep;
        totalStepsSpan.textContent = totalSteps;
        
        // Update button states
        prevBtn.disabled = currentStep === 1;
        nextBtn.disabled = currentStep === totalSteps;
        
        // Hide all step content
        for (let i = 1; i <= totalSteps; i++) {
            const stepContent = document.getElementById(`step-${i}`);
            if (stepContent) {
                stepContent.style.display = 'none';
            }
        }
        
        // Show current step content
        const currentStepContent = document.getElementById(`step-${currentStep}`);
        if (currentStepContent) {
            currentStepContent.style.display = 'block';
        }
        
        // Update step content based on current step
        if (processData) {
            updateStepContent(currentStep, processData);
        }
    }
    
    function updateStepContent(step, data) {
        switch(step) {
            case 1:
                updateStep1(data);
                break;
            case 2:
                updateStep2(data);
                break;
            case 3:
                updateStep3(data);
                break;
            case 4:
                updateStep4(data);
                break;
        }
    }
    
    function updateStep1(data) {
        const textBreakdown = document.querySelector('#step-1 .text-breakdown');
        if (textBreakdown) {
            textBreakdown.innerHTML = `
                <div class="text-analysis">
                    <h4>Original Input Text:</h4>
                    <div class="original-text">${data.original_text || 'Text not available'}</div>
                    <div class="text-stats">
                        <p><strong>Character Count:</strong> ${data.original_text ? data.original_text.length : 0}</p>
                        <p><strong>Word Count:</strong> ${data.original_text ? data.original_text.split(' ').length : 0}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep2(data) {
        const tokenBreakdown = document.querySelector('#step-2 .token-breakdown');
        if (tokenBreakdown && data.tokens) {
            const tokenHtml = data.tokens.map((token, index) => {
                const isSubword = token.startsWith('##');
                const isSpecial = token === '[CLS]' || token === '[SEP]';
                let tokenClass = 'token-item';
                if (isSubword) tokenClass += ' subword-token';
                else if (isSpecial) tokenClass += ' special-token';
                else tokenClass += ' full-token';
                
                return `
                    <div class="${tokenClass}">
                        <span class="token-text">${token}</span>
                        <span class="token-type">${isSubword ? 'Subword' : isSpecial ? 'Special' : 'Full Token'}</span>
                    </div>
                `;
            }).join('');
            
            tokenBreakdown.innerHTML = `
                <div class="tokenization-results">
                    <h4>Tokenization Results:</h4>
                    <div class="token-list-display">${tokenHtml}</div>
                    <div class="token-stats">
                        <p><strong>Total Tokens:</strong> ${data.tokens.length}</p>
                        <p><strong>Full Tokens:</strong> ${data.tokens.filter(t => !t.startsWith('##') && t !== '[CLS]' && t !== '[SEP]').length}</p>
                        <p><strong>Subword Tokens:</strong> ${data.tokens.filter(t => t.startsWith('##')).length}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep3(data) {
        const tokenIdsBreakdown = document.querySelector('#step-3 .token-ids-breakdown');
        if (tokenIdsBreakdown && data.tokens && data.token_ids) {
            const tokenIdsHtml = data.tokens.map((token, index) => {
                const tokenId = data.token_ids[index];
                return `
                    <div class="token-id-item">
                        <span class="token-text">${token}</span>
                        <span class="token-id">ID: ${tokenId}</span>
                    </div>
                `;
            }).join('');
            
            tokenIdsBreakdown.innerHTML = `
                <div class="token-ids-results">
                    <h4>Token to ID Mapping:</h4>
                    <div class="token-ids-display">${tokenIdsHtml}</div>
                    <div class="token-ids-stats">
                        <p><strong>Vocabulary Size:</strong> ${data.technical_info ? data.technical_info.vocab_size.toLocaleString() : 'N/A'}</p>
                        <p><strong>ID Range:</strong> ${Math.min(...data.token_ids)} - ${Math.max(...data.token_ids)}</p>
                    </div>
                </div>
            `;
        }
    }
    
    function updateStep4(data) {
        const embeddingBreakdown = document.querySelector('#step-4 .embedding-breakdown');
        if (embeddingBreakdown && data.embeddings) {
            const sampleEmbeddings = data.embeddings.slice(0, 5); // Show first 5 tokens
            const embeddingHtml = sampleEmbeddings.map((embedding, index) => {
                const token = data.tokens[index];
                const vectorPreview = embedding.slice(0, 10).map(val => val.toFixed(3)).join(', ');
                return `
                    <div class="embedding-item">
                        <span class="token-text">${token}</span>
                        <span class="vector-preview">[${vectorPreview}...]</span>
                        <span class="vector-dim">${embedding.length} dimensions</span>
                    </div>
                `;
            }).join('');
            
            embeddingBreakdown.innerHTML = `
                <div class="embedding-results">
                    <h4>Vector Embeddings (Sample):</h4>
                    <div class="embedding-display">${embeddingHtml}</div>
                    <div class="embedding-stats">
                        <p><strong>Original Dimensions:</strong> ${data.technical_info ? data.technical_info.embedding_shape[1] : 'N/A'}</p>
                        <p><strong>Reduced to 3D:</strong> For visualization</p>
                        <p><strong>Total Embeddings:</strong> ${data.embeddings.length}</p>
                    </div>
                </div>
            `;
        }
    }
    
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
        
        console.log('Sending request to /process with text:', text);
        
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
            console.log('Plot data structure:', data.plot_data);
            console.log('Plot data keys:', Object.keys(data.plot_data || {}));
            console.log('Plot data.data:', data.plot_data?.data);
            console.log('Plot data.data[0]:', data.plot_data?.data?.[0]);
            
            // Store data for step-by-step process
            processData = {
                ...data,
                original_text: inputText.value.trim()
            };
            
            // Update all visualizations
            updateTokenList(data.tokens, data.token_ids);
            updateTechnicalInfo(data);  // Pass the entire data object
            updateEmbeddingPlot(data.plot_data);
            updateAttentionHeatmap(data.attention_scores, data.tokens);
            
            // Initialize step-by-step process
            currentStep = 1;
            updateStepDisplay();
        })
        .catch(error => {
            console.error('Error details:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            
            // More specific error handling
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                alert('Network error: Unable to connect to the server. Please check if the Flask application is running.');
            } else {
                alert(`Error: ${error.message}`);
            }
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
            
            // Clear any existing content
            container.innerHTML = '';
            
            const containerWidth = container.offsetWidth;
            // Improved dimensions for better visualization
            const width = Math.min(containerWidth, 1200);  // Increased max width
            const height = Math.min(width * 0.8, 800);     // Increased height ratio and max height
    
            // Validate plot data structure
            if (!plotData || !plotData.data || !Array.isArray(plotData.data) || plotData.data.length === 0) {
                console.error('Invalid plot data structure:', plotData);
                container.innerHTML = '<p style="color: red; text-align: center; padding: 20px;">Error: Invalid plot data structure</p>';
                return;
            }
            
            // Create a deep copy of the plot data to avoid modifying the original
            const data = JSON.parse(JSON.stringify(plotData.data));
            
            // Test with simple data if the original data seems problematic
            if (!data[0].x || !data[0].y || !data[0].z || data[0].x.length === 0) {
                console.warn('Data appears to be empty, creating test plot');
                data[0] = {
                    type: 'scatter3d',
                    x: [0, 1, -1, 0.5, -0.5],
                    y: [0, 0.5, -0.5, 1, -1],
                    z: [0, 0.3, -0.3, 0.7, -0.7],
                    mode: 'markers',
                    text: ['test1', 'test2', 'test3', 'test4', 'test5'],
                    marker: {
                        size: 10,
                        color: [0, 1, 2, 3, 4],
                        colorscale: 'Viridis',
                        opacity: 0.9
                    }
                };
            }
            
            // Ensure all required properties are present
            if (!data[0].type) data[0].type = 'scatter3d';
            if (!data[0].mode) data[0].mode = 'markers';
            if (!data[0].marker) data[0].marker = { size: 8, color: 'blue' };
            
            console.log('Original data[0]:', data[0]);
            console.log('Data[0] keys:', Object.keys(data[0]));
            console.log('X coordinates:', data[0].x);
            console.log('Y coordinates:', data[0].y);
            console.log('Z coordinates:', data[0].z);
            console.log('Text values:', data[0].text);
            
            // Ensure marker properties are set correctly for better visibility
            data[0].marker = {
                ...data[0].marker,
                size: 8,  // Increased size for better visibility
                opacity: 0.9,  // Increased opacity
                line: {
                    color: 'black',
                    width: 1
                },
                symbol: 'circle'  // Ensure consistent symbol
            };
            
            // Add proper hover template
            data[0].hovertemplate = 
                '<b>Token:</b> %{text}<br>' +
                '<b>Position:</b> %{marker.color}<br>' +
                '<b>X:</b> %{x:.3f}<br>' +
                '<b>Y:</b> %{y:.3f}<br>' +
                '<b>Z:</b> %{z:.3f}<br>' +
                '<extra></extra>';
            
            // Handle text labels based on checkbox
            const showTextLabels = document.getElementById('show-text-labels').checked;
            if (showTextLabels) {
                data[0].mode = 'markers+text';
                if (!data[0].textposition) {
                    data[0].textposition = 'top center';
                }
                if (!data[0].textfont) {
                    data[0].textfont = {
                        'size': 8,
                        'color': 'black'
                    };
                }
            } else {
                data[0].mode = 'markers';
            }
    
            // Use the layout from the backend data if available, otherwise create a default one
            let layout;
            if (plotData.layout) {
                layout = {
                    ...plotData.layout,
                    width: width,
                    height: height
                };
            } else {
                layout = {
                    width: width,
                    height: height,
                    title: {
                        text: '3D Token Embeddings',
                        font: { size: 16 }
                    },
                    scene: {
                        aspectmode: 'cube',
                        camera: {
                            up: {x: 0, y: 0, z: 1},
                            center: {x: 0, y: 0, z: 0},
                            eye: {x: 1.5, y: 1.5, z: 1.5}
                        },
                        xaxis: {
                            title: 'X',
                            showgrid: true,
                            zeroline: true
                        },
                        yaxis: {
                            title: 'Y',
                            showgrid: true,
                            zeroline: true
                        },
                        zaxis: {
                            title: 'Z',
                            showgrid: true,
                            zeroline: true
                        }
                    },
                    margin: {
                        l: 0,
                        r: 0,
                        t: 50,
                        b: 0,
                        pad: 10
                    },
                    showlegend: false
                };
            }
    
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
    
            console.log('About to create plot with data:', data);
            console.log('Layout:', layout);
            console.log('Config:', config);
            
            // Try to create the plot
            try {
                console.log('Final data structure:', data);
                console.log('Final layout structure:', layout);
                
                Plotly.newPlot('embedding-plot', data, layout, config).then(function() {
                    console.log('Plot created successfully');
                }).catch(function(error) {
                    console.error('Error creating plot:', error);
                    // Fallback to simple test plot
                    console.log('Trying fallback test plot...');
                    const fallbackData = [{
                        type: 'scatter3d',
                        x: [0, 1, -1],
                        y: [0, 1, -1],
                        z: [0, 1, -1],
                        mode: 'markers',
                        marker: {
                            size: 10,
                            color: 'red'
                        }
                    }];
                    const fallbackLayout = {
                        scene: {
                            xaxis: {title: 'X'},
                            yaxis: {title: 'Y'},
                            zaxis: {title: 'Z'}
                        }
                    };
                    Plotly.newPlot('embedding-plot', fallbackData, fallbackLayout, config);
                });
            } catch (error) {
                console.error('Exception during plot creation:', error);
            }
            
            // Add event listener for point selection
            container.on('plotly_click', function(data) {
                const point = data.points[0];
                console.log('Clicked token:', point.text);
                console.log('Position:', point.marker.color);
                console.log('Coordinates:', [point.x, point.y, point.z]);
            });
    
        } catch (error) {
            console.error('Error creating 3D plot:', error);
            console.error('Plot data:', plotData);
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
        
        // Improve color scale for better attention visualization
        const data = [{
            z: attentionScores,
            x: tokens,
            y: tokens,
            type: 'heatmap',
            colorscale: [
                [0, 'rgb(0,0,100)'],      // Dark blue for low attention
                [0.2, 'rgb(0,100,200)'],   // Blue for moderate attention
                [0.4, 'rgb(0,200,255)'],   // Light blue for higher attention
                [0.6, 'rgb(100,255,100)'], // Green for high attention
                [0.8, 'rgb(255,255,0)'],   // Yellow for very high attention
                [1, 'rgb(255,200,0)']      // Orange for highest attention
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
    
        // Add annotations for strong attention values with improved positioning
        const showAttentionValues = document.getElementById('show-attention-values')?.checked ?? true;
        
        if (showAttentionValues) {
            const threshold = 0.1; // Lower threshold to show more attention patterns
            const maxScore = Math.max(...attentionScores.flat());
            const minScore = Math.min(...attentionScores.flat());
            
            attentionScores.forEach((row, i) => {
                row.forEach((score, j) => {
                    // Show self-attention (diagonal) and strong cross-attention
                    if (score > threshold || (i === j && score > maxScore * 0.3)) {
                        layout.annotations.push({
                            x: j,
                            y: i,
                            text: score.toFixed(3),
                            showarrow: false,
                            xanchor: 'center',
                            yanchor: 'middle',
                            font: {
                                color: score > maxScore * 0.6 ? 'white' : 'black',
                                size: Math.max(6, Math.min(fontSize, 10)) // Limit font size
                            }
                        });
                    }
                });
            });
        }
    
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
