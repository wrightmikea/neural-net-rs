// Neural Network Training Platform - Application Logic
// Minimal JavaScript for bootstrapping WASM and DOM manipulation
// All neural network logic implemented in Rust/WASM

import init, {
    NeuralNetwork,
    listExamples,
    getExampleInfo
} from './wasm/neural_net_wasm.js';

// Application state
let wasmModule;
let currentNetwork = null;
let currentExampleInfo = null;
let trainingData = null;
let lossHistory = [];
let trainingStartTime = null;
let trainingInterval = null;
let eventSource = null;

// DOM elements
const elements = {
    exampleSelect: document.getElementById('example-select'),
    exampleDescription: document.getElementById('example-description'),
    epochsInput: document.getElementById('epochs-input'),
    learningRateInput: document.getElementById('learning-rate-input'),
    trainButton: document.getElementById('train-button'),
    stopButton: document.getElementById('stop-button'),
    evaluateButton: document.getElementById('evaluate-button'),
    trainingStatus: document.querySelector('.status-text'),
    progressContainer: document.getElementById('progress-container'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    currentEpoch: document.getElementById('current-epoch'),
    currentLoss: document.getElementById('current-loss'),
    trainingTime: document.getElementById('training-time'),
    input1: document.getElementById('input1'),
    input2: document.getElementById('input2'),
    outputDisplay: document.getElementById('output-display'),
    architectureDisplay: document.getElementById('architecture-display'),
    truthTableBody: document.getElementById('truth-table-body'),
    lossChart: document.getElementById('loss-chart'),
};

// Chart context
const chartCtx = elements.lossChart.getContext('2d');
let chartWidth = elements.lossChart.width;
let chartHeight = elements.lossChart.height;

// Initialize application
async function initApp() {
    try {
        // Initialize WASM module
        wasmModule = await init();
        console.log('WASM module loaded successfully');

        // Load examples
        await loadExamples();

        // Setup event listeners
        setupEventListeners();

        // Update UI
        updateExampleInfo();
        updateStatus('Ready to train', 'success');

    } catch (error) {
        console.error('Failed to initialize app:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

// Load and populate examples
async function loadExamples() {
    try {
        const examples = listExamples();
        elements.exampleSelect.innerHTML = '';

        // Cache example data (we don't have access to training data in WASM)
        // Truth table will work best for simple examples
        window.exampleCache = {};

        examples.forEach(ex => {
            const option = document.createElement('option');
            option.value = ex.name;
            option.textContent = `${ex.name.toUpperCase()} Gate (${ex.architecture.join('-')})`;
            elements.exampleSelect.appendChild(option);
        });

    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    elements.exampleSelect.addEventListener('change', updateExampleInfo);
    elements.trainButton.addEventListener('click', startTraining);
    elements.stopButton.addEventListener('click', stopTraining);
    elements.evaluateButton.addEventListener('click', evaluateNetwork);
    elements.input1.addEventListener('input', evaluateNetwork);
    elements.input2.addEventListener('input', evaluateNetwork);
}

// Update example information
function updateExampleInfo() {
    try {
        const exampleName = elements.exampleSelect.value;
        const info = getExampleInfo(exampleName);
        currentExampleInfo = info;

        elements.exampleDescription.textContent = `${info.description} | Architecture: [${info.architecture.join(' → ')}]`;

        // Display architecture
        displayArchitecture(info.architecture);

        // Update input/output UI
        updateTestingUI(info.architecture);

    } catch (error) {
        console.error('Failed to update example info:', error);
    }
}

// Display network architecture
function displayArchitecture(layers) {
    const layerNames = ['Input', ...Array(layers.length - 2).fill('Hidden'), 'Output'];

    elements.architectureDisplay.innerHTML = layers.map((size, idx) => `
        <div class="architecture-layer">
            ${layerNames[idx]}<br>
            <small>${size} neuron${size > 1 ? 's' : ''}</small>
        </div>
        ${idx < layers.length - 1 ? '<span class="architecture-arrow">→</span>' : ''}
    `).join('');
}

// Update testing UI based on architecture
function updateTestingUI(architecture) {
    const inputSize = architecture[0];
    const outputSize = architecture[architecture.length - 1];

    const testGrid = document.querySelector('.test-grid');
    testGrid.innerHTML = '';

    // Create input fields
    for (let i = 0; i < inputSize; i++) {
        const inputDiv = document.createElement('div');
        inputDiv.className = 'form-group';
        inputDiv.innerHTML = `
            <label for="input${i}">Input ${i + 1}</label>
            <input type="number" id="input${i}" value="0.0" min="-1" max="1" step="0.1">
        `;
        testGrid.appendChild(inputDiv);
    }

    // Add evaluate button
    const buttonDiv = document.createElement('div');
    buttonDiv.className = 'form-group';
    buttonDiv.innerHTML = `<button id="evaluate-button" class="btn btn-primary" disabled>Evaluate</button>`;
    testGrid.appendChild(buttonDiv);

    // Add output display
    const outputDiv = document.createElement('div');
    outputDiv.className = 'form-group';
    outputDiv.innerHTML = `
        <label>Output</label>
        <div id="output-display" class="output-display">N/A</div>
    `;
    testGrid.appendChild(outputDiv);

    // Re-attach event listeners
    document.getElementById('evaluate-button').addEventListener('click', evaluateNetwork);
    for (let i = 0; i < inputSize; i++) {
        document.getElementById(`input${i}`).addEventListener('input', evaluateNetwork);
    }

    // Update elements reference
    elements.evaluateButton = document.getElementById('evaluate-button');
    elements.outputDisplay = document.getElementById('output-display');
}

// Start training
async function startTraining() {
    const mode = document.querySelector('input[name="mode"]:checked').value;

    if (mode === 'wasm') {
        await startWasmTraining();
    } else {
        await startApiTraining();
    }
}

// Start WASM training (local)
async function startWasmTraining() {
    try {
        const exampleName = elements.exampleSelect.value;
        const epochs = parseInt(elements.epochsInput.value);
        const learningRate = parseFloat(elements.learningRateInput.value);

        // Reset state
        lossHistory = [];
        clearChart();

        // Create network
        currentNetwork = NeuralNetwork.fromExample(exampleName, learningRate);

        // Update UI
        updateStatus('Training locally with WASM...', 'training');
        elements.trainButton.disabled = true;
        elements.stopButton.disabled = false;
        elements.evaluateButton.disabled = true;
        elements.progressContainer.style.display = 'block';
        trainingStartTime = Date.now();

        // Update timer
        trainingInterval = setInterval(updateTrainingTime, 100);

        // Create progress callback for WASM training
        const progressCallback = (epoch, loss) => {
            updateTrainingProgress(epoch, loss, epochs);
        };

        // Train network (blocking - runs in main thread)
        // Note: In production, this should run in a Web Worker
        await currentNetwork.train(exampleName, epochs, progressCallback);

        // Training complete
        completeTraining();

    } catch (error) {
        console.error('Training failed:', error);
        updateStatus(`Training failed: ${error.message}`, 'error');
        stopTraining();
    }
}

// Start API training (remote with SSE)
async function startApiTraining() {
    try {
        const exampleName = elements.exampleSelect.value;
        const epochs = parseInt(elements.epochsInput.value);
        const learningRate = parseFloat(elements.learningRateInput.value);

        // Reset state
        lossHistory = [];
        clearChart();

        // Update UI
        updateStatus('Connecting to training server...', 'training');
        elements.trainButton.disabled = true;
        elements.stopButton.disabled = false;
        elements.evaluateButton.disabled = true;
        elements.progressContainer.style.display = 'block';
        trainingStartTime = Date.now();

        // Update timer
        trainingInterval = setInterval(updateTrainingTime, 100);

        // Connect to SSE endpoint
        const response = await fetch('/api/train/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                example: exampleName,
                epochs: epochs,
                learning_rate: learningRate
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        // Setup SSE reader
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        updateStatus('Training on server (live updates)...', 'training');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));
                    updateTrainingProgress(data.epoch, data.loss, epochs);
                }
            }
        }

        // Training complete
        updateStatus('Training completed on server!', 'success');
        completeTraining();

    } catch (error) {
        console.error('API training failed:', error);
        updateStatus(`API training failed: ${error.message}`, 'error');
        stopTraining();
    }
}

// Update training progress
function updateTrainingProgress(epoch, loss, totalEpochs) {
    elements.currentEpoch.textContent = epoch;
    elements.currentLoss.textContent = loss.toFixed(6);

    const progress = (epoch / totalEpochs) * 100;
    elements.progressFill.style.width = `${progress}%`;
    elements.progressText.textContent = `${Math.round(progress)}%`;

    // Add to loss history
    lossHistory.push({ epoch, loss });

    // Update chart every 10 epochs or at the end
    if (epoch % 10 === 0 || epoch === totalEpochs) {
        drawChart();
    }
}

// Complete training
function completeTraining() {
    updateStatus('Training completed!', 'success');
    elements.trainButton.disabled = false;
    elements.stopButton.disabled = true;
    elements.evaluateButton.disabled = false;

    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }

    // Update truth table
    updateTruthTable();
}

// Stop training
function stopTraining() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }

    updateStatus('Training stopped', 'warning');
    elements.trainButton.disabled = false;
    elements.stopButton.disabled = true;
    elements.progressContainer.style.display = 'none';
}

// Update training time
function updateTrainingTime() {
    if (trainingStartTime) {
        const elapsed = (Date.now() - trainingStartTime) / 1000;
        elements.trainingTime.textContent = `${elapsed.toFixed(1)}s`;
    }
}

// Evaluate network
function evaluateNetwork() {
    if (!currentNetwork || !currentExampleInfo) {
        elements.outputDisplay.textContent = 'Train first';
        return;
    }

    try {
        const inputSize = currentExampleInfo.architecture[0];
        const outputSize = currentExampleInfo.architecture[currentExampleInfo.architecture.length - 1];

        // Collect all input values
        const inputs = [];
        for (let i = 0; i < inputSize; i++) {
            const inputElem = document.getElementById(`input${i}`);
            if (inputElem) {
                inputs.push(parseFloat(inputElem.value));
            }
        }

        const outputs = currentNetwork.evaluate(inputs);

        // Display output based on size
        if (outputSize === 1) {
            // Single output - show the value
            elements.outputDisplay.textContent = outputs[0].toFixed(4);
        } else {
            // Multiple outputs - show predicted class (argmax)
            const maxIdx = outputs.indexOf(Math.max(...outputs));
            elements.outputDisplay.textContent = `Class ${maxIdx + 1} (${outputs[maxIdx].toFixed(3)})`;
        }

    } catch (error) {
        console.error('Evaluation failed:', error);
        elements.outputDisplay.textContent = 'Error';
    }
}

// Update truth table
function updateTruthTable() {
    if (!currentNetwork || !currentExampleInfo) return;

    const inputSize = currentExampleInfo.architecture[0];
    const outputSize = currentExampleInfo.architecture[currentExampleInfo.architecture.length - 1];

    // Hide truth table for complex examples (>3 inputs or >4 outputs)
    const truthTableSection = document.getElementById('truth-table');
    if (inputSize > 3 || outputSize > 4) {
        truthTableSection.style.display = 'none';
        return;
    }
    truthTableSection.style.display = 'block';

    // Generate all binary combinations for inputs
    const numCombinations = Math.pow(2, inputSize);
    const testInputs = [];
    for (let i = 0; i < numCombinations; i++) {
        const input = [];
        for (let j = inputSize - 1; j >= 0; j--) {
            input.push((i >> j) & 1 ? 1.0 : 0.0);
        }
        testInputs.push(input);
    }

    // Build table header
    let headerHtml = '<tr>';
    for (let i = 0; i < inputSize; i++) {
        headerHtml += `<th>In ${i + 1}</th>`;
    }
    if (outputSize === 1) {
        headerHtml += '<th>Expected</th><th>Predicted</th><th>Error</th>';
    } else {
        headerHtml += '<th>Expected Class</th><th>Predicted Class</th><th>Confidence</th>';
    }
    headerHtml += '</tr>';

    const thead = elements.truthTableBody.closest('table').querySelector('thead');
    thead.innerHTML = headerHtml;

    // Get example to find expected outputs
    const exampleName = elements.exampleSelect.value;
    const example = window.exampleCache?.[exampleName];

    elements.truthTableBody.innerHTML = testInputs.map((input) => {
        const outputs = currentNetwork.evaluate(input);

        let row = '<tr>';
        // Input columns
        for (let val of input) {
            row += `<td>${val.toFixed(1)}</td>`;
        }

        if (outputSize === 1) {
            // Single output - show value and error
            const predicted = outputs[0];
            // Find expected value if we have training data
            let expected = 0;
            if (example && example.inputs) {
                const matchIdx = example.inputs.findIndex(inp =>
                    inp.every((v, i) => Math.abs(v - input[i]) < 0.01)
                );
                if (matchIdx >= 0 && example.targets[matchIdx]) {
                    expected = example.targets[matchIdx][0];
                }
            }
            const error = Math.abs(predicted - expected);
            const errorClass = error < 0.1 ? 'error-low' : 'error-high';
            row += `<td>${expected.toFixed(1)}</td>`;
            row += `<td>${predicted.toFixed(4)}</td>`;
            row += `<td class="${errorClass}">${error.toFixed(4)}</td>`;
        } else {
            // Multi-output - show predicted class
            const maxIdx = outputs.indexOf(Math.max(...outputs));
            const confidence = outputs[maxIdx];
            // Find expected class
            let expectedClass = 0;
            if (example && example.inputs) {
                const matchIdx = example.inputs.findIndex(inp =>
                    inp.every((v, i) => Math.abs(v - input[i]) < 0.01)
                );
                if (matchIdx >= 0 && example.targets[matchIdx]) {
                    expectedClass = example.targets[matchIdx].indexOf(1.0) + 1;
                }
            }
            const correct = (maxIdx + 1) === expectedClass;
            const classColor = correct ? 'error-low' : 'error-high';
            row += `<td>${expectedClass}</td>`;
            row += `<td class="${classColor}">${maxIdx + 1}</td>`;
            row += `<td>${confidence.toFixed(3)}</td>`;
        }

        row += '</tr>';
        return row;
    }).join('');
}

// Draw loss chart
function drawChart() {
    // Clear canvas
    chartCtx.clearRect(0, 0, chartWidth, chartHeight);

    if (lossHistory.length === 0) return;

    // Calculate scales
    const padding = 40;
    const graphWidth = chartWidth - 2 * padding;
    const graphHeight = chartHeight - 2 * padding;

    const maxLoss = Math.max(...lossHistory.map(h => h.loss));
    const maxEpoch = Math.max(...lossHistory.map(h => h.epoch));

    // Draw axes
    chartCtx.strokeStyle = '#666';
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    chartCtx.moveTo(padding, padding);
    chartCtx.lineTo(padding, chartHeight - padding);
    chartCtx.lineTo(chartWidth - padding, chartHeight - padding);
    chartCtx.stroke();

    // Draw grid
    chartCtx.strokeStyle = '#e1e4e8';
    chartCtx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (graphHeight / 5) * i;
        chartCtx.beginPath();
        chartCtx.moveTo(padding, y);
        chartCtx.lineTo(chartWidth - padding, y);
        chartCtx.stroke();
    }

    // Draw loss line
    chartCtx.strokeStyle = '#4a90e2';
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();

    lossHistory.forEach((point, idx) => {
        const x = padding + (point.epoch / maxEpoch) * graphWidth;
        const y = (chartHeight - padding) - (point.loss / maxLoss) * graphHeight;

        if (idx === 0) {
            chartCtx.moveTo(x, y);
        } else {
            chartCtx.lineTo(x, y);
        }
    });

    chartCtx.stroke();

    // Draw labels
    chartCtx.fillStyle = '#666';
    chartCtx.font = '12px sans-serif';
    chartCtx.textAlign = 'center';

    // X-axis label
    chartCtx.fillText('Epochs', chartWidth / 2, chartHeight - 10);

    // Y-axis label
    chartCtx.save();
    chartCtx.translate(15, chartHeight / 2);
    chartCtx.rotate(-Math.PI / 2);
    chartCtx.fillText('Loss', 0, 0);
    chartCtx.restore();

    // Draw scale values
    chartCtx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const y = padding + (graphHeight / 5) * i;
        const value = maxLoss * (1 - i / 5);
        chartCtx.fillText(value.toFixed(3), padding - 10, y + 5);
    }

    chartCtx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
        const x = padding + (graphWidth / 5) * i;
        const value = Math.round(maxEpoch * (i / 5));
        chartCtx.fillText(value, x, chartHeight - padding + 20);
    }
}

// Clear chart
function clearChart() {
    chartCtx.clearRect(0, 0, chartWidth, chartHeight);

    // Draw placeholder text
    chartCtx.fillStyle = '#999';
    chartCtx.font = '16px sans-serif';
    chartCtx.textAlign = 'center';
    chartCtx.fillText('Training loss will appear here', chartWidth / 2, chartHeight / 2);
}

// Update status message
function updateStatus(message, type = 'info') {
    elements.trainingStatus.textContent = message;

    const statusBox = document.getElementById('training-status');
    statusBox.classList.remove('training-active');

    if (type === 'training') {
        statusBox.classList.add('training-active');
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
