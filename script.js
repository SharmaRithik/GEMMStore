// script.js

// DOM Elements
const shaderCodeTextarea = document.getElementById('shader-code');
const loadExampleButton = document.getElementById('load-example');
const clearCodeButton = document.getElementById('clear-code');
const statusContainer = document.getElementById('status-container');
const matrixSizesCheckboxes = document.querySelectorAll('#matrix-sizes .checkbox-item');
const matrixConfigurationsContainer = document.getElementById('matrix-configurations');
const timingResultsContainer = document.getElementById('timing-results');
const runAllMatricesButton = document.getElementById('run-all-matrices');

// State management
let selectedSizes = [512, 1024];
let matrixConfigurations = [];
let results = {};

// Load example shader code from shader.wgsl
loadExampleButton.addEventListener('click', () => {
  fetch('shader.wgsl')
    .then(response => response.text())
    .then(data => {
      shaderCodeTextarea.value = data;
      showStatus('Example shader code loaded from shader.wgsl', 'success');
    })
    .catch(error => {
      console.error('Error loading shader file:', error);
      showStatus('Error loading shader file', 'error');
    });
});

// Clear shader code
clearCodeButton.addEventListener('click', () => {
  shaderCodeTextarea.value = '';
  showStatus('Shader code cleared', 'success');
});

// Show status message
function showStatus(message, type) {
  statusContainer.innerHTML = `<div class="status ${type}">${message}</div>`;
  setTimeout(() => {
    statusContainer.innerHTML = '';
  }, 5000);
}

// Toggle checkbox items
function setupCheckboxToggles() {
  matrixSizesCheckboxes.forEach(item => {
    item.addEventListener('click', () => {
      const size = parseInt(item.getAttribute('data-size'));
      const checkbox = item.querySelector('input');

      if (item.classList.contains('active')) {
        // Only remove if at least one size remains selected
        if (selectedSizes.length > 1) {
          item.classList.remove('active');
          checkbox.checked = false;
          selectedSizes = selectedSizes.filter(s => s !== size);
        }
      } else {
        item.classList.add('active');
        checkbox.checked = true;
        if (!selectedSizes.includes(size)) {
          selectedSizes.push(size);
        }
      }
      generateMatrixConfigurations();
    });
  });
}

// Calculate GFLOPs for matrix multiplication
function calculateGflops(size, executionTimeMs) {
  const operations = 2 * Math.pow(size, 3);
  const seconds = executionTimeMs / 1000;
  return (operations / seconds) / 1e9;
}

// Generate matrix configurations based on selected sizes
function generateMatrixConfigurations() {
  matrixConfigurations = [];
  for (const size of selectedSizes) {
    matrixConfigurations.push({
      id: `${size}x${size}`,
      matrixA: [size, size],
      matrixB: [size, size],
      result: [size, size]
    });
  }
  renderMatrixConfigurations();
}

// Render matrix configurations in the UI
function renderMatrixConfigurations() {
  matrixConfigurationsContainer.innerHTML = '';
  if (matrixConfigurations.length === 0) {
    matrixConfigurationsContainer.innerHTML = '<p class="placeholder-text">No configurations selected</p>';
    return;
  }
  matrixConfigurations.forEach(config => {
    const matrixRow = document.createElement('div');
    matrixRow.className = 'matrix-row';
    matrixRow.setAttribute('data-config-id', config.id);
    matrixRow.innerHTML = `
      <div class="matrix-label">
        <div>
          <span class="matrix-dimensions">[${config.matrixA[0]}×${config.matrixA[1]}]</span> × 
          <span class="matrix-dimensions">[${config.matrixB[0]}×${config.matrixB[1]}]</span> = 
          <span class="matrix-dimensions">[${config.result[0]}×${config.result[1]}]</span>
        </div>
        <button class="run-btn" data-config-id="${config.id}">Run</button>
      </div>
    `;
    matrixConfigurationsContainer.appendChild(matrixRow);
    const runButton = matrixRow.querySelector('.run-btn');
    runButton.addEventListener('click', () => {
      const configId = runButton.getAttribute('data-config-id');
      const config = matrixConfigurations.find(c => c.id === configId);
      if (config) {
        runConfigurationTest(config);
      }
    });
  });
}

// Run all selected configurations
async function runAllConfigurations() {
  const shaderCode = shaderCodeTextarea.value.trim();
  if (!shaderCode) {
    showStatus('Please enter shader code', 'error');
    return;
  }
  if (matrixConfigurations.length === 0) {
    showStatus('No configurations selected', 'error');
    return;
  }
  results = {};
  timingResultsContainer.innerHTML = '<p class="placeholder-text">Running tests...</p>';
  runAllMatricesButton.disabled = true;
  document.querySelectorAll('.run-btn').forEach(btn => btn.disabled = true);
  for (const config of matrixConfigurations) {
    await runConfigurationTest(config);
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  runAllMatricesButton.disabled = false;
  document.querySelectorAll('.run-btn').forEach(btn => btn.disabled = false);
  renderAllResults();
}

// Run a single configuration test
async function runConfigurationTest(config) {
  const shaderCode = shaderCodeTextarea.value.trim();
  if (!shaderCode) {
    showStatus('Please enter shader code', 'error');
    return;
  }
  try {
    showStatus(`Running test for ${config.id}...`, 'success');
    const timingData = {
      setupTime: 0,
      bufferTime: 0,
      compileTime: 0,
      execTime: 0,
      totalTime: 0,
      gflops: 0
    };
    const totalStartTime = performance.now();

    // 1. GPU Setup
    const setupStartTime = performance.now();
    if (!navigator.gpu) {
      showStatus('WebGPU is not supported in your browser', 'error');
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      showStatus('Failed to get GPU adapter', 'error');
      return;
    }
    if (!adapter.features.has("shader-f16")) {
      showStatus('16-bit floating-point value support is not available', 'error');
      return;
    }
    const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
    const setupEndTime = performance.now();
    timingData.setupTime = setupEndTime - setupStartTime;

    // 2. Shader Compilation
    const compileStartTime = performance.now();
    const shaderModule = device.createShaderModule({ code: shaderCode });
    const compileEndTime = performance.now();
    timingData.compileTime = compileEndTime - compileStartTime;

    // 3. Buffer Creation
    const bufferStartTime = performance.now();
    const matrixSize = {
      a: config.matrixA,
      b: config.matrixB,
      result: config.result
    };
    const matrixASize = matrixSize.a[0] * matrixSize.a[1];
    const matrixBSize = matrixSize.b[0] * matrixSize.b[1];
    const resultMatrixSize = matrixSize.result[0] * matrixSize.result[1];
    const matrixAData = new Uint16Array(matrixASize).fill(0x3C00); // f16 1.0
    const matrixBData = new Uint16Array(matrixBSize).fill(0x4000); // f16 2.0
    const resultMatrixData = new Uint16Array(resultMatrixSize).fill(0);
    const matrixABuffer = device.createBuffer({
      size: 8 + matrixAData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const matrixBBuffer = device.createBuffer({
      size: 8 + matrixBData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const resultMatrixBuffer = device.createBuffer({
      size: 8 + resultMatrixData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    const readbackBuffer = device.createBuffer({
      size: 8 + resultMatrixData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Prepare buffer data with dimension headers
    const matrixABufferData = new ArrayBuffer(8 + matrixAData.byteLength);
    const matrixADataView = new DataView(matrixABufferData);
    matrixADataView.setUint32(0, matrixSize.a[0], true);
    matrixADataView.setUint32(4, matrixSize.a[1], true);
    new Uint16Array(matrixABufferData, 8).set(matrixAData);

    const matrixBBufferData = new ArrayBuffer(8 + matrixBData.byteLength);
    const matrixBDataView = new DataView(matrixBBufferData);
    matrixBDataView.setUint32(0, matrixSize.b[0], true);
    matrixBDataView.setUint32(4, matrixSize.b[1], true);
    new Uint16Array(matrixBBufferData, 8).set(matrixBData);

    const resultMatrixBufferData = new ArrayBuffer(8 + resultMatrixData.byteLength);
    const resultMatrixDataView = new DataView(resultMatrixBufferData);
    resultMatrixDataView.setUint32(0, matrixSize.result[0], true);
    resultMatrixDataView.setUint32(4, matrixSize.result[1], true);
    new Uint16Array(resultMatrixBufferData, 8).set(resultMatrixData);

    // Write data to GPU buffers
    device.queue.writeBuffer(matrixABuffer, 0, matrixABufferData);
    device.queue.writeBuffer(matrixBBuffer, 0, matrixBBufferData);
    device.queue.writeBuffer(resultMatrixBuffer, 0, resultMatrixBufferData);

    // Create bind group layout and bind group
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
      ]
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: matrixABuffer } },
        { binding: 1, resource: { buffer: matrixBBuffer } },
        { binding: 2, resource: { buffer: resultMatrixBuffer } }
      ]
    });
    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    });
    const computePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "main" }
    });
    const bufferEndTime = performance.now();
    timingData.bufferTime = bufferEndTime - bufferStartTime;

    // 4. Execute compute shader
    const execStartTime = performance.now();
    const workgroupSize = 8; // Matches shader @workgroup_size(8,8)
    const dispatchX = Math.ceil(matrixSize.result[0] / workgroupSize);
    const dispatchY = Math.ceil(matrixSize.result[1] / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY);
    passEncoder.end();
    commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer, 0,
      readbackBuffer, 0,
      8 + resultMatrixData.byteLength
    );
    const commandBuffer = commandEncoder.finish();
    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const execEndTime = performance.now();
    timingData.execTime = execEndTime - execStartTime;
    timingData.gflops = calculateGflops(matrixSize.a[0], timingData.execTime);
    readbackBuffer.unmap();
    const totalEndTime = performance.now();
    timingData.totalTime = totalEndTime - totalStartTime;

    // Store and render results
    results[config.id] = { config: config, timing: timingData };
    renderResults(config, timingData);
    return timingData;
  } catch (error) {
    console.error(`Error executing configuration ${config.id}:`, error);
    showStatus(`Error with ${config.id}: ${error.message}`, 'error');
    results[config.id] = { config: config, error: error.message };
    return null;
  }
}

// Render results for a single configuration
function renderResults(config, timingData) {
  const configElement = document.querySelector(`.matrix-row[data-config-id="${config.id}"]`);
  if (configElement) {
    const runButton = configElement.querySelector('.run-btn');
    runButton.textContent = `${timingData.gflops.toFixed(2)} GFLOPs`;
  }
  renderAllResults();
}

// Render comprehensive results for all configurations
function renderAllResults() {
  if (Object.keys(results).length === 0) {
    timingResultsContainer.innerHTML = '<p class="placeholder-text">No results yet</p>';
    return;
  }
  let resultsHtml = '<div class="comparison-grid">';
  selectedSizes.forEach(size => {
    const configId = `${size}x${size}`;
    const result = results[configId];
    if (result) {
      if (result.error) {
        resultsHtml += `
          <div class="timing-card" style="background-color: #fde7e9;">
            <div class="label">${size}×${size}</div>
            <div class="time" style="color: #a80000;">Error</div>
          </div>
        `;
      } else {
        resultsHtml += `
          <div class="timing-card">
            <div class="label">${size}×${size}</div>
            <div class="time">${result.timing.gflops.toFixed(2)} GFLOPs</div>
          </div>
        `;
      }
    } else {
      resultsHtml += `
        <div class="timing-card" style="background-color: #f0f0f0;">
          <div class="label">${size}×${size}</div>
          <div class="time" style="color: #888;">Not Run</div>
        </div>
      `;
    }
  });
  resultsHtml += '</div>';
  resultsHtml += `
    <h4 style="margin-top: 20px; margin-bottom: 10px;">Detailed Performance</h4>
    <table style="width:100%; border-collapse: collapse;">
      <thead>
        <tr>
          <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Size</th>
          <th style="text-align: right; padding: 8px; border-bottom: 2px solid #ddd;">GFLOPs</th>
          <th style="text-align: right; padding: 8px; border-bottom: 2px solid #ddd;">Exec Time</th>
          <th style="text-align: right; padding: 8px; border-bottom: 2px solid #ddd;">Total Time</th>
        </tr>
      </thead>
      <tbody>
  `;
  selectedSizes.forEach(size => {
    const configId = `${size}x${size}`;
    const result = results[configId];
    if (result && !result.error) {
      resultsHtml += `
        <tr>
          <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">${size}×${size}</td>
          <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd; font-weight: bold; color: #0078d4;">${result.timing.gflops.toFixed(2)} GFLOPs</td>
          <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">${result.timing.execTime.toFixed(2)} ms</td>
          <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">${result.timing.totalTime.toFixed(2)} ms</td>
        </tr>
      `;
    }
  });
  resultsHtml += '</tbody></table>';
  if (Object.keys(results).filter(id => !results[id].error).length > 0) {
    resultsHtml += `
      <h4 style="margin-top: 20px; margin-bottom: 10px;">GFLOPs Comparison</h4>
      <div style="width: 100%; height: 200px; background-color: white; position: relative; margin-top: 15px; border-left: 1px solid #ddd; border-bottom: 1px solid #ddd;">
    `;
    const maxGflops = Math.max(...Object.values(results).filter(r => !r.error).map(r => r.timing.gflops));
    Object.entries(results)
      .filter(([id, r]) => !r.error)
      .forEach(([id, result], index, filteredResults) => {
        const width = 100 / filteredResults.length;
        const height = (result.timing.gflops / maxGflops) * 100;
        const left = (index * width);
        resultsHtml += `
          <div style="position: absolute; bottom: 0; left: ${left}%; width: ${width * 0.8}%; height: ${height}%; background-color: #0078d4; margin-left: ${width * 0.1}%;">
            <div style="position: absolute; top: -25px; width: 100%; text-align: center; font-size: 12px;">${result.timing.gflops.toFixed(2)} GFLOPs</div>
          </div>
          <div style="position: absolute; bottom: -25px; left: ${left}%; width: ${width}%; text-align: center; font-size: 12px;">
            ${result.config.matrixA[0]}×${result.config.matrixA[0]}
          </div>
        `;
      });
    resultsHtml += '</div>';
  }
  timingResultsContainer.innerHTML = resultsHtml;
}

// Initialize the application
function init() {
  setupCheckboxToggles();
  generateMatrixConfigurations();
  runAllMatricesButton.addEventListener('click', runAllConfigurations);
}

init();

