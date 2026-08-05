import { NCAWebGLRuntime } from "./nca_runtime.js?v=13";

const params = new URLSearchParams(window.location.search);
const requestedModel = params.get("model");
let modelId = requestedModel && /^[A-Za-z0-9_.-]+$/.test(requestedModel) ? requestedModel : null;
const assetVersion = params.get("v");
const APP_VERSION = "v13";

const canvas = document.getElementById("nca-canvas");
const status = document.getElementById("status");
const modelSelect = document.getElementById("model-select");
const resetButton = document.getElementById("reset");
const playPauseButton = document.getElementById("play-pause");
const stepOnceButton = document.getElementById("step-once");
const displayRgbButton = document.getElementById("display-rgb");
const displayHiddenGridButton = document.getElementById("display-hidden-grid");
const zeroButton = document.getElementById("brush-zero");
const randomButton = document.getElementById("brush-random");
const brushTargetAllButton = document.getElementById("brush-target-all");
const brushTargetRgbButton = document.getElementById("brush-target-rgb");
const playbackSpeedInput = document.getElementById("playback-speed");
const playbackSpeedValue = document.getElementById("playback-speed-value");
const radiusInput = document.getElementById("brush-radius");
const hiddenSaturationInput = document.getElementById("hidden-saturation");
const hiddenSaturationValue = document.getElementById("hidden-saturation-value");
const validateButton = document.getElementById("validate");
const modelMetadata = document.getElementById("model-metadata");
const channelSummaryBody = document.getElementById("channel-summary-body");
const channelChangeChart = document.getElementById("channel-change-chart");
const liveIntensityLegend = document.getElementById("live-intensity-legend");
const liveIntensityChart = document.getElementById("live-intensity-chart");
playPauseButton.textContent = "Play";

let runtime = null;
let paused = true;
let brushMode = "zero";
let brushTarget = "all";
let displayMode = "rgb";
let pointerDown = false;
let detailMessage = "";
let stepAccumulator = 0.0;
let animationStarted = false;
let liveIntensitySeries = [];
let lastLiveIntensityStep = -1;
let lastLiveIntensitySampleMs = 0;

const LIVE_INTENSITY_MAX_SAMPLES = 600;
const LIVE_INTENSITY_SAMPLE_STEP_INTERVAL = 1;
const LIVE_INTENSITY_SAMPLE_MS = 250;
const RGB_INTENSITY_LINES = [
  { key: "ch0", label: "channel 0 mean", color: "#ef6f6c" },
  { key: "ch1", label: "channel 1 mean", color: "#72c66b" },
  { key: "ch2", label: "channel 2 mean", color: "#5fb3d4" },
];
const HIDDEN_INTENSITY_COLOR = "#8f9ba5";

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchFloat32(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return new Float32Array(await response.arrayBuffer());
}

function modelDir(id = modelId) {
  return `./models/${id}`;
}

function versionSuffix() {
  return assetVersion ? `?v=${encodeURIComponent(assetVersion)}` : "";
}

async function loadModelIndex() {
  try {
    const index = await fetchJson(`./models/index.json${versionSuffix()}`);
    return index.models ?? [];
  } catch (error) {
    console.warn("Model index unavailable; falling back to requested model.", error);
    return modelId ? [{ id: modelId, label: modelId }] : [];
  }
}

function populateModelSelect(models) {
  modelSelect.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label ?? model.id;
    modelSelect.appendChild(option);
  }
  if ((modelId === null || !models.some((model) => model.id === modelId)) && models.length > 0) {
    modelId = models[0].id;
  }
  modelSelect.value = modelId;
}

function updateModelUrl(id) {
  const url = new URL(window.location.href);
  url.searchParams.set("model", id);
  if (assetVersion) url.searchParams.set("v", assetVersion);
  window.history.replaceState(null, "", url);
}

function canvasToGrid(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  return [
    Math.floor(x * runtime.width),
    Math.floor(y * runtime.height),
  ];
}

function paint(event) {
  if (!runtime) return;
  const [x, y] = canvasToGrid(event);
  runtime.paint(x, y, Number(radiusInput.value), brushMode, brushTarget);
  liveIntensitySample(true);
}

function setBrushMode(mode) {
  brushMode = mode;
  zeroButton.setAttribute("aria-pressed", String(mode === "zero"));
  randomButton.setAttribute("aria-pressed", String(mode === "random"));
}

function setBrushTarget(target) {
  brushTarget = target;
  brushTargetAllButton.setAttribute("aria-pressed", String(target === "all"));
  brushTargetRgbButton.setAttribute("aria-pressed", String(target === "rgb"));
}

function setDisplayMode(mode) {
  displayMode = mode;
  displayRgbButton.setAttribute("aria-pressed", String(mode === "rgb"));
  displayHiddenGridButton.setAttribute("aria-pressed", String(mode === "hidden-grid"));
}

function drawOptions() {
  return {
    mode: displayMode === "hidden-grid" ? "hidden-grid" : "rgb",
    gridChannelOffset: 3,
    gridSaturation: Number(hiddenSaturationInput.value),
  };
}

function playbackSpeed() {
  return 2 ** Number(playbackSpeedInput.value);
}

function playbackSpeedLabel() {
  const speed = playbackSpeed();
  return speed < 1 ? `1/${Math.round(1 / speed)}x` : `${speed}x`;
}

function formatValue(value) {
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(3);
  }
  return value.toPrecision(4);
}

function setMetadata(manifest) {
  const entries = [
    ["model id", manifest.modelId],
    ["family", manifest.family],
    ["channels", manifest.channels],
    ["kernels", manifest.kernels.join(", ")],
    ["padding", manifest.padding],
    ["grid", `${manifest.gridSize[0]} x ${manifest.gridSize[1]}`],
    ["fire rate", manifest.fireRate],
    ["validation steps", manifest.validation.referenceSteps],
  ];
  modelMetadata.replaceChildren();
  for (const [label, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    modelMetadata.append(dt, dd);
  }
}

function channelStats(state, channels, height, width) {
  const pixels = height * width;
  const sum = new Float64Array(channels);
  const max = new Float32Array(channels);
  const min = new Float32Array(channels);
  for (let ch = 0; ch < channels; ch += 1) {
    max[ch] = -Infinity;
    min[ch] = Infinity;
    const offset = ch * pixels;
    for (let i = 0; i < pixels; i += 1) {
      const value = state[offset + i];
      sum[ch] += value;
      if (value > max[ch]) max[ch] = value;
      if (value < min[ch]) min[ch] = value;
    }
  }
  return { sum, max, min };
}

function summarizeChannels(manifest, x0, reference) {
  const [channels, height, width] = manifest.initialState.shape;
  const initial = channelStats(x0, channels, height, width);
  const final = channelStats(reference, channels, height, width);
  return Array.from({ length: channels }, (_, channel) => ({
    channel,
    initialSum: initial.sum[channel],
    referenceSum: final.sum[channel],
    deltaSum: final.sum[channel] - initial.sum[channel],
    referenceMax: final.max[channel],
  })).sort((a, b) => Math.abs(b.deltaSum) - Math.abs(a.deltaSum)).slice(0, 8);
}

function renderSummaryTable(rows) {
  channelSummaryBody.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of [
      row.channel,
      row.initialSum,
      row.referenceSum,
      row.deltaSum,
      row.referenceMax,
    ]) {
      const td = document.createElement("td");
      td.textContent = typeof value === "number" ? formatValue(value) : String(value);
      tr.appendChild(td);
    }
    channelSummaryBody.appendChild(tr);
  }
}

function renderDeltaChart(rows) {
  const svgNS = "http://www.w3.org/2000/svg";
  const width = 680;
  const rowHeight = 28;
  const mid = 380;
  const height = rows.length * rowHeight + 48;
  const maxAbs = Math.max(1e-8, ...rows.map((row) => Math.abs(row.deltaSum)));
  const scale = 250 / maxAbs;
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const axis = document.createElementNS(svgNS, "line");
  axis.setAttribute("x1", String(mid));
  axis.setAttribute("y1", "18");
  axis.setAttribute("x2", String(mid));
  axis.setAttribute("y2", String(height - 20));
  axis.setAttribute("stroke", "#6b737c");
  svg.appendChild(axis);

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const y = 28 + i * rowHeight;
    const barWidth = Math.abs(row.deltaSum) * scale;
    const x = row.deltaSum >= 0 ? mid : mid - barWidth;
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", "12");
    label.setAttribute("y", String(y + 15));
    label.setAttribute("fill", "#d8dee4");
    label.textContent = `channel ${row.channel}`;
    svg.appendChild(label);

    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(barWidth));
    rect.setAttribute("height", "18");
    rect.setAttribute("fill", row.deltaSum >= 0 ? "#5fb3d4" : "#e0776d");
    rect.setAttribute("rx", "3");
    svg.appendChild(rect);

    const value = document.createElementNS(svgNS, "text");
    value.setAttribute("x", String(mid + 260));
    value.setAttribute("y", String(y + 15));
    value.setAttribute("fill", "#aab4bb");
    value.textContent = formatValue(row.deltaSum);
    svg.appendChild(value);
  }
  channelChangeChart.replaceChildren(svg);
}

function renderAssetAnalysis(manifest, x0, reference) {
  setMetadata(manifest);
  const rows = summarizeChannels(manifest, x0, reference);
  renderSummaryTable(rows);
  renderDeltaChart(rows);
}

function resetLiveIntensityPlot() {
  liveIntensitySeries = [];
  lastLiveIntensityStep = -1;
  lastLiveIntensitySampleMs = 0;
  renderLiveIntensityLegend();
  renderLiveIntensityChart();
}

function renderLiveIntensityLegend() {
  liveIntensityLegend.replaceChildren();
  for (const line of RGB_INTENSITY_LINES) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = line.color;
    const label = document.createElement("span");
    label.textContent = line.label;
    item.append(swatch, label);
    liveIntensityLegend.appendChild(item);
  }
  const hiddenItem = document.createElement("span");
  hiddenItem.className = "legend-item";
  const hiddenSwatch = document.createElement("span");
  hiddenSwatch.className = "legend-swatch";
  hiddenSwatch.style.background = HIDDEN_INTENSITY_COLOR;
  const hiddenLabel = document.createElement("span");
  hiddenLabel.textContent = "hidden channel means";
  hiddenItem.append(hiddenSwatch, hiddenLabel);
  liveIntensityLegend.appendChild(hiddenItem);
}

function liveIntensitySample(force = false) {
  if (!runtime || (!force && runtime.stepCount === lastLiveIntensityStep)) return;
  const now = performance.now();
  if (!force && now - lastLiveIntensitySampleMs < LIVE_INTENSITY_SAMPLE_MS) return;
  if (
    !force &&
    lastLiveIntensityStep >= 0 &&
    runtime.stepCount - lastLiveIntensityStep < LIVE_INTENSITY_SAMPLE_STEP_INTERVAL
  ) {
    return;
  }
  const state = runtime.readState();
  const channels = runtime.channels;
  const pixels = runtime.width * runtime.height;
  const means = new Float64Array(channels);
  for (let ch = 0; ch < channels; ch += 1) {
    let sum = 0.0;
    const offset = ch * pixels;
    for (let i = 0; i < pixels; i += 1) {
      sum += state[offset + i];
    }
    means[ch] = sum / pixels;
  }
  const sample = {
    step: runtime.stepCount,
    ch0: means[0] ?? 0.0,
    ch1: means[1] ?? 0.0,
    ch2: means[2] ?? 0.0,
    hidden: Array.from(means.slice(3)),
  };
  if (force && liveIntensitySeries.at(-1)?.step === runtime.stepCount) {
    liveIntensitySeries[liveIntensitySeries.length - 1] = sample;
  } else {
    liveIntensitySeries.push(sample);
  }
  if (liveIntensitySeries.length > LIVE_INTENSITY_MAX_SAMPLES) {
    liveIntensitySeries = liveIntensitySeries.slice(-LIVE_INTENSITY_MAX_SAMPLES);
  }
  lastLiveIntensityStep = runtime.stepCount;
  lastLiveIntensitySampleMs = now;
  renderLiveIntensityChart();
}

function renderLiveIntensityChart() {
  const svgNS = "http://www.w3.org/2000/svg";
  const width = 900;
  const height = 280;
  const margin = { top: 20, right: 18, bottom: 34, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const steps = liveIntensitySeries.map((sample) => sample.step);
  const xMin = steps.length > 0 ? Math.min(...steps) : 0;
  const xMax = steps.length > 1 ? Math.max(...steps) : Math.max(1, xMin + 1);
  const values = [];
  for (const sample of liveIntensitySeries) {
    for (const line of RGB_INTENSITY_LINES) values.push(sample[line.key]);
    for (const value of sample.hidden) values.push(value);
  }
  let yMin = values.length > 0 ? Math.min(...values) : 0;
  let yMax = values.length > 0 ? Math.max(...values) : 1;
  if (yMin === yMax) {
    yMin -= 0.5;
    yMax += 0.5;
  }
  const yPad = 0.08 * (yMax - yMin);
  yMin -= yPad;
  yMax += yPad;

  const xScale = (step) => margin.left + ((step - xMin) / (xMax - xMin)) * plotWidth;
  const yScale = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * plotHeight;

  function addLine(x1, y1, x2, y2, stroke = "#31383e") {
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    line.setAttribute("stroke", stroke);
    svg.appendChild(line);
  }

  addLine(margin.left, margin.top, margin.left, margin.top + plotHeight, "#6b737c");
  addLine(margin.left, margin.top + plotHeight, margin.left + plotWidth, margin.top + plotHeight, "#6b737c");

  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const y = yScale(value);
    addLine(margin.left, y, margin.left + plotWidth, y, "#242a2f");
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(margin.left - 8));
    label.setAttribute("y", String(y + 4));
    label.setAttribute("fill", "#8f9ba5");
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "12");
    label.textContent = formatValue(value);
    svg.appendChild(label);
  }

  const xLabel = document.createElementNS(svgNS, "text");
  xLabel.setAttribute("x", String(margin.left + plotWidth));
  xLabel.setAttribute("y", String(height - 8));
  xLabel.setAttribute("fill", "#8f9ba5");
  xLabel.setAttribute("text-anchor", "end");
  xLabel.setAttribute("font-size", "12");
  xLabel.textContent = `step ${xMax}`;
  svg.appendChild(xLabel);

  const hiddenCount = liveIntensitySeries.reduce(
    (count, sample) => Math.max(count, sample.hidden.length),
    0,
  );
  for (let hiddenIndex = 0; hiddenIndex < hiddenCount; hiddenIndex += 1) {
    if (liveIntensitySeries.length === 0) continue;
    const points = liveIntensitySeries
      .filter((sample) => hiddenIndex < sample.hidden.length)
      .map((sample) => `${xScale(sample.step).toFixed(2)},${yScale(sample.hidden[hiddenIndex]).toFixed(2)}`)
      .join(" ");
    const polyline = document.createElementNS(svgNS, "polyline");
    polyline.setAttribute("points", points);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", HIDDEN_INTENSITY_COLOR);
    polyline.setAttribute("stroke-opacity", "0.38");
    polyline.setAttribute("stroke-width", "1.2");
    polyline.setAttribute("stroke-linejoin", "round");
    polyline.setAttribute("stroke-linecap", "round");
    svg.appendChild(polyline);
  }

  for (const line of RGB_INTENSITY_LINES) {
    if (liveIntensitySeries.length === 0) continue;
    const points = liveIntensitySeries
      .map((sample) => `${xScale(sample.step).toFixed(2)},${yScale(sample[line.key]).toFixed(2)}`)
      .join(" ");
    const polyline = document.createElementNS(svgNS, "polyline");
    polyline.setAttribute("points", points);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", line.color);
    polyline.setAttribute("stroke-width", "2.4");
    polyline.setAttribute("stroke-linejoin", "round");
    polyline.setAttribute("stroke-linecap", "round");
    svg.appendChild(polyline);
  }

  if (liveIntensitySeries.length === 0) {
    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", String(width / 2));
    label.setAttribute("y", String(height / 2));
    label.setAttribute("fill", "#8f9ba5");
    label.setAttribute("text-anchor", "middle");
    label.textContent = "Run or step the simulation to collect live means.";
    svg.appendChild(label);
  }

  liveIntensityChart.replaceChildren(svg);
}

function render() {
  let stepped = false;
  if (runtime && !paused) {
    stepAccumulator += playbackSpeed();
    const stepsThisFrame = Math.floor(stepAccumulator);
    stepAccumulator -= stepsThisFrame;
    for (let i = 0; i < stepsThisFrame; i += 1) {
      runtime.step();
    }
    stepped = stepsThisFrame > 0;
  }
  if (stepped) liveIntensitySample();
  if (runtime) {
    runtime.draw(drawOptions());
    const state = paused ? "Paused" : "Playing";
    setStatus(`${state} | Step ${runtime.stepCount} | ${APP_VERSION}${detailMessage ? ` | ${detailMessage}` : ""}`);
  }
  requestAnimationFrame(render);
}

async function init() {
  try {
    const models = await loadModelIndex();
    populateModelSelect(models);
    if (modelId === null) {
      throw new Error("No exported models found. Run an export command or refresh models/index.json.");
    }
    await loadModel(modelId);
    if (!animationStarted) {
      animationStarted = true;
      requestAnimationFrame(render);
    }
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

async function loadModel(nextModelId) {
  try {
    modelId = nextModelId;
    paused = true;
    playPauseButton.textContent = "Play";
    detailMessage = "";
    stepAccumulator = 0.0;
    setStatus(`Loading ${modelId}...`);
    const suffix = versionSuffix();
    const dir = modelDir(modelId);
    const manifest = await fetchJson(`${dir}/manifest.json${suffix}`);
    const [weights, x0, reference] = await Promise.all([
      fetchFloat32(`${dir}/${manifest.weights.path}${suffix}`),
      fetchFloat32(`${dir}/${manifest.initialState.path}${suffix}`),
      fetchFloat32(`${dir}/${manifest.validation.reference.path}${suffix}`),
    ]);
    runtime = new NCAWebGLRuntime(canvas, manifest, weights, x0);
    runtime.reset();
    modelSelect.value = modelId;
    renderAssetAnalysis(manifest, x0, reference);
    resetLiveIntensityPlot();
    liveIntensitySample(true);
    runtime.draw(drawOptions());
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

async function reloadSelectedModel() {
  await loadModel(modelSelect.value);
  updateModelUrl(modelSelect.value);
}

async function validateCurrentModel() {
  if (!runtime) return;
  try {
    const referenceInfo = runtime.manifest.validation.reference;
    const suffix = versionSuffix();
    const reference = await fetchFloat32(`${modelDir()}/${referenceInfo.path}${suffix}`);
    const result = runtime.validate(reference);
    detailMessage = `Validation max abs error: ${result.maxAbsError.toExponential(3)}`;
    resetLiveIntensityPlot();
    liveIntensitySample(true);
    console.log(result);
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

resetButton.addEventListener("click", () => {
  detailMessage = "";
  stepAccumulator = 0.0;
  runtime?.reset();
  resetLiveIntensityPlot();
  liveIntensitySample(true);
});
playPauseButton.addEventListener("click", () => {
  paused = !paused;
  stepAccumulator = 0.0;
  playPauseButton.textContent = paused ? "Play" : "Pause";
});
stepOnceButton.addEventListener("click", () => {
  if (!runtime) return;
  detailMessage = "";
  stepAccumulator = 0.0;
  runtime.step();
  liveIntensitySample(true);
  runtime.draw(drawOptions());
});
playbackSpeedInput.addEventListener("input", () => {
  playbackSpeedValue.textContent = playbackSpeedLabel();
});
displayRgbButton.addEventListener("click", () => setDisplayMode("rgb"));
displayHiddenGridButton.addEventListener("click", () => setDisplayMode("hidden-grid"));
hiddenSaturationInput.addEventListener("input", () => {
  hiddenSaturationValue.textContent = `${hiddenSaturationInput.value}x`;
});
zeroButton.addEventListener("click", () => setBrushMode("zero"));
randomButton.addEventListener("click", () => setBrushMode("random"));
brushTargetAllButton.addEventListener("click", () => setBrushTarget("all"));
brushTargetRgbButton.addEventListener("click", () => setBrushTarget("rgb"));
modelSelect.addEventListener("change", reloadSelectedModel);
validateButton.addEventListener("click", validateCurrentModel);

canvas.addEventListener("pointerdown", (event) => {
  pointerDown = true;
  canvas.setPointerCapture(event.pointerId);
  paint(event);
});
canvas.addEventListener("pointermove", (event) => {
  if (pointerDown) paint(event);
});
canvas.addEventListener("pointerup", () => {
  pointerDown = false;
});
canvas.addEventListener("pointercancel", () => {
  pointerDown = false;
});

init();
