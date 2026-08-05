const VERTEX_SHADER = `#version 300 es
precision highp float;
const vec2 POSITIONS[3] = vec2[3](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);
out vec2 vUv;
void main() {
  vec2 position = POSITIONS[gl_VertexID];
  vUv = 0.5 * position + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}`;

const PERCEPTION_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform sampler2D uGradX;
uniform sampler2D uGradY;
uniform sampler2D uLap;
uniform int uWidth;
uniform int uHeight;
uniform int uChannels;
uniform int uOutGroups;
uniform int uPaddingMode;
out vec4 outValue;

float component(vec4 v, int c) {
  if (c == 0) return v.r;
  if (c == 1) return v.g;
  if (c == 2) return v.b;
  return v.a;
}

void setComponent(inout vec4 v, int c, float value) {
  if (c == 0) v.r = value;
  else if (c == 1) v.g = value;
  else if (c == 2) v.b = value;
  else v.a = value;
}

float readState(int x, int y, int ch) {
  if (uPaddingMode == 1) {
    x = clamp(x, 0, uWidth - 1);
    y = clamp(y, 0, uHeight - 1);
  } else {
    x = (x + uWidth) % uWidth;
    y = (y + uHeight) % uHeight;
  }
  int group = ch / 4;
  int comp = ch - group * 4;
  return component(texelFetch(uState, ivec2(x + group * uWidth, y), 0), comp);
}

float readKernel(sampler2D tex, int kx, int ky) {
  return texelFetch(tex, ivec2(kx, ky), 0).r;
}

float convolve3(int x, int y, int ch, sampler2D kernel) {
  float sum = 0.0;
  for (int ky = 0; ky < 3; ky++) {
    for (int kx = 0; kx < 3; kx++) {
      sum += readKernel(kernel, kx, ky) * readState(x + kx - 1, y + ky - 1, ch);
    }
  }
  return sum;
}

void main() {
  ivec2 frag = ivec2(gl_FragCoord.xy);
  int x = frag.x % uWidth;
  int y = frag.y;
  int group = frag.x / uWidth;
  vec4 result = vec4(0.0);
  for (int i = 0; i < 4; i++) {
    int feature = group * 4 + i;
    int band = feature / uChannels;
    int ch = feature - band * uChannels;
    if (group >= uOutGroups || ch >= uChannels) continue;
    float value = 0.0;
    if (band == 0) {
      value = readState(x, y, ch);
    } else if (band == 1) {
      value = convolve3(x, y, ch, uGradX);
    } else if (band == 2) {
      value = convolve3(x, y, ch, uGradY);
    } else if (band == 3) {
      value = convolve3(x, y, ch, uLap);
    }
    setComponent(result, i, value);
  }
  outValue = result;
}`;

const DENSE_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uInput;
uniform sampler2D uWeights;
uniform sampler2D uBias;
uniform int uWidth;
uniform int uInputChannels;
uniform int uOutputChannels;
uniform int uUseBias;
uniform int uRelu;
out vec4 outValue;

const int MAX_INPUT_CHANNELS = 512;

float component(vec4 v, int c) {
  if (c == 0) return v.r;
  if (c == 1) return v.g;
  if (c == 2) return v.b;
  return v.a;
}

void setComponent(inout vec4 v, int c, float value) {
  if (c == 0) v.r = value;
  else if (c == 1) v.g = value;
  else if (c == 2) v.b = value;
  else v.a = value;
}

float readInput(int x, int y, int ch) {
  int group = ch / 4;
  int comp = ch - group * 4;
  return component(texelFetch(uInput, ivec2(x + group * uWidth, y), 0), comp);
}

float readWeight(int outCh, int inCh) {
  return texelFetch(uWeights, ivec2(inCh, outCh), 0).r;
}

void main() {
  ivec2 frag = ivec2(gl_FragCoord.xy);
  int x = frag.x % uWidth;
  int y = frag.y;
  int group = frag.x / uWidth;
  vec4 result = vec4(0.0);
  for (int i = 0; i < 4; i++) {
    int outCh = group * 4 + i;
    if (outCh >= uOutputChannels) continue;
    float value = 0.0;
    for (int inCh = 0; inCh < MAX_INPUT_CHANNELS; inCh++) {
      if (inCh >= uInputChannels) break;
      value += readWeight(outCh, inCh) * readInput(x, y, inCh);
    }
    if (uUseBias == 1) {
      value += texelFetch(uBias, ivec2(outCh, 0), 0).r;
    }
    if (uRelu == 1) {
      value = max(value, 0.0);
    }
    setComponent(result, i, value);
  }
  outValue = result;
}`;

const UPDATE_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform sampler2D uDelta;
uniform int uWidth;
uniform int uChannels;
uniform float uFireRate;
uniform float uSeed;
out vec4 outValue;

float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float component(vec4 v, int c) {
  if (c == 0) return v.r;
  if (c == 1) return v.g;
  if (c == 2) return v.b;
  return v.a;
}

void setComponent(inout vec4 v, int c, float value) {
  if (c == 0) v.r = value;
  else if (c == 1) v.g = value;
  else if (c == 2) v.b = value;
  else v.a = value;
}

void main() {
  ivec2 frag = ivec2(gl_FragCoord.xy);
  int x = frag.x % uWidth;
  int y = frag.y;
  int group = frag.x / uWidth;
  vec4 state = texelFetch(uState, frag, 0);
  vec4 delta = texelFetch(uDelta, frag, 0);
  vec4 nextValue = state;
  for (int i = 0; i < 4; i++) {
    int ch = group * 4 + i;
    if (ch >= uChannels) {
      setComponent(nextValue, i, 0.0);
    } else {
      float mask = hash13(vec3(float(x), float(y), uSeed + float(ch) * 17.0)) <= uFireRate ? 1.0 : 0.0;
      setComponent(nextValue, i, component(state, i) + mask * component(delta, i));
    }
  }
  outValue = nextValue;
}`;

const PAINT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform int uWidth;
uniform int uHeight;
uniform int uChannels;
uniform vec2 uPos;
uniform float uRadius;
uniform int uMode;
uniform int uTarget;
uniform float uSeed;
out vec4 outValue;

float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void setComponent(inout vec4 v, int c, float value) {
  if (c == 0) v.r = value;
  else if (c == 1) v.g = value;
  else if (c == 2) v.b = value;
  else v.a = value;
}

float component(vec4 v, int c) {
  if (c == 0) return v.r;
  if (c == 1) return v.g;
  if (c == 2) return v.b;
  return v.a;
}

void main() {
  ivec2 frag = ivec2(gl_FragCoord.xy);
  int x = frag.x % uWidth;
  int y = frag.y;
  int group = frag.x / uWidth;
  vec2 d = vec2(float(x), float(y)) - uPos;
  vec4 current = texelFetch(uState, frag, 0);
  if (length(d) > uRadius) {
    outValue = current;
    return;
  }
  vec4 painted = vec4(0.0);
  if (uMode == 1) {
    painted = vec4(
      hash13(vec3(float(x), float(y), uSeed + float(group * 4))),
      hash13(vec3(float(x), float(y), uSeed + float(group * 4 + 1))),
      hash13(vec3(float(x), float(y), uSeed + float(group * 4 + 2))),
      hash13(vec3(float(x), float(y), uSeed + float(group * 4 + 3)))
    ) - 0.5;
  }
  outValue = painted;
  for (int i = 0; i < 4; i++) {
    int ch = group * 4 + i;
    if (ch >= uChannels) {
      setComponent(outValue, i, 0.0);
    } else if (uTarget == 1 && ch >= 3) {
      setComponent(outValue, i, component(current, i));
    }
  }
}`;

const DRAW_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform int uWidth;
uniform int uHeight;
uniform int uChannels;
uniform int uDrawMode;
uniform int uGridCols;
uniform int uGridRows;
uniform int uGridChannelOffset;
uniform int uGridChannels;
uniform float uGridSaturation;
in vec2 vUv;
out vec4 outColor;

float readChannel(int x, int y, int ch) {
  int group = ch / 4;
  int comp = ch - group * 4;
  vec4 v = texelFetch(uState, ivec2(x + group * uWidth, y), 0);
  if (comp == 0) return v.r;
  if (comp == 1) return v.g;
  if (comp == 2) return v.b;
  return v.a;
}

void main() {
  vec3 rgb = vec3(0.0);
  if (uDrawMode == 1) {
    int gridX = int(clamp(floor(vUv.x * float(uGridCols)), 0.0, float(uGridCols - 1)));
    int gridYDisplay = int(clamp(floor(vUv.y * float(uGridRows)), 0.0, float(uGridRows - 1)));
    int gridY = uGridRows - 1 - gridYDisplay;
    vec2 tileUv = fract(vec2(vUv.x * float(uGridCols), vUv.y * float(uGridRows)));
    int x = int(clamp(floor(tileUv.x * float(uWidth)), 0.0, float(uWidth - 1)));
    int yDisplay = int(clamp(floor(tileUv.y * float(uHeight)), 0.0, float(uHeight - 1)));
    int y = uHeight - 1 - yDisplay;
    int tile = gridY * uGridCols + gridX;
    int baseChannel = uGridChannelOffset + tile * 3;
    for (int i = 0; i < 3; i++) {
      int ch = baseChannel + i;
      if (ch < uGridChannelOffset + uGridChannels && ch < uChannels) {
        float value = 0.5 + 0.5 * tanh(uGridSaturation * readChannel(x, y, ch));
        if (i == 0) rgb.r = value;
        else if (i == 1) rgb.g = value;
        else rgb.b = value;
      }
    }
  } else {
    int x = int(clamp(floor(vUv.x * float(uWidth)), 0.0, float(uWidth - 1)));
    int yDisplay = int(clamp(floor(vUv.y * float(uHeight)), 0.0, float(uHeight - 1)));
    int y = uHeight - 1 - yDisplay;
    rgb = vec3(readChannel(x, y, 0), readChannel(x, y, 1), readChannel(x, y, 2));
  }
  outColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}`;

function ceilDiv(a, b) {
  return Math.ceil(a / b);
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

function createProgram(gl, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return program;
}

function texture2D(gl, width, height, data = null) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  return tex;
}

function scalarTexture(gl, width, height, data) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
  return tex;
}

function framebuffer(gl, texture) {
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error("Framebuffer is incomplete.");
  }
  return fb;
}

function packState(state, channels, height, width) {
  const groups = ceilDiv(channels, 4);
  const packed = new Float32Array(width * groups * height * 4);
  for (let ch = 0; ch < channels; ch++) {
    const group = Math.floor(ch / 4);
    const comp = ch - group * 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const src = ch * height * width + y * width + x;
        const dst = (y * width * groups + group * width + x) * 4 + comp;
        packed[dst] = state[src];
      }
    }
  }
  return packed;
}

function unpackState(packed, channels, height, width) {
  const groups = ceilDiv(channels, 4);
  const state = new Float32Array(channels * height * width);
  for (let ch = 0; ch < channels; ch++) {
    const group = Math.floor(ch / 4);
    const comp = ch - group * 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const src = (y * width * groups + group * width + x) * 4 + comp;
        const dst = ch * height * width + y * width + x;
        state[dst] = packed[src];
      }
    }
  }
  return state;
}

function tensorView(weights, tensorInfo) {
  return weights.subarray(
    tensorInfo.byteOffset / Float32Array.BYTES_PER_ELEMENT,
    (tensorInfo.byteOffset + tensorInfo.byteLength) / Float32Array.BYTES_PER_ELEMENT,
  );
}

export class NCAWebGLRuntime {
  constructor(canvas, manifest, weights, initialState) {
    this.canvas = canvas;
    this.manifest = manifest;
    this.width = manifest.gridSize[0];
    this.height = manifest.gridSize[1];
    this.channels = manifest.channels;
    this.featureChannels = manifest.featureChannels;
    this.hiddenChannels = manifest.hiddenChannels;
    this.stateGroups = ceilDiv(this.channels, 4);
    this.featureGroups = ceilDiv(this.featureChannels, 4);
    this.hiddenGroups = ceilDiv(this.hiddenChannels, 4);
    this.stepCount = 0;
    this.initialState = initialState;

    if (
      manifest.family !== "NCA" ||
      manifest.activation !== "relu" ||
      !["CIRCULAR", "REPLICATE"].includes(manifest.padding)
    ) {
      throw new Error("This MVP runtime supports only plain relu NCA with CIRCULAR or REPLICATE padding.");
    }

    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float is required for float render targets.");
    }
    this.gl = gl;
    this.programs = {
      perception: createProgram(gl, PERCEPTION_SHADER),
      dense: createProgram(gl, DENSE_SHADER),
      update: createProgram(gl, UPDATE_SHADER),
      paint: createProgram(gl, PAINT_SHADER),
      draw: createProgram(gl, DRAW_SHADER),
    };

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.stateA = this.createTarget(this.width * this.stateGroups, this.height);
    this.stateB = this.createTarget(this.width * this.stateGroups, this.height);
    this.perception = this.createTarget(this.width * this.featureGroups, this.height);
    this.hidden = this.createTarget(this.width * this.hiddenGroups, this.height);
    this.delta = this.createTarget(this.width * this.stateGroups, this.height);

    const tensors = manifest.weights.tensors;
    this.weightTextures = {
      w0: scalarTexture(gl, this.featureChannels, this.hiddenChannels, tensorView(weights, tensors.w0)),
      w1: scalarTexture(gl, this.hiddenChannels, this.channels, tensorView(weights, tensors.w1)),
      b1: scalarTexture(gl, this.channels, 1, tensorView(weights, tensors.b1)),
      gradX: scalarTexture(gl, 3, 3, tensorView(weights, tensors.grad_x)),
      gradY: scalarTexture(gl, 3, 3, tensorView(weights, tensors.grad_y)),
      lap: scalarTexture(gl, 3, 3, tensorView(weights, tensors.lap)),
    };
    this.emptyBias = scalarTexture(gl, 1, 1, new Float32Array([0]));
  }

  createTarget(width, height) {
    const tex = texture2D(this.gl, width, height);
    return { width, height, tex, fb: framebuffer(this.gl, tex) };
  }

  useProgram(program) {
    const gl = this.gl;
    gl.useProgram(program);
    gl.bindVertexArray(this.vao);
  }

  bindTexture(program, name, texture, unit) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, name), unit);
  }

  drawTo(target, program) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fb);
    gl.viewport(0, 0, target.width, target.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  reset() {
    const gl = this.gl;
    const packed = packState(this.initialState, this.channels, this.height, this.width);
    for (const target of [this.stateA, this.stateB]) {
      gl.bindTexture(gl.TEXTURE_2D, target.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, target.width, target.height, gl.RGBA, gl.FLOAT, packed);
    }
    this.stepCount = 0;
  }

  step(fireRateOverride = null) {
    this.runPerception();
    this.runDense(this.perception, this.hidden, this.weightTextures.w0, this.emptyBias, this.featureChannels, this.hiddenChannels, false, true);
    this.runDense(this.hidden, this.delta, this.weightTextures.w1, this.weightTextures.b1, this.hiddenChannels, this.channels, true, false);
    this.runUpdate(fireRateOverride ?? this.manifest.fireRate);
    [this.stateA, this.stateB] = [this.stateB, this.stateA];
    this.stepCount += 1;
  }

  runPerception() {
    const gl = this.gl;
    const program = this.programs.perception;
    this.useProgram(program);
    this.bindTexture(program, "uState", this.stateA.tex, 0);
    this.bindTexture(program, "uGradX", this.weightTextures.gradX, 1);
    this.bindTexture(program, "uGradY", this.weightTextures.gradY, 2);
    this.bindTexture(program, "uLap", this.weightTextures.lap, 3);
    gl.uniform1i(gl.getUniformLocation(program, "uWidth"), this.width);
    gl.uniform1i(gl.getUniformLocation(program, "uHeight"), this.height);
    gl.uniform1i(gl.getUniformLocation(program, "uChannels"), this.channels);
    gl.uniform1i(gl.getUniformLocation(program, "uOutGroups"), this.featureGroups);
    gl.uniform1i(gl.getUniformLocation(program, "uPaddingMode"), this.manifest.padding === "REPLICATE" ? 1 : 0);
    this.drawTo(this.perception, program);
  }

  runDense(input, output, weights, bias, inputChannels, outputChannels, useBias, relu) {
    const gl = this.gl;
    const program = this.programs.dense;
    this.useProgram(program);
    this.bindTexture(program, "uInput", input.tex, 0);
    this.bindTexture(program, "uWeights", weights, 1);
    this.bindTexture(program, "uBias", bias, 2);
    gl.uniform1i(gl.getUniformLocation(program, "uWidth"), this.width);
    gl.uniform1i(gl.getUniformLocation(program, "uInputChannels"), inputChannels);
    gl.uniform1i(gl.getUniformLocation(program, "uOutputChannels"), outputChannels);
    gl.uniform1i(gl.getUniformLocation(program, "uUseBias"), useBias ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(program, "uRelu"), relu ? 1 : 0);
    this.drawTo(output, program);
  }

  runUpdate(fireRate) {
    const gl = this.gl;
    const program = this.programs.update;
    this.useProgram(program);
    this.bindTexture(program, "uState", this.stateA.tex, 0);
    this.bindTexture(program, "uDelta", this.delta.tex, 1);
    gl.uniform1i(gl.getUniformLocation(program, "uWidth"), this.width);
    gl.uniform1i(gl.getUniformLocation(program, "uChannels"), this.channels);
    gl.uniform1f(gl.getUniformLocation(program, "uFireRate"), fireRate);
    gl.uniform1f(gl.getUniformLocation(program, "uSeed"), this.stepCount + 1);
    this.drawTo(this.stateB, program);
  }

  paint(x, y, radius, mode, target = "all") {
    const gl = this.gl;
    const program = this.programs.paint;
    this.useProgram(program);
    this.bindTexture(program, "uState", this.stateA.tex, 0);
    gl.uniform1i(gl.getUniformLocation(program, "uWidth"), this.width);
    gl.uniform1i(gl.getUniformLocation(program, "uHeight"), this.height);
    gl.uniform1i(gl.getUniformLocation(program, "uChannels"), this.channels);
    gl.uniform2f(gl.getUniformLocation(program, "uPos"), x, y);
    gl.uniform1f(gl.getUniformLocation(program, "uRadius"), radius);
    gl.uniform1i(gl.getUniformLocation(program, "uMode"), mode === "random" ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(program, "uTarget"), target === "rgb" ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(program, "uSeed"), Math.random() * 1000);
    this.drawTo(this.stateB, program);
    [this.stateA, this.stateB] = [this.stateB, this.stateA];
  }

  draw(options = {}) {
    const gl = this.gl;
    const program = this.programs.draw;
    const drawMode = options.mode === "hidden-grid" ? 1 : 0;
    const gridChannelOffset = options.gridChannelOffset ?? 3;
    const gridChannels = Math.max(0, Math.min(
      options.gridChannels ?? (this.channels - gridChannelOffset),
      this.channels - gridChannelOffset,
    ));
    const tileCount = Math.max(1, Math.ceil(gridChannels / 3));
    const gridCols = options.gridCols ?? Math.ceil(Math.sqrt(tileCount));
    const gridRows = options.gridRows ?? Math.ceil(tileCount / gridCols);
    const gridSaturation = options.gridSaturation ?? 1.0;
    this.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.bindTexture(program, "uState", this.stateA.tex, 0);
    gl.uniform1i(gl.getUniformLocation(program, "uWidth"), this.width);
    gl.uniform1i(gl.getUniformLocation(program, "uHeight"), this.height);
    gl.uniform1i(gl.getUniformLocation(program, "uChannels"), this.channels);
    gl.uniform1i(gl.getUniformLocation(program, "uDrawMode"), drawMode);
    gl.uniform1i(gl.getUniformLocation(program, "uGridCols"), gridCols);
    gl.uniform1i(gl.getUniformLocation(program, "uGridRows"), gridRows);
    gl.uniform1i(gl.getUniformLocation(program, "uGridChannelOffset"), gridChannelOffset);
    gl.uniform1i(gl.getUniformLocation(program, "uGridChannels"), gridChannels);
    gl.uniform1f(gl.getUniformLocation(program, "uGridSaturation"), gridSaturation);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  readState() {
    const gl = this.gl;
    const packed = new Float32Array(this.width * this.stateGroups * this.height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.stateA.fb);
    gl.readPixels(0, 0, this.width * this.stateGroups, this.height, gl.RGBA, gl.FLOAT, packed);
    return unpackState(packed, this.channels, this.height, this.width);
  }

  validate(reference) {
    this.reset();
    const steps = this.manifest.validation.referenceSteps;
    for (let i = 0; i < steps; i++) {
      this.step(1.0);
    }
    const actual = this.readState();
    let maxAbsError = 0.0;
    for (let i = 0; i < reference.length; i++) {
      maxAbsError = Math.max(maxAbsError, Math.abs(actual[i] - reference[i]));
    }
    return { maxAbsError, actualLength: actual.length, referenceLength: reference.length };
  }
}
