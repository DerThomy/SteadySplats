// Code shared by the WebGPU renderer (viewer.js, shaders.js) and the WebGL 2 fallback (webgl.js, shaders_gl.js):
// constants, small vector and matrix helpers, and the per-frame state with its uniform block.

export const NEAR = 0.1;           // clip planes of the reference renderer's camera paths
export const FAR = 2000.0;
export const REST_FRAMES_MAX = 1024;    // frames averaged while the camera rests; after that the viewer idles
export const FLAG_SPATIAL = 1, FLAG_TEMPORAL = 2, FLAG_WHITE_NOISE = 4, FLAG_RESET = 8;
export const K_EMPTY = [0x3fffffff, 0xffff0000];  // empty K-buffer entry (see shaders.js)

// =====================================================================================================================
// Small vector and matrix helpers (column-major 4x4 matrices as Float32Array)
// =====================================================================================================================

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export function mat4Mul(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
    }
    return out;
}

export function mat4Invert(m) {
    const inv = new Float32Array(16);
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    det = det === 0 ? 1 : 1 / det;
    for (let i = 0; i < 16; i++) inv[i] *= det;
    return inv;
}

export const sameMatrix = (a, b) => !!b && a.every((v, i) => v === b[i]);

// World -> camera for a camera at `position` with world-space axes right, down, forward (COLMAP convention)
export function viewMatrix(position, right, down, forward) {
    const m = new Float32Array(16);
    const rows = [right, down, forward];
    for (let r = 0; r < 3; r++) {
        m[0 * 4 + r] = rows[r][0];
        m[1 * 4 + r] = rows[r][1];
        m[2 * 4 + r] = rows[r][2];
        m[3 * 4 + r] = -dot(rows[r], position);
    }
    m[15] = 1;
    return m;
}

// Camera -> clip for pixel focal lengths and a centered principal point: x right, y up, depth in [0, 1]
export function projectionMatrix(fx, fy, width, height) {
    const m = new Float32Array(16);
    m[0] = (2 * fx) / width;
    m[5] = (-2 * fy) / height;
    m[10] = FAR / (FAR - NEAR);
    m[11] = 1;
    m[14] = (-FAR * NEAR) / (FAR - NEAR);
    return m;
}

// STBN Cranley-Patterson shift of the 64-frame cycle, computed like the reference renderer (u32 arithmetic)
export function stbnShift(frameIdx) {
    const cycle = Math.floor(frameIdx / 64);
    let u0 = 0x9e3779b9;
    u0 = (u0 + (u0 << 10)) >>> 0;
    u0 = (u0 ^ (u0 >>> 6)) >>> 0;
    u0 = (u0 + (u0 << 3)) >>> 0;
    u0 = (u0 ^ (u0 >>> 11)) >>> 0;
    u0 = (u0 + (u0 << 15)) >>> 0;
    const shift = (u0 + Math.imul(cycle, 0x9e3779b9)) >>> 0;
    return Math.fround(shift / 4294967295);
}

export function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    if (e === 0) return s * m * 5.960464477539063e-8;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
}

// =====================================================================================================================
// Per-frame state shared by both renderers
// =====================================================================================================================

export class RendererBase {
    constructor() {
        this.frameIdx = 0;
        this.parity = 0;
        this.prevViewProj = null;
        this.resetPending = true;
        this.flags = FLAG_SPATIAL | FLAG_TEMPORAL;
        this.converge = true;     // average the frames while the camera rests (temporal mode)
        this.restFrames = 0;
        this.restLimit = REST_FRAMES_MAX;
        this.size = [0, 0];
        this.scene = null;
        this.gpuTimes = null;
    }

    // The per-frame uniforms (struct Frame, 384 bytes; std140 in GLSL); `cam` gives world-space axes and the focal
    // length at the render height. Returns null if there is nothing to render: the camera has rested long enough for
    // the average to converge.
    frameData(cam, fx, fy) {
        const [w, h] = this.size;
        const view = viewMatrix(cam.position, cam.right, cam.down, cam.forward);
        const proj = projectionMatrix(fx, fy, w, h);
        const viewProj = mat4Mul(proj, view);
        const reset = this.resetPending || !this.prevViewProj;
        const still = !reset && sameMatrix(view, this.prevView) && sameMatrix(proj, this.prevProj);
        this.restFrames = still ? this.restFrames + 1 : 0;
        if (this.converge && (this.flags & FLAG_TEMPORAL) && this.restFrames > this.restLimit) return null;
        const data = new ArrayBuffer(384);
        const f = new Float32Array(data);
        const u = new Uint32Array(data);
        f.set(view, 0);
        f.set(proj, 16);
        f.set(mat4Invert(view), 32);
        f.set(mat4Invert(proj), 48);
        f.set(reset ? viewProj : this.prevViewProj, 64);
        f.set([cam.position[0], cam.position[1], cam.position[2], 1], 80);
        f.set([w, h, fx, fy], 84);
        u[88] = this.scene.header.count;
        u[89] = this.frameIdx >>> 0;
        f[90] = stbnShift(this.frameIdx);
        u[91] = this.flags | (reset ? FLAG_RESET : 0);
        f[92] = this.scene.header.shRange / this.scene.header.shSteps;
        u[93] = (Math.random() * 4294967295) >>> 0;
        u[94] = this.converge ? this.restFrames : 0;
        this.prevViewProj = viewProj;
        this.prevView = view;
        this.prevProj = proj;
        this.resetPending = false;
        return data;
    }
}
