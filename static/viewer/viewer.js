// SteadySplats viewer: stochastic rendering of 3D Gaussians at 1 sample per pixel with per-primitive STBN,
// history-based spatial reuse and temporal accumulation (see shaders.js). Renders with WebGPU, or with the WebGL 2
// fallback (webgl.js) where WebGPU is missing, or shows the project page's teaser video. Scenes are .ssplat files written
// by convert_ply.js (notes folder, viewer-tools).
import { PREPROCESS, PREPROCESS_FLOAT, RASTER, REUSE, DISPLAY } from './shaders.js';
import { FLAG_SPATIAL, FLAG_TEMPORAL, FLAG_WHITE_NOISE, K_EMPTY, RendererBase, add, cross, dot, halfToFloat, normalize,
    scale3, sub } from './common.js';
import { WebGLRenderer } from './webgl.js';

const SCENES = [
    { id: 'counter', name: 'Counter' },
    { id: 'kitchen', name: 'Kitchen' },
    { id: 'room', name: 'Room' },
    { id: 'bonsai', name: 'Bonsai' },
    { id: 'garden', name: 'Garden' },
];
const SCENE_DIR = 'static/viewer/scenes/';
// The same scenes trained with and without our variance loss, with the same number of Gaussians: <scene>.ssplat and
// <scene>-lvar0.ssplat
const MODELS = {
    lvar1: { suffix: '', hint: 'Trained with our variance loss (\u03bbvar\u00a0=\u00a01).' },
    lvar0: { suffix: '-lvar0', hint: 'The same training without the variance loss (\u03bbvar\u00a0=\u00a00) and with as many Gaussians: noisier at 1 sample per pixel.' },
};
const SCENE_CACHE_SIZE = 2;  // decoded scenes kept, so that switching back and forth needs no download
const BACKEND_KEY = 'steadysplats-viewer-backend';  // 'webgl' after WebGPU stalled or crashed in this browser
const PANEL_KEY = 'steadysplats-viewer-panel';      // 'open' or 'closed': the controls as the viewer left them
const STALL_MS = 5000;      // a WebGPU frame unfinished for this long counts as a stall
const MAX_IN_FLIGHT = 2;    // WebGPU frames queued at most, so that a slow GPU cannot pile up work
const STBN_URL = 'static/stbn/stbn_scalar.js';

const params = new URLSearchParams(location.search);
const TEST = params.has('test');
// Test mode can load any scene file, e.g. a validation build with the reference renderer's splat order
if (TEST && params.get('scene') && !SCENES.some((s) => s.id === params.get('scene'))) {
    SCENES.push({ id: params.get('scene'), name: params.get('scene'), custom: true });
}

// =====================================================================================================================
// Scene loading
// =====================================================================================================================

async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length')) || 0;
    const reader = res.body.getReader();
    const parts = [];
    let received = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        received += value.length;
        if (total) onProgress(received / total);
    }
    const out = new Uint8Array(received);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
}

async function loadScene(id, onProgress) {
    const file = await fetchWithProgress(`${SCENE_DIR}${id}.ssplat`, (f) => onProgress(0.85 * f, 'Downloading'));
    if (String.fromCharCode(...file.subarray(0, 4)) !== 'SSPL') throw new Error('not an .ssplat file');
    const headerLen = new DataView(file.buffer).getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(file.subarray(8, 8 + headerLen)).replace(/\0+$/, ''));
    onProgress(0.88, 'Decompressing');
    const stream = new Blob([file.subarray(8 + headerLen)]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    const payload = new Uint8Array(await new Response(stream).arrayBuffer());
    onProgress(0.95, 'Preparing');
    return { header, ...decodeScene(header, payload) };
}

// Interleaves the streams into 7 words per Gaussian for the GPU (layout in shaders.js, PREPROCESS)
function decodeScene(header, payload) {
    const n = header.count;
    const s = Object.fromEntries(header.streams.map((st) => [st.name, payload.subarray(st.offset, st.offset + st.bytes)]));
    if (header.precision === 'float') {
        // Lossless validation build: 23 floats per Gaussian, used as they are
        return { words: new Uint32Array(s.float.buffer.slice(s.float.byteOffset, s.float.byteOffset + s.float.byteLength)), chunks: new Float32Array(12) };
    }
    const chunks = new Float32Array(s.chunks.buffer.slice(s.chunks.byteOffset, s.chunks.byteOffset + s.chunks.byteLength));
    const pos = new Uint16Array(n * 3);
    for (let d = 0; d < 3; d++) {
        const lo = s.position.subarray((2 * d) * n, (2 * d + 1) * n);
        const hi = s.position.subarray((2 * d + 1) * n, (2 * d + 2) * n);
        let prev = 0;
        for (let k = 0; k < n; k++) {
            if (k % header.chunkSize === 0) prev = 0;
            prev = (prev + (lo[k] | (hi[k] << 8))) & 0xffff;
            pos[k * 3 + d] = prev;
        }
    }
    const r = s.rotation;
    const sc = s.scale, op = s.opacity, col = s.color, sh = s.sh1;
    const words = new Uint32Array(n * 7);
    for (let k = 0; k < n; k++) {
        const w = k * 7, p = k * 3, h = k * 9;
        words[w] = pos[p] | (pos[p + 1] << 16);
        words[w + 1] = (pos[p + 2] | (sc[p] << 16) | (sc[p + 1] << 24)) >>> 0;
        words[w + 2] = (sc[p + 2] | (op[k] << 8) | (col[p] << 16) | (col[p + 1] << 24)) >>> 0;
        words[w + 3] = (col[p + 2] | (sh[h] << 8) | (sh[h + 1] << 16) | (sh[h + 2] << 24)) >>> 0;
        words[w + 4] = (sh[h + 3] | (sh[h + 4] << 8) | (sh[h + 5] << 16) | (sh[h + 6] << 24)) >>> 0;
        words[w + 5] = sh[h + 7] | (sh[h + 8] << 8);
        words[w + 6] = (r[k] | (r[n + k] << 8) | (r[2 * n + k] << 16) | (r[3 * n + k] << 24)) >>> 0;
    }
    return { words, chunks };
}

function loadStbn() {
    return new Promise((resolve, reject) => {
        const finish = () => {
            const bin = atob(window.STBN_DATA.base64);
            const data = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
            resolve(data);
        };
        if (window.STBN_DATA) { finish(); return; }
        const script = document.createElement('script');
        script.src = STBN_URL;
        script.onload = finish;
        script.onerror = () => reject(new Error(`cannot load ${STBN_URL}`));
        document.head.appendChild(script);
    });
}

// =====================================================================================================================
// Camera: orbit around a target, with the scene's up direction
// =====================================================================================================================

class OrbitCamera {
    constructor() {
        this.up = [0, 0, 1];
        this.target = [0, 0, 0];
        this.distance = 3;
        this.yaw = 0;
        this.pitch = 0;
        this.fy = 1000;           // focal length in pixels at the reference height below
        this.refHeight = 1000;
        this.pose = null;         // exact pose (test mode) instead of the orbit
    }

    setUp(up) {
        this.up = normalize(up);
        const ref = Math.abs(this.up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        this.h0 = normalize(cross(this.up, ref));
        this.h1 = cross(this.up, this.h0);
    }

    // Starts from a dataset camera (position, camera-to-world rotation rows) looking at the scene center's depth
    setFromCamera(cam, center) {
        const forward = [cam.rotation[0][2], cam.rotation[1][2], cam.rotation[2][2]];
        this.distance = Math.max(0.3, dot(sub(center, cam.position), forward));
        this.target = add(cam.position, scale3(forward, this.distance));
        const back = scale3(forward, -1);
        this.pitch = Math.asin(Math.max(-1, Math.min(1, dot(back, this.up))));
        this.yaw = Math.atan2(dot(back, this.h1), dot(back, this.h0));
        this.fy = cam.fy;
        this.refHeight = cam.height;
        this.pose = null;
    }

    axes() {
        if (this.pose) {
            const R = this.pose.rotation;
            return {
                position: this.pose.position,
                right: [R[0][0], R[1][0], R[2][0]], down: [R[0][1], R[1][1], R[2][1]], forward: [R[0][2], R[1][2], R[2][2]],
            };
        }
        const cp = Math.cos(this.pitch);
        const back = add(add(scale3(this.h0, cp * Math.cos(this.yaw)), scale3(this.h1, cp * Math.sin(this.yaw))), scale3(this.up, Math.sin(this.pitch)));
        const position = add(this.target, scale3(back, this.distance));
        const forward = scale3(back, -1);
        const right = normalize(cross(forward, this.up));
        const down = cross(forward, right);
        return { position, right, down, forward };
    }

    orbit(dx, dy) {
        this.yaw -= dx * 0.005;
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.005));
    }

    pan(dx, dy, viewHeight) {
        const { right, down } = this.axes();
        const s = this.distance / (this.fy * (viewHeight / this.refHeight));
        this.target = add(this.target, add(scale3(right, -dx * s), scale3(down, -dy * s)));
    }

    dolly(factor) {
        this.distance = Math.max(0.05, Math.min(200, this.distance * factor));
    }

    move(forwardAmount, rightAmount, upAmount) {
        const { right, forward } = this.axes();
        const d = add(add(scale3(forward, forwardAmount), scale3(right, rightAmount)), scale3(this.up, upAmount));
        this.target = add(this.target, scale3(d, this.distance));
    }
}

// =====================================================================================================================
// Renderer
// =====================================================================================================================

class WebGPURenderer extends RendererBase {
    constructor(device, context, format, stbnData) {
        super();
        this.backend = 'WebGPU';
        this.maxPixels = 2560 * 1600;  // render resolution cap (K-buffers and the rest average need 80 bytes per pixel)
        this.device = device;
        this.context = context;
        this.format = format;
        this.maxInFlight = MAX_IN_FLIGHT;
        this.pending = [];  // submit times of frames the GPU has not finished

        this.frameBuffer = device.createBuffer({ size: 384, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.drawArgs = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.stbn = device.createTexture({ size: [128, 128, 64], dimension: '3d', format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: this.stbn }, stbnData, { bytesPerRow: 128, rowsPerImage: 128 }, [128, 128, 64]);
        this.linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

        const module = (code) => device.createShaderModule({ code });
        this.preprocessPipelines = {
            packed: device.createComputePipeline({ layout: 'auto', compute: { module: module(PREPROCESS), entryPoint: 'main' } }),
            float: device.createComputePipeline({ layout: 'auto', compute: { module: module(PREPROCESS_FLOAT), entryPoint: 'main' } }),
        };
        const raster = module(RASTER);
        this.rasterPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: raster, entryPoint: 'vs' },
            fragment: { module: raster, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'r32uint' }] },
            primitive: { topology: 'triangle-strip', cullMode: 'none' },
            depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        });
        this.reusePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: module(REUSE), entryPoint: 'main' } });
        const display = module(DISPLAY);
        this.displayPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: display, entryPoint: 'vs' },
            fragment: { module: display, entryPoint: 'fs', targets: [{ format }] },
            primitive: { topology: 'triangle-list' },
        });

        // GPU timing per pass, if available
        this.timing = device.features.has('timestamp-query');
        if (this.timing) {
            this.querySet = device.createQuerySet({ type: 'timestamp', count: 8 });
            this.queryResolve = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            this.queryRead = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            this.queryBusy = false;
            this.gpuTimes = null;
        }
    }

    setScene(scene) {
        const d = this.device;
        const n = scene.header.count;
        const limit = Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize);
        if (Math.max(scene.words.byteLength, n * 32) > limit) {
            throw new Error(`it needs ${Math.ceil(n * 32 / 2 ** 20)} MB per GPU buffer, and WebGPU offers ${Math.floor(limit / 2 ** 20)} MB on this device. The WebGL 2 fallback (?backend=webgl) can show it.`);
        }
        for (const b of [this.splatBuffer, this.chunkBuffer, this.projected, this.visible]) if (b) b.destroy();
        this.scene = scene;
        this.splatBuffer = d.createBuffer({ size: scene.words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(this.splatBuffer, 0, scene.words);
        this.chunkBuffer = d.createBuffer({ size: scene.chunks.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        d.queue.writeBuffer(this.chunkBuffer, 0, scene.chunks);
        this.projected = d.createBuffer({ size: n * 32, usage: GPUBufferUsage.STORAGE });
        this.visible = d.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
        this.preprocessPipeline = this.preprocessPipelines[scene.header.precision === 'float' ? 'float' : 'packed'];
        this.preprocessGroup = d.createBindGroup({
            layout: this.preprocessPipeline.getBindGroupLayout(0),
            entries: [this.frameBuffer, this.splatBuffer, this.chunkBuffer, this.projected, this.visible, this.drawArgs]
                .map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        this.rasterGroup = d.createBindGroup({
            layout: this.rasterPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.frameBuffer } },
                { binding: 1, resource: { buffer: this.projected } },
                { binding: 2, resource: { buffer: this.visible } },
                { binding: 3, resource: this.stbn.createView() },
            ],
        });
        if (this.size[0]) this.createTargets(this.size[0], this.size[1]);
        this.resetPending = true;
    }

    resize(width, height) {
        if (width === this.size[0] && height === this.size[1]) return;
        this.size = [width, height];
        if (this.scene) this.createTargets(width, height);
        this.resetPending = true;
    }

    createTargets(width, height) {
        const d = this.device;
        for (const t of [this.colorTex, this.idTex, this.depthTex, this.restAccum, ...(this.hist || []), ...(this.kbuf || [])]) if (t) t.destroy();
        const tex = (format, usage) => d.createTexture({ size: [width, height], format, usage });
        const RA = GPUTextureUsage.RENDER_ATTACHMENT, TB = GPUTextureUsage.TEXTURE_BINDING;
        this.colorTex = tex('rgba8unorm', RA | TB);
        this.idTex = tex('r32uint', RA | TB);
        this.depthTex = tex('depth32float', RA | TB);
        this.hist = [0, 1].map(() => tex('rgba16float', GPUTextureUsage.STORAGE_BINDING | TB | GPUTextureUsage.COPY_SRC));
        const kBytes = width * height * 4 * 8;
        this.kbuf = [0, 1].map(() => {
            const b = d.createBuffer({ size: kBytes, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
            const u = new Uint32Array(b.getMappedRange());
            for (let i = 0; i < u.length; i += 2) { u[i] = K_EMPTY[0]; u[i + 1] = K_EMPTY[1]; }
            b.unmap();
            return b;
        });
        this.restAccum = d.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE });
        this.reuseGroups = [0, 1].map((p) => d.createBindGroup({
            layout: this.reusePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.frameBuffer } },
                { binding: 1, resource: this.colorTex.createView() },
                { binding: 2, resource: this.idTex.createView() },
                { binding: 3, resource: this.depthTex.createView() },
                { binding: 4, resource: { buffer: this.projected } },
                { binding: 5, resource: { buffer: this.kbuf[p] } },
                { binding: 6, resource: { buffer: this.kbuf[1 - p] } },
                { binding: 7, resource: this.hist[p].createView() },
                { binding: 8, resource: this.linear },
                { binding: 9, resource: this.hist[1 - p].createView() },
                { binding: 10, resource: { buffer: this.restAccum } },
            ],
        }));
        this.displayGroups = [0, 1].map((p) => d.createBindGroup({
            layout: this.displayPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: this.hist[1 - p].createView() }, { binding: 1, resource: this.linear }],
        }));
    }

    // Renders a frame; returns false if there is nothing to render (converged at rest, see RendererBase.frameData)
    render(cam, fx, fy, toCanvas = true) {
        const d = this.device;
        const [w, h] = this.size;
        if (this.pending.length >= this.maxInFlight) return false;  // the GPU is still busy with earlier frames
        const frame = this.frameData(cam, fx, fy);
        if (!frame) return false;
        d.queue.writeBuffer(this.frameBuffer, 0, frame);
        d.queue.writeBuffer(this.drawArgs, 0, new Uint32Array([4, 0, 0, 0]));
        const enc = d.createCommandEncoder();
        const ts = (i) => (this.timing ? { querySet: this.querySet, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 } : undefined);

        const pre = enc.beginComputePass({ timestampWrites: ts(0) });
        pre.setPipeline(this.preprocessPipeline);
        pre.setBindGroup(0, this.preprocessGroup);
        pre.dispatchWorkgroups(Math.ceil(this.scene.header.count / 256));
        pre.end();

        const rp = enc.beginRenderPass({
            colorAttachments: [
                { view: this.colorTex.createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' },
                { view: this.idTex.createView(), clearValue: [4294967295, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
            ],
            depthStencilAttachment: { view: this.depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
            timestampWrites: ts(2),
        });
        rp.setPipeline(this.rasterPipeline);
        rp.setBindGroup(0, this.rasterGroup);
        rp.drawIndirect(this.drawArgs, 0);
        rp.end();

        const reuse = enc.beginComputePass({ timestampWrites: ts(4) });
        reuse.setPipeline(this.reusePipeline);
        reuse.setBindGroup(0, this.reuseGroups[this.parity]);
        reuse.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        reuse.end();

        if (toCanvas) {
            const dp = enc.beginRenderPass({
                colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
                timestampWrites: ts(6),
            });
            dp.setPipeline(this.displayPipeline);
            dp.setBindGroup(0, this.displayGroups[this.parity]);
            dp.draw(3);
            dp.end();
        }

        const readTiming = this.timing && !this.queryBusy;
        if (readTiming) {
            enc.resolveQuerySet(this.querySet, 0, toCanvas ? 8 : 6, this.queryResolve, 0);
            enc.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, 64);
        }
        d.queue.submit([enc.finish()]);
        this.pending.push(performance.now());
        d.queue.onSubmittedWorkDone().then(() => this.pending.shift());
        if (readTiming) {
            this.queryBusy = true;
            this.queryRead.mapAsync(GPUMapMode.READ).then(() => {
                const t = new BigInt64Array(this.queryRead.getMappedRange().slice(0));
                this.queryRead.unmap();
                const ms = (a, b) => Number(t[b] - t[a]) / 1e6;
                this.gpuTimes = { preprocess: ms(0, 1), raster: ms(2, 3), reuse: ms(4, 5), total: ms(0, toCanvas ? 7 : 5) };
                this.queryBusy = false;
            }).catch(() => { this.queryBusy = false; });
        }
        this.outputTexture = this.hist[1 - this.parity];
        this.parity ^= 1;
        this.frameIdx++;
        return true;
    }

    // Test helper: the last output as 8-bit RGBA, rows top to bottom
    async readOutput() {
        const [w, h] = this.size;
        const bytesPerRow = Math.ceil((w * 8) / 256) * 256;
        const buf = this.device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = this.device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: this.outputTexture }, { buffer: buf, bytesPerRow }, [w, h]);
        this.device.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        const half = new Uint16Array(buf.getMappedRange());
        const out = new Uint8Array(w * h * 4);
        const rowHalves = bytesPerRow / 2;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w * 4; x++) {
                const v = halfToFloat(half[y * rowHalves + x]);
                out[(y * w) * 4 + x] = (x & 3) === 3 ? 255 : Math.max(0, Math.min(255, Math.round(v * 255)));
            }
        }
        buf.unmap();
        buf.destroy();
        return out;
    }

    done() {
        return this.device.queue.onSubmittedWorkDone();
    }

    // True if the oldest unfinished frame has been on the GPU for longer than STALL_MS
    stalled(now) {
        return this.pending.length > 0 && now - this.pending[0] > STALL_MS;
    }

    async visibleCount() {
        const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.drawArgs, 0, buf, 0, 16);
        this.device.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        const v = new Uint32Array(buf.getMappedRange())[1];
        buf.unmap();
        buf.destroy();
        return v;
    }
}

// =====================================================================================================================
// Page
// =====================================================================================================================

const $ = (id) => document.getElementById(id);

// WebGPU if it works, else the WebGL 2 fallback (?backend=webgl forces it); without either, the reasons
async function createRenderer(canvas, stbn) {
    const issues = [];
    const backend = params.get('backend');
    if (backend === 'webgpu') storeBackend(null);  // retry WebGPU after an earlier stall or crash
    const why = { stall: 'WebGPU stopped responding on this GPU.', lost: 'The WebGPU device was lost.' }[params.get('why')];
    if (backend === 'webgl') {
        if (why) issues.push(why);
    } else if (backend !== 'webgpu' && loadBackend() === 'webgl') {
        issues.push('WebGPU stalled or crashed in this browser before.');
    } else {
        const r = await createWebGPURenderer(canvas, stbn);
        if (typeof r !== 'string') return { renderer: r, issues };
        issues.push(r);
    }
    const r = WebGLRenderer.create(canvas, stbn, backend === 'webgl' && !params.get('why'));
    if (typeof r !== 'string') {
        canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); showError('The WebGL context was lost. Reload the page.'); });
        return { renderer: r, issues };
    }
    issues.push(r);
    return { renderer: null, issues };
}

// What the user can do: enable WebGPU on Linux (off by default for most GPUs so far), or let the browser use the GPU
function hints(issues) {
    if (issues.some((i) => /software/.test(i))) {
        return ' The browser seems to run without GPU acceleration: check chrome://gpu, the setting "Use graphics'
            + ' acceleration when available", and the GPU driver (for WebGPU on Linux, its Vulkan driver).';
    }
    const linux = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);
    return linux && window.isSecureContext && !navigator.gpu
        ? ' On Linux, Chrome offers WebGPU after enabling chrome://flags/#enable-unsafe-webgpu and #enable-vulkan.' : '';
}

// The page elements viewer.js needs: an outdated or cached viewer.html lacks some of them
const REQUIRED_IDS = ['viewer-canvas', 'panel-open', 'panel-close', 'fullscreen', 'scene-buttons', 'model-hint', 'render-scale', 'view-prev', 'view-next', 'converge', 'method-hint', 'stat-backend',
    'stat-fps', 'stat-gpu', 'stat-res', 'stat-count', 'stat-rest', 'backend-note', 'loading', 'loading-label',
    'loading-fill', 'error', 'fallback'];

// The WebGPU renderer, or why WebGPU cannot be used here
async function createWebGPURenderer(canvas, stbn) {
    if (!navigator.gpu) {
        // Browsers offer WebGPU only on secure pages (HTTPS or localhost)
        return window.isSecureContext ? 'This browser does not offer WebGPU.'
            : `WebGPU only works on secure pages (HTTPS or localhost), and this page came over plain HTTP from ${location.host}.`;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
    if (!adapter) return 'WebGPU found no usable GPU (hardware acceleration off, or the GPU or driver blocked).';
    const info = adapter.info || {};
    console.info('WebGPU adapter:', info.vendor, info.architecture, info.description, info.isFallbackAdapter ? '(software)' : '');
    // A software adapter (SwiftShader, llvmpipe) takes seconds per frame; WebGL 2 often still has the GPU
    if (info.isFallbackAdapter || adapter.isFallbackAdapter) return 'WebGPU only has a software renderer here, so the browser does not use the GPU for it.';
    if (adapter.limits.maxComputeWorkgroupStorageSize < 20 * 1024 + 512) return 'This GPU offers too little WebGPU workgroup memory.';
    let device;
    try {
        device = await adapter.requestDevice({
            requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
            requiredLimits: {
                maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
            },
        });
    } catch (err) {
        return `WebGPU could not create a device (${err.message}).`;
    }
    device.lost.then((lost) => {
        console.warn('WebGPU device lost:', lost.reason, lost.message);
        if (lost.reason !== 'destroyed') switchToWebGL('lost');
    });
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const renderer = new WebGPURenderer(device, context, format, stbn);
    renderer.adapterInfo = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ');
    return renderer;
}

// WebGPU stalled or crashed: reload with the WebGL 2 fallback, and keep using it in this browser
function switchToWebGL(why) {
    if (TEST) { showError(`WebGPU ${why === 'lost' ? 'device was lost' : 'stalled'}.`); return; }
    storeBackend('webgl');
    const next = new URLSearchParams(location.search);
    next.set('backend', 'webgl');
    next.set('why', why);
    location.replace(`${location.pathname}?${next}`);
}

function loadBackend() {
    try { return localStorage.getItem(BACKEND_KEY); } catch { return null; }
}

function storeBackend(value) {
    try {
        if (value) localStorage.setItem(BACKEND_KEY, value);
        else localStorage.removeItem(BACKEND_KEY);
    } catch {
        // storage unavailable: the choice holds for this page only
    }
}

async function main() {
    const missing = REQUIRED_IDS.filter((id) => !$(id));
    if (missing.length) {
        const message = `This viewer.html does not match static/viewer/viewer.js (missing #${missing.join(', #')}). `
            + 'Update both, then reload without the cache (Ctrl+Shift+R).';
        console.error(message);
        if ($('error')) showError(message);
        else document.body.textContent = message;
        return;
    }
    const canvas = $('viewer-canvas');
    const { renderer, issues } = await createRenderer(canvas, await loadStbn());
    if (!renderer) { showFallback(issues); return; }
    $('stat-backend').textContent = renderer.backend;
    if (renderer.adapterInfo) $('stat-backend').title = renderer.adapterInfo;
    if (issues.length) {
        // Running on the fallback: say why, and offer WebGPU again if it only failed at run time
        const note = $('backend-note');
        note.textContent = `${issues[0]} The viewer uses its WebGL 2 fallback, which is slower.${hints(issues)} `;
        if (navigator.gpu && (params.get('why') || loadBackend() === 'webgl')) {
            const retry = document.createElement('a');
            const next = new URLSearchParams(location.search);
            next.set('backend', 'webgpu');
            next.delete('why');
            retry.href = `?${next}`;
            retry.textContent = 'Try WebGPU again';
            note.append(retry);
        }
        note.hidden = false;
    }

    const camera = new OrbitCamera();
    let renderScale = 1;
    let sceneIndex = Math.max(0, SCENES.findIndex((s) => s.id === params.get('scene')));
    let model = MODELS[params.get('model')] ? params.get('model') : 'lvar1';
    const sceneCache = new Map();
    let viewIndex = 0;
    let loading = null;

    // Loads scene `index` with the selected model; a model switch keeps the camera, so both can be compared in one view
    async function selectScene(index, keepView = false) {
        sceneIndex = index;
        const info = SCENES[index];
        document.querySelectorAll('[data-scene]').forEach((b) => b.classList.toggle('active', b.dataset.scene === info.id));
        document.querySelectorAll('[data-model]').forEach((b) => b.classList.toggle('active', b.dataset.model === model));
        $('model-hint').textContent = MODELS[model].hint + ' Colors reduced to SH degree 1 and compressed for the web.';
        const file = info.custom ? info.id : info.id + MODELS[model].suffix;
        const token = {};
        loading = token;
        try {
            let scene = sceneCache.get(file);
            if (!scene) {
                setLoading(0, 'Downloading');
                scene = await loadScene(file, (f, label) => { if (loading === token) setLoading(f, label); });
                sceneCache.set(file, scene);
                while (sceneCache.size > SCENE_CACHE_SIZE) sceneCache.delete(sceneCache.keys().next().value);
            }
            if (loading !== token) return;
            renderer.setScene(scene);
            if (!keepView) {
                camera.setUp(scene.header.up);
                viewIndex = 0;
                camera.setFromCamera(scene.header.cameras[0], scene.header.center);
            }
            $('stat-count').textContent = `${(scene.header.count / 1e6).toFixed(2)} M`;
            setLoading(null);
        } catch (err) {
            console.error(err);
            showError(`Cannot load ${info.name}${model === 'lvar0' ? ' (\u03bbvar\u00a0=\u00a00)' : ''}: ${err.message}`);
        }
    }

    function setView(i) {
        const cams = renderer.scene.header.cameras;
        viewIndex = (i + cams.length) % cams.length;
        camera.setFromCamera(cams[viewIndex], renderer.scene.header.center);
        renderer.resetPending = true;
    }

    // Render resolution: canvas size in device pixels, scaled and capped
    function updateSize() {
        const dpr = window.devicePixelRatio || 1;
        let w = canvas.clientWidth * dpr * renderScale;
        let h = canvas.clientHeight * dpr * renderScale;
        const cap = Math.sqrt(renderer.maxPixels / Math.max(1, w * h));
        if (cap < 1) { w *= cap; h *= cap; }
        w = Math.max(16, Math.floor(w));
        h = Math.max(16, Math.floor(h));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        renderer.resize(w, h);
        $('stat-res').textContent = `${w} × ${h}`;
    }

    // Focal length: the dataset camera's vertical field of view, applied to the render height
    function focal() {
        const fy = (camera.fy / camera.refHeight) * renderer.size[1];
        return [fy, fy];
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------------------------------------------------
    const pointers = new Map();
    let pinchDistance = 0;
    let pinchCenter = null;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
        canvas.setPointerCapture(e.pointerId);
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
        if (pointers.size === 2) {
            const [a, b] = [...pointers.values()];
            pinchDistance = Math.hypot(a.x - b.x, a.y - b.y);
            pinchCenter = [(a.x + b.x) / 2, (a.y + b.y) / 2];
        }
    });
    canvas.addEventListener('pointermove', (e) => {
        const p = pointers.get(e.pointerId);
        if (!p || !renderer.scene) return;
        const dpr = (window.devicePixelRatio || 1) * renderScale;
        const dx = e.clientX - p.x, dy = e.clientY - p.y;
        p.x = e.clientX; p.y = e.clientY;
        if (pointers.size === 2) {
            const [a, b] = [...pointers.values()];
            const dist = Math.hypot(a.x - b.x, a.y - b.y);
            const c = [(a.x + b.x) / 2, (a.y + b.y) / 2];
            if (pinchDistance > 0) camera.dolly(pinchDistance / Math.max(dist, 1));
            if (pinchCenter) camera.pan((c[0] - pinchCenter[0]) * dpr, (c[1] - pinchCenter[1]) * dpr, renderer.size[1]);
            pinchDistance = dist;
            pinchCenter = c;
        } else if (p.button === 2 || p.button === 1 || p.shift) {
            camera.pan(dx * dpr, dy * dpr, renderer.size[1]);
        } else {
            camera.orbit(dx, dy);
        }
    });
    const endPointer = (e) => {
        pointers.delete(e.pointerId);
        if (pointers.size < 2) { pinchDistance = 0; pinchCenter = null; }
    };
    canvas.addEventListener('pointerup', endPointer);
    canvas.addEventListener('pointercancel', endPointer);
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera.dolly(Math.exp(e.deltaY * 0.001));
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { if (renderer.scene) setView(viewIndex); });
    const keys = new Set();
    window.addEventListener('keydown', (e) => { if (!e.target.closest('input, select, button')) keys.add(e.key.toLowerCase()); });
    window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => keys.clear());

    // ---------------------------------------------------------------------------------------------------------------
    // Controls
    // ---------------------------------------------------------------------------------------------------------------
    const scenesBox = $('scene-buttons');
    SCENES.forEach((s, i) => {
        const b = document.createElement('button');
        b.textContent = s.name;
        b.dataset.scene = s.id;
        b.addEventListener('click', () => selectScene(i));
        scenesBox.appendChild(b);
    });
    document.querySelectorAll('[data-model]').forEach((b) => b.addEventListener('click', () => {
        if (b.dataset.model === model) return;
        model = b.dataset.model;
        selectScene(sceneIndex, true);
    }));
    // Spatial reuse and temporal accumulation, each on or off
    const METHOD_HINTS = [
        'Plain stochastic rendering: one random sample per pixel and frame.',
        'Spatial reuse only: every frame on its own.',
        'Temporal accumulation of the raw samples, without spatial reuse.',
        'SteadySplats: spatial and temporal reuse, as in the paper.',
    ];
    function updateMethod() {
        const spatial = (renderer.flags & FLAG_SPATIAL) !== 0, temporal = (renderer.flags & FLAG_TEMPORAL) !== 0;
        document.querySelectorAll('[data-toggle]').forEach((b) => {
            const on = b.dataset.toggle === 'spatial' ? spatial : temporal;
            b.classList.toggle('active', on);
            b.setAttribute('aria-pressed', String(on));
        });
        $('method-hint').textContent = METHOD_HINTS[(spatial ? 1 : 0) + (temporal ? 2 : 0)];
        $('converge').disabled = !temporal;  // the rest average belongs to the temporal accumulation
    }
    document.querySelectorAll('[data-toggle]').forEach((b) => b.addEventListener('click', () => {
        renderer.flags ^= b.dataset.toggle === 'spatial' ? FLAG_SPATIAL : FLAG_TEMPORAL;
        renderer.resetPending = true;
        updateMethod();
    }));
    updateMethod();
    document.querySelectorAll('[data-noise]').forEach((b) => b.addEventListener('click', () => {
        renderer.flags = (renderer.flags & ~FLAG_WHITE_NOISE) | (b.dataset.noise === 'white' ? FLAG_WHITE_NOISE : 0);
        renderer.resetPending = true;
        document.querySelectorAll('[data-noise]').forEach((x) => x.classList.toggle('active', x === b));
    }));
    $('render-scale').addEventListener('change', (e) => { renderScale = Number(e.target.value); });

    // Touch screens: the gestures instead of mouse and keys
    if (matchMedia("(pointer: coarse)").matches && $("help")) {
        $("help").textContent = "Drag to orbit, pinch to zoom, drag with two fingers to pan, double-tap to reset.";
    }

    // Controls panel: collapses to a menu button (remembered); closed at first on small screens
    const panel = document.querySelector('.panel');
    function setPanel(open, remember = true) {
        panel.hidden = !open;
        $('panel-open').hidden = open;
        if (remember) { try { localStorage.setItem(PANEL_KEY, open ? 'open' : 'closed'); } catch { /* not remembered */ } }
    }
    let panelState = null;
    try { panelState = localStorage.getItem(PANEL_KEY); } catch { /* storage unavailable */ }
    setPanel(panelState ? panelState === 'open' : !matchMedia('(max-width: 700px), (max-height: 500px)').matches, false);
    $('panel-open').addEventListener('click', () => setPanel(true));
    $('panel-close').addEventListener('click', () => setPanel(false));

    // Fullscreen (hides the browser's bars on phones), where the browser allows it for the page
    const root = document.documentElement;
    const fullscreenElement = () => document.fullscreenElement || document.webkitFullscreenElement;
    if (!(document.fullscreenEnabled || document.webkitFullscreenEnabled)) $('fullscreen').hidden = true;
    $('fullscreen').addEventListener('click', () => {
        if (fullscreenElement()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        else (root.requestFullscreen || root.webkitRequestFullscreen).call(root);
    });
    const updateFullscreen = () => {
        const on = !!fullscreenElement();
        $('fullscreen').classList.toggle('active', on);
        $('fullscreen').setAttribute('aria-label', on ? 'Leave fullscreen' : 'Fullscreen');
        $('fullscreen').title = on ? 'Leave fullscreen' : 'Fullscreen';
    };
    document.addEventListener('fullscreenchange', updateFullscreen);
    document.addEventListener('webkitfullscreenchange', updateFullscreen);
    $('converge').addEventListener('change', (e) => { renderer.converge = e.target.checked; renderer.restFrames = 0; });
    $('view-prev').addEventListener('click', () => renderer.scene && setView(viewIndex - 1));
    $('view-next').addEventListener('click', () => renderer.scene && setView(viewIndex + 1));

    // ---------------------------------------------------------------------------------------------------------------
    // Loop
    // ---------------------------------------------------------------------------------------------------------------
    let last = performance.now();
    let fpsFrames = 0, fpsTime = 0;
    function loop(now) {
        const dt = Math.min(0.1, (now - last) / 1000);
        last = now;
        if (renderer.scene && !document.hidden) {
            const speed = (keys.has('shift') ? 3 : 1) * dt;
            const f = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
            const r = (keys.has('d') ? 1 : 0) - (keys.has('a') ? 1 : 0);
            const u = (keys.has('e') ? 1 : 0) - (keys.has('q') ? 1 : 0);
            if (f || r || u) camera.move(f * speed, r * speed, u * speed);
            updateSize();
            const [fx, fy] = focal();
            if (renderer.render(camera.axes(), fx, fy)) fpsFrames++;
            if (renderer.stalled?.(now)) { switchToWebGL('stall'); return; }
            fpsTime += dt;
            if (fpsTime > 0.5) {
                const idle = renderer.converge && (renderer.flags & FLAG_TEMPORAL) && renderer.restFrames > renderer.restLimit;
                $('stat-fps').textContent = idle ? 'idle' : `${Math.round(fpsFrames / fpsTime)}`;
                const n = renderer.restFrames;
                $('stat-rest').textContent = !renderer.converge || !(renderer.flags & FLAG_TEMPORAL) ? 'off'
                    : n === 0 ? 'moving' : n > renderer.restLimit ? `converged (${renderer.restLimit} frames)` : `${n} frames`;
                const g = renderer.gpuTimes;
                $('stat-gpu').textContent = g ? `${g.total.toFixed(2)} ms (raster ${g.raster.toFixed(2)}, reuse ${g.reuse.toFixed(2)})` : 'n/a';
                fpsFrames = 0;
                fpsTime = 0;
            }
        }
        if (!TEST) requestAnimationFrame(loop);
    }

    if (TEST) {
        // Manual stepping for automated comparisons with the reference renderer's frames
        renderer.restLimit = Infinity;
        renderer.maxInFlight = Infinity;
        window.viewerTest = {
            renderer, camera,
            loadScene: async (id, m) => { if (m) model = m; return selectScene(SCENES.findIndex((s) => s.id === id)); },
            // pose: { position, rotation (camera-to-world rows), fx, fy, width, height }
            renderPose(pose, frameIdx, flags) {
                camera.pose = pose;
                if (canvas.width !== pose.width || canvas.height !== pose.height) { canvas.width = pose.width; canvas.height = pose.height; }
                renderer.resize(pose.width, pose.height);
                if (frameIdx !== undefined) renderer.frameIdx = frameIdx;
                if (flags !== undefined) renderer.flags = flags;
                renderer.render(camera.axes(), pose.fx, pose.fy);
            },
            reset() { renderer.resetPending = true; renderer.prevViewProj = null; },
            readOutput: () => renderer.readOutput(),
            visibleCount: () => renderer.visibleCount(),
            done: () => renderer.done(),
        };
        await selectScene(sceneIndex);
        window.viewerReady = true;
    } else {
        selectScene(sceneIndex);
        requestAnimationFrame(loop);
    }
}

function setLoading(fraction, label) {
    const box = $('loading');
    if (fraction === null) { box.hidden = true; return; }
    box.hidden = false;
    $('loading-label').textContent = `${label}...`;
    $('loading-fill').style.width = `${Math.round(fraction * 100)}%`;
}

// Neither WebGPU nor WebGL 2 works: show the project page's teaser video instead, with the reasons
function showFallback(issues) {
    $('loading').hidden = true;
    document.querySelector('.panel').hidden = true;
    const box = $('fallback');
    box.querySelector('.reasons').textContent = issues.join(' ') + hints(issues);
    box.hidden = false;
    box.querySelector('video').play().catch(() => {});
}

function showError(message) {
    if ($('loading')) $('loading').hidden = true;
    const box = $('error');
    box.hidden = false;
    box.querySelector('p').textContent = message;
}

main().catch((err) => { console.error(err); showError(err.message); });
