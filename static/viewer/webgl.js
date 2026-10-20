// WebGL 2 fallback of the viewer for browsers without WebGPU (e.g. Chrome and Firefox on most Linux systems, Safari
// before 26): the same method and frame state as the WebGPU renderer in viewer.js, with the passes of shaders_gl.js.
import { K_EMPTY, RendererBase } from './common.js';
import { DISPLAY_FS, FULLSCREEN_VS, RASTER_FS, RASTER_VS, REUSE_FS } from './shaders_gl.js';

const SPLAT_TEXTURE_WIDTH = 4096;   // Gaussians per row of the data textures
const CHUNK_TEXTURE_WIDTH = 1536;   // 512 chunks of 3 texels per row

// Texture units: the rasterization samples 0-3, the reuse pass 4-11, the display 12, so that no pass samples a texture
// it renders to
const UNIT = { splatsA: 0, splatsB: 1, chunks: 2, stbn: 3, colorTex: 4, idTex: 5, depthTex: 6, gaussTex: 7, kInA: 8,
    kInB: 9, histTex: 10, restIn: 11, image: 12 };

function compile(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
}

function link(gl, vs, fs) {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    // Uniform block at binding 0, samplers at their fixed units
    const block = gl.getUniformBlockIndex(program, 'FrameBlock');
    if (block !== gl.INVALID_INDEX) {
        if (gl.getActiveUniformBlockParameter(program, block, gl.UNIFORM_BLOCK_DATA_SIZE) !== 384) throw new Error('unexpected uniform block layout');
        gl.uniformBlockBinding(program, block, 0);
    }
    gl.useProgram(program);
    for (const [name, unit] of Object.entries(UNIT)) {
        const loc = gl.getUniformLocation(program, name);
        if (loc) gl.uniform1i(loc, unit);
    }
    return program;
}

function texture(gl, target, internalFormat, width, height, filter, depth = 0) {
    const t = gl.createTexture();
    gl.bindTexture(target, t);
    if (target === gl.TEXTURE_3D) gl.texStorage3D(target, 1, internalFormat, width, height, depth);
    else gl.texStorage2D(target, 1, internalFormat, width, height);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
}

function framebuffer(gl, colors, depth) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    colors.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    if (depth) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
    gl.drawBuffers(colors.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete (0x${status.toString(16)})`);
    return fb;
}

export class WebGLRenderer extends RendererBase {
    // The renderer, or why WebGL 2 cannot run it here. A software renderer (SwiftShader, llvmpipe) is refused unless
    // allowSoftware: it needs seconds per frame and blocks the page while it renders.
    static create(canvas, stbnData, allowSoftware = false) {
        const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false,
            premultipliedAlpha: false, powerPreference: 'high-performance' });
        if (!gl) return 'WebGL 2 is not available either.';
        const debug = gl.getExtension('WEBGL_debug_renderer_info');
        const rendererName = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
        console.info('WebGL renderer:', rendererName);
        const software = /swiftshader|llvmpipe|lavapipe|software/i.test(rendererName);
        if (software && !allowSoftware) return 'WebGL 2 also runs without the GPU here (software rendering, far too slow for the viewer).';
        if (!gl.getExtension('EXT_color_buffer_float')) return 'WebGL 2 cannot render to float textures here (EXT_color_buffer_float).';
        if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 4 || gl.getParameter(gl.MAX_TEXTURE_SIZE) < 4096) return 'The WebGL 2 limits of this GPU are too small.';
        try {
            const renderer = new WebGLRenderer(gl, stbnData);
            renderer.adapterInfo = rendererName;
            return renderer;
        } catch (err) {
            return `The WebGL 2 fallback failed to start (${err.message}).`;
        }
    }

    constructor(gl, stbnData) {
        super();
        this.backend = 'WebGL 2 (fallback)';
        this.maxPixels = 1920 * 1200;  // render resolution cap (140 bytes per pixel)
        this.gl = gl;
        this.rasterProgram = link(gl, RASTER_VS, RASTER_FS);
        this.reuseProgram = link(gl, FULLSCREEN_VS, REUSE_FS);
        this.displayProgram = link(gl, FULLSCREEN_VS, DISPLAY_FS);
        this.vao = gl.createVertexArray();  // no attributes: everything comes from textures
        this.frameUbo = gl.createBuffer();
        gl.bindBuffer(gl.UNIFORM_BUFFER, this.frameUbo);
        gl.bufferData(gl.UNIFORM_BUFFER, 384, gl.DYNAMIC_DRAW);
        gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.frameUbo);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        this.stbn = texture(gl, gl.TEXTURE_3D, gl.R8, 128, 128, gl.NEAREST, 64);
        gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, 128, 128, 64, gl.RED, gl.UNSIGNED_BYTE, stbnData);
        // GPU timing, if the browser offers it
        this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
        this.queries = null;
    }

    setScene(scene) {
        const gl = this.gl;
        if (scene.header.precision === 'float') throw new Error('the WebGL 2 fallback renders packed scenes only');
        for (const t of [this.splatsA, this.splatsB, this.chunks]) if (t) gl.deleteTexture(t);
        const n = scene.header.count, words = scene.words;
        // 7 words per Gaussian into two RGBA32UI textures (words 0-3 and 4-6)
        const tw = SPLAT_TEXTURE_WIDTH, th = Math.ceil(n / tw);
        const a = new Uint32Array(tw * th * 4), b = new Uint32Array(tw * th * 4);
        for (let k = 0, w = 0, o = 0; k < n; k++, w += 7, o += 4) {
            a[o] = words[w]; a[o + 1] = words[w + 1]; a[o + 2] = words[w + 2]; a[o + 3] = words[w + 3];
            b[o] = words[w + 4]; b[o + 1] = words[w + 5]; b[o + 2] = words[w + 6];
        }
        this.splatsA = texture(gl, gl.TEXTURE_2D, gl.RGBA32UI, tw, th, gl.NEAREST);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tw, th, gl.RGBA_INTEGER, gl.UNSIGNED_INT, a);
        this.splatsB = texture(gl, gl.TEXTURE_2D, gl.RGBA32UI, tw, th, gl.NEAREST);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tw, th, gl.RGBA_INTEGER, gl.UNSIGNED_INT, b);
        // 12 floats per chunk as 3 RGBA32F texels
        const cw = CHUNK_TEXTURE_WIDTH, ch = Math.max(1, Math.ceil(scene.chunks.length / 4 / cw));
        const c = new Float32Array(cw * ch * 4);
        c.set(scene.chunks);
        this.chunks = texture(gl, gl.TEXTURE_2D, gl.RGBA32F, cw, ch, gl.NEAREST);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, cw, ch, gl.RGBA, gl.FLOAT, c);
        this.scene = scene;
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
        const gl = this.gl;
        for (const t of this.targets || []) gl.deleteTexture(t);
        for (const f of this.framebuffers || []) gl.deleteFramebuffer(f);
        const tex = (format, filter = gl.NEAREST) => texture(gl, gl.TEXTURE_2D, format, width, height, filter);
        this.colorTex = tex(gl.RGBA8);
        this.idTex = tex(gl.R32UI);
        this.gaussTex = tex(gl.RGBA32UI);
        this.depthTex = tex(gl.DEPTH_COMPONENT32F);
        this.hist = [tex(gl.RGBA16F, gl.LINEAR), tex(gl.RGBA16F, gl.LINEAR)];
        this.kA = [tex(gl.RGBA32UI), tex(gl.RGBA32UI)];
        this.kB = [tex(gl.RGBA32UI), tex(gl.RGBA32UI)];
        this.rest = [tex(gl.RGBA32F), tex(gl.RGBA32F)];
        this.targets = [this.colorTex, this.idTex, this.gaussTex, this.depthTex, ...this.hist, ...this.kA, ...this.kB, ...this.rest];
        this.rasterFbo = framebuffer(gl, [this.colorTex, this.idTex, this.gaussTex], this.depthTex);
        this.reuseFbo = [0, 1].map((p) => framebuffer(gl, [this.hist[p], this.kA[p], this.kB[p], this.rest[p]]));
        this.readFbo = [0, 1].map((p) => framebuffer(gl, [this.hist[p]]));
        this.framebuffers = [this.rasterFbo, ...this.reuseFbo, ...this.readFbo];
        // Empty K-buffers
        const empty = new Uint32Array([K_EMPTY[0], K_EMPTY[1], K_EMPTY[0], K_EMPTY[1]]);
        for (const fb of this.reuseFbo) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
            gl.clearBufferuiv(gl.COLOR, 1, empty);
            gl.clearBufferuiv(gl.COLOR, 2, empty);
            gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
            gl.clearBufferfv(gl.COLOR, 3, [0, 0, 0, 0]);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    bind(target, unit, t) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(target, t);
    }

    // Renders a frame; returns false if there is nothing to render (converged at rest, see RendererBase.frameData)
    render(cam, fx, fy, toCanvas = true) {
        const gl = this.gl;
        this.collectTiming();
        const frame = this.frameData(cam, fx, fy);
        if (!frame) return false;
        const [w, h] = this.size;
        const p = this.parity, q = 1 - p;
        gl.bindBuffer(gl.UNIFORM_BUFFER, this.frameUbo);
        gl.bufferSubData(gl.UNIFORM_BUFFER, 0, frame);
        gl.bindVertexArray(this.vao);
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        const timing = this.timer && !this.queries ? [gl.createQuery(), gl.createQuery()] : null;

        // 1. Stochastic rasterization: a quad per Gaussian, the depth test keeps the closest surviving fragment
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.rasterFbo);
        gl.viewport(0, 0, w, h);
        gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 1]);
        gl.clearBufferuiv(gl.COLOR, 1, [0xffffffff, 0, 0, 0]);
        gl.clearBufferuiv(gl.COLOR, 2, [0, 0, 0, 0]);
        gl.depthMask(true);
        gl.clearBufferfv(gl.DEPTH, 0, [1]);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.useProgram(this.rasterProgram);
        this.bind(gl.TEXTURE_2D, UNIT.splatsA, this.splatsA);
        this.bind(gl.TEXTURE_2D, UNIT.splatsB, this.splatsB);
        this.bind(gl.TEXTURE_2D, UNIT.chunks, this.chunks);
        this.bind(gl.TEXTURE_3D, UNIT.stbn, this.stbn);
        if (timing) gl.beginQuery(this.timer.TIME_ELAPSED_EXT, timing[0]);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.scene.header.count);
        if (timing) gl.endQuery(this.timer.TIME_ELAPSED_EXT);
        gl.disable(gl.DEPTH_TEST);

        // 2. Spatial reuse, temporal accumulation and the rest average
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.reuseFbo[p]);
        gl.useProgram(this.reuseProgram);
        this.bind(gl.TEXTURE_2D, UNIT.colorTex, this.colorTex);
        this.bind(gl.TEXTURE_2D, UNIT.idTex, this.idTex);
        this.bind(gl.TEXTURE_2D, UNIT.depthTex, this.depthTex);
        this.bind(gl.TEXTURE_2D, UNIT.gaussTex, this.gaussTex);
        this.bind(gl.TEXTURE_2D, UNIT.kInA, this.kA[q]);
        this.bind(gl.TEXTURE_2D, UNIT.kInB, this.kB[q]);
        this.bind(gl.TEXTURE_2D, UNIT.histTex, this.hist[q]);
        this.bind(gl.TEXTURE_2D, UNIT.restIn, this.rest[q]);
        if (timing) gl.beginQuery(this.timer.TIME_ELAPSED_EXT, timing[1]);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        if (timing) gl.endQuery(this.timer.TIME_ELAPSED_EXT);

        // 3. Display
        if (toCanvas) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.useProgram(this.displayProgram);
            this.bind(gl.TEXTURE_2D, UNIT.image, this.hist[p]);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
        }
        if (timing) this.queries = timing;
        this.output = p;
        this.parity = q;
        this.frameIdx++;
        return true;
    }

    // Reads finished timer queries (raster, reuse) from an earlier frame
    collectTiming() {
        const gl = this.gl;
        if (!this.queries) return;
        const [qr, qs] = this.queries;
        if (!gl.getQueryParameter(qs, gl.QUERY_RESULT_AVAILABLE)) return;
        if (!gl.getParameter(this.timer.GPU_DISJOINT_EXT)) {
            const raster = gl.getQueryParameter(qr, gl.QUERY_RESULT) / 1e6, reuse = gl.getQueryParameter(qs, gl.QUERY_RESULT) / 1e6;
            this.gpuTimes = { preprocess: 0, raster, reuse, total: raster + reuse };
        }
        gl.deleteQuery(qr);
        gl.deleteQuery(qs);
        this.queries = null;
    }

    done() {
        this.gl.finish();
        return Promise.resolve();
    }

    // Test helper: the last output as 8-bit RGBA, rows top to bottom (as stored)
    async readOutput() {
        const gl = this.gl;
        const [w, h] = this.size;
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.readFbo[this.output]);
        const px = new Float32Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, px);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        const out = new Uint8Array(w * h * 4);
        for (let i = 0; i < out.length; i++) out[i] = (i & 3) === 3 ? 255 : Math.max(0, Math.min(255, Math.round(px[i] * 255)));
        return out;
    }

    async visibleCount() {
        return null;  // no compaction in WebGL 2: every Gaussian is drawn, culled ones as degenerate quads
    }
}
