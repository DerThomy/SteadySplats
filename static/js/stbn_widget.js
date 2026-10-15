// Spatio-temporal blue noise widget (#temporal section). Two overlapping Gaussians are rendered stochastically with
// one sample per pixel and frame, as in our renderer: a Gaussian is kept at a pixel if its threshold u is below its
// opacity, and the closest kept Gaussian colors the pixel. The thresholds come from white noise, from blue noise
// shared by all Gaussians, or from blue noise offset per primitive (ours). The STBN volume is loaded on demand from
// static/stbn/stbn_scalar.js via a script tag, so the widget also works when the page is opened via file:///.
(function () {
    'use strict';

    const DATA_URL = 'static/stbn/stbn_scalar.js';
    const SIZE = 64; // image size in pixels
    const LOOP_FRAMES = 128; // the frame average restarts after this many frames
    const FRAME_MS = 100;
    const PHI_INV = 0.6180339887498949; // golden-ratio rotation of the thresholds after each pass through the volume
    // Per-primitive texture offsets follow the R2 low-discrepancy sequence
    const R2_G = 1.32471795724474602596;
    const R2_STEPS = [1 / R2_G, 1 / (R2_G * R2_G)];
    const VARIANTS = ['white', 'shared', 'ours'];

    // Front to back: center, standard deviations along the rotated axes, rotation, peak opacity, color
    // (overlapping enough that shared blue noise visibly hides the teal Gaussian behind the orange one)
    const GAUSSIANS = [
        { cx: 27, cy: 30, sx: 12, sy: 9, angle: 0.5, alpha: 0.8, color: [249, 115, 22] },
        { cx: 36, cy: 34, sx: 10, sy: 13, angle: -0.35, alpha: 0.75, color: [20, 184, 166] }
    ];
    const BACKGROUND = [241, 245, 249];

    // Integer hash (lowbias32) giving independent uniform thresholds for white noise
    function whiteNoise(key) {
        let k = key;
        k ^= k >>> 16;
        k = Math.imul(k, 0x7feb352d);
        k ^= k >>> 15;
        k = Math.imul(k, 0x846ca68b);
        k ^= k >>> 16;
        return (k >>> 8) / 16777216;
    }

    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = src;
            s.onload = () => { s.remove(); resolve(); };
            s.onerror = () => { s.remove(); reject(new Error(`Failed to load ${src}`)); };
            document.head.appendChild(s);
        });
    }

    function initStbnWidget() {
        const root = document.getElementById('stbn-widget');
        if (!root) return;
        const N = SIZE * SIZE;

        // Opacity of each Gaussian at each pixel center
        const alpha = GAUSSIANS.map((g) => {
            const a = new Float32Array(N);
            const cos = Math.cos(g.angle), sin = Math.sin(g.angle);
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    const dx = x + 0.5 - g.cx, dy = y + 0.5 - g.cy;
                    const u = cos * dx + sin * dy, v = -sin * dx + cos * dy;
                    a[y * SIZE + x] = g.alpha * Math.exp(-0.5 * (u * u / (g.sx * g.sx) + v * v / (g.sy * g.sy)));
                }
            }
            return a;
        });

        // Exact alpha blending, front to back, as the reference
        const reference = new Float32Array(3 * N);
        for (let i = 0; i < N; i++) {
            const a0 = alpha[0][i], a1 = alpha[1][i];
            for (let c = 0; c < 3; c++) {
                reference[3 * i + c] = (a0 * GAUSSIANS[0].color[c] + (1 - a0) * a1 * GAUSSIANS[1].color[c] +
                    (1 - a0) * (1 - a1) * BACKGROUND[c]) / 255;
            }
        }

        const canvases = {};
        ['white', 'shared', 'ours', 'reference'].forEach((name) => {
            const canvas = document.getElementById(`stbn-canvas-${name}`);
            canvas.width = canvas.height = SIZE;
            canvases[name] = canvas.getContext('2d');
        });
        const psnrLabels = Object.fromEntries(VARIANTS.map((v) => [v, document.getElementById(`stbn-psnr-${v}`)]));
        const frameBadge = document.getElementById('stbn-frameBadge');
        const playBtn = document.getElementById('stbn-playBtn');
        const viewButtons = root.querySelectorAll('[data-stbn-view]');

        const frameImage = Object.fromEntries(VARIANTS.map((v) => [v, new Float32Array(3 * N)]));
        const sum = Object.fromEntries(VARIANTS.map((v) => [v, new Float32Array(3 * N)]));
        const shown = new Float32Array(3 * N);
        const pixels = new ImageData(SIZE, SIZE);

        let stbn = null;
        let frame = 0; // frame index since the last restart
        let view = 'frames';
        let playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let visible = false;

        // Blue noise threshold at texel (x, y) of slice t
        function blueNoise(x, y, t) {
            const v = stbn[((t % 64) * 128 + (y & 127)) * 128 + (x & 127)];
            const u = (v + 0.5) / 256 + PHI_INV * Math.floor(t / 64);
            return u - Math.floor(u);
        }

        const offsets = GAUSSIANS.map((g, p) => R2_STEPS.map((step) => Math.floor(((0.5 + p * step) % 1) * 128)));

        function threshold(variant, x, y, p, t) {
            if (variant === 'white') return whiteNoise(((t * GAUSSIANS.length + p) * SIZE + y) * SIZE + x);
            if (variant === 'shared') return blueNoise(x, y, t);
            return blueNoise(x + offsets[p][0], y + offsets[p][1], t);
        }

        // One sample per pixel: the closest Gaussian whose threshold lies below its opacity, else the background
        function renderFrame(variant, t, out) {
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    const i = y * SIZE + x;
                    let color = BACKGROUND;
                    for (let p = 0; p < GAUSSIANS.length; p++) {
                        if (threshold(variant, x, y, p, t) < alpha[p][i]) {
                            color = GAUSSIANS[p].color;
                            break;
                        }
                    }
                    for (let c = 0; c < 3; c++) out[3 * i + c] = color[c] / 255;
                }
            }
        }

        function draw(ctx, img) {
            const d = pixels.data;
            for (let i = 0; i < N; i++) {
                d[4 * i] = Math.round(img[3 * i] * 255);
                d[4 * i + 1] = Math.round(img[3 * i + 1] * 255);
                d[4 * i + 2] = Math.round(img[3 * i + 2] * 255);
                d[4 * i + 3] = 255;
            }
            ctx.putImageData(pixels, 0, 0);
        }

        function psnr(img) {
            let mse = 0;
            for (let i = 0; i < img.length; i++) mse += (img[i] - reference[i]) * (img[i] - reference[i]);
            return 10 * Math.log10(img.length / mse);
        }

        // Shows each variant in the current view, with its PSNR to the reference
        function show() {
            VARIANTS.forEach((variant) => {
                if (view === 'frame') {
                    shown.set(frameImage[variant]);
                } else {
                    const s = sum[variant];
                    for (let i = 0; i < shown.length; i++) shown[i] = s[i] / (frame + 1);
                }
                draw(canvases[variant], shown);
                psnrLabels[variant].textContent = `PSNR ${psnr(shown).toFixed(1)} dB`;
            });
            frameBadge.textContent = view !== 'frames' ? `Frame ${frame + 1}`
                : frame === 0 ? '1 frame averaged' : `${frame + 1} frames averaged`;
        }

        // Renders frame t of every variant and adds it to the running sums
        function renderAll(t) {
            VARIANTS.forEach((variant) => {
                renderFrame(variant, t, frameImage[variant]);
                const s = sum[variant], img = frameImage[variant];
                for (let i = 0; i < s.length; i++) s[i] += img[i];
            });
        }

        function restart() {
            frame = 0;
            VARIANTS.forEach((variant) => sum[variant].fill(0));
            renderAll(0);
            show();
        }

        function tick() {
            if (!stbn || !playing || !visible || document.hidden) return;
            if (frame + 1 >= LOOP_FRAMES) {
                restart();
                return;
            }
            frame++;
            renderAll(frame);
            show();
        }

        function setPlaying(value) {
            playing = value;
            playBtn.querySelector('use').setAttribute('href', playing ? '#icon-pause' : '#icon-play');
            playBtn.title = playing ? 'Pause' : 'Play';
        }

        playBtn.addEventListener('click', () => setPlaying(!playing));
        viewButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                view = btn.dataset.stbnView;
                viewButtons.forEach((b) => {
                    b.classList.toggle('button-17-selected', b === btn);
                    b.classList.toggle('button-17', b !== btn);
                });
                if (!stbn) return;
                // The frame average starts over, so its convergence can be watched from the first frame
                if (view === 'frames') {
                    restart();
                    setPlaying(true);
                } else {
                    show();
                }
            });
        });
        setPlaying(playing);

        draw(canvases.reference, reference);
        new IntersectionObserver((entries) => {
            visible = entries[entries.length - 1].isIntersecting;
        }).observe(root);

        // Load the blue noise volume once the widget approaches the viewport
        new IntersectionObserver((entries, observer) => {
            if (!entries.some((entry) => entry.isIntersecting)) return;
            observer.disconnect();
            frameBadge.textContent = 'Loading...';
            loadScript(DATA_URL).then(() => {
                const data = window.STBN_DATA;
                const binary = atob(data.base64);
                stbn = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) stbn[i] = binary.charCodeAt(i);
                data.base64 = null;
                restart();
                setInterval(tick, FRAME_MS);
            }).catch((err) => {
                console.warn(err);
                frameBadge.textContent = 'Blue noise unavailable';
            });
        }, { rootMargin: '100% 0px' }).observe(root);
    }

    document.addEventListener('DOMContentLoaded', initStbnWidget);
})();
