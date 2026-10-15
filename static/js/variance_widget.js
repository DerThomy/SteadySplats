// Stochastic Splats Explorer (variance widget). Scene data is loaded on demand from
// static/variance/data_<scene>.js via script tags, so the widget also works when opened via file:///.
// The data is packed by pack_variance_data.js (notes folder) and rebuilt here into the original binary layout.
(function () {
    const DATA_DIR = 'static/variance/';

    function base64ToBytes(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    async function inflateRaw(bytes) {
        const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
        return new Response(stream).arrayBuffer();
    }

    // Rebuilds the original scene binary (16-byte header, offset table, then per pixel and model: variance,
    // mean color, entry count and the entries' Gaussian id, probability and color) from the packed format,
    // which pack_variance_data.js documents. Entries refer to a table of Gaussians or to an entry of the left
    // or top neighbor.
    function unpackScene(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const width = view.getUint16(6, true);
        const N = view.getUint32(16, true);
        let o = 16 + 28;
        const models = [0, 1].map((m) => {
            const s = {
                G: view.getUint32(20 + 12 * m, true),
                E: view.getUint32(24 + 12 * m, true),
                X: view.getUint32(28 + 12 * m, true)
            };
            s.count = o; o += N;
            s.variance = o; o += 4 * N;
            s.mean = o; o += 3 * N;
            s.refs = o; o += s.E;
            s.tableIndex = o; o += 3 * s.X;
            s.prob = o; o += s.E;
            s.ids = o; o += 4 * s.G;
            s.colors = o; o += 3 * s.G;
            // First entry of each pixel, and the Gaussian (table index) of each entry once resolved
            s.firstEntry = new Uint32Array(N + 1);
            for (let p = 0; p < N; p++) s.firstEntry[p + 1] = s.firstEntry[p] + bytes[s.count + p];
            s.entryGaussian = new Uint32Array(s.E);
            s.nextTableIndex = 0;
            return s;
        });

        const out = new Uint8Array(16 + 4 * N + 16 * N + 8 * (models[0].E + models[1].E));
        const outView = new DataView(out.buffer);
        out.set(bytes.subarray(0, 16), 0);
        let pos = 16 + 4 * N;
        for (let p = 0; p < N; p++) {
            outView.setUint32(16 + 4 * p, pos, true);
            for (const s of models) {
                // Variance (float32, little endian; the planes start with the most significant byte), mean, count
                for (let b = 0; b < 4; b++) out[pos + b] = bytes[s.variance + (3 - b) * N + p];
                for (let c = 0; c < 3; c++) out[pos + 4 + c] = bytes[s.mean + c * N + p];
                out[pos + 7] = bytes[s.count + p];
                pos += 8;
                for (let e = s.firstEntry[p]; e < s.firstEntry[p + 1]; e++) {
                    const ref = bytes[s.refs + e];
                    let g;
                    if (ref < 120) {
                        g = s.entryGaussian[s.firstEntry[p - 1] + ref];
                    } else if (ref < 240) {
                        g = s.entryGaussian[s.firstEntry[p - width] + ref - 120];
                    } else {
                        const t = s.nextTableIndex++;
                        g = (bytes[s.tableIndex + t] << 16) | (bytes[s.tableIndex + s.X + t] << 8) | bytes[s.tableIndex + 2 * s.X + t];
                    }
                    s.entryGaussian[e] = g;
                    // Gaussian id (int32, little endian), probability, Gaussian color
                    for (let b = 0; b < 4; b++) out[pos + b] = bytes[s.ids + (3 - b) * s.G + g];
                    out[pos + 4] = bytes[s.prob + e];
                    for (let c = 0; c < 3; c++) out[pos + 5 + c] = bytes[s.colors + c * s.G + g];
                    pos += 8;
                }
            }
        }
        return out.buffer;
    }

    // Packed scenes are rebuilt; scenes in the original format (plain base64) are still accepted
    async function decodeScene(sceneObj) {
        if (sceneObj.packed) {
            const packed = base64ToBytes(sceneObj.packed);
            sceneObj.packed = null; // release the large string
            return unpackScene(new Uint8Array(await inflateRaw(packed)));
        }
        const bytes = base64ToBytes(sceneObj.base64);
        sceneObj.base64 = null;
        return bytes.buffer;
    }

    window.VarianceData = { unpackScene }; // also used by pack_variance_data.js to check its output

    function initVarianceWidget() {
        const root = document.getElementById('variance-widget');
        if (!root) return;

        let activeScene = null;
        let requestedScene = null;
        let imgWidth = 622;
        let imgHeight = 412;
        let totalPixels = imgWidth * imgHeight;

        let splitX = Math.floor(imgWidth / 2);
        let isDraggingSplit = false;

        let currentX = 182;
        let currentY = 82;
        // Pixel under the pointer, shown in the loupe (null until the pointer is over the image)
        let hoverX = null;
        let hoverY = null;
        let isLocked = false;
        let isPaused = false;
        let isVisible = false;
        let isPointerOver = false;
        let sampleTimer = null;

        let dataView = null;
        let indexTable = null;

        let samplePixels0 = new Uint8ClampedArray(totalPixels * 4);
        let samplePixels1 = new Uint8ClampedArray(totalPixels * 4);
        let compositePixels = new Uint8ClampedArray(totalPixels * 4);

        const canvas = document.getElementById('variance-imageCanvas');
        const ctx = canvas.getContext('2d');
        const overlayCanvas = document.getElementById('variance-overlayCanvas');
        const overlayCtx = overlayCanvas.getContext('2d');
        const loupeCanvas = document.getElementById('variance-loupeCanvas');
        const loupeCtx = loupeCanvas.getContext('2d');
        const floatingLoupe = document.getElementById('variance-floatingLoupe');
        const canvasWrapper = document.getElementById('variance-canvasWrapper');
        const tooltip = document.getElementById('variance-tooltip');
        const pauseBadge = document.getElementById('variance-pauseBadge');
        const lockBadge = document.getElementById('variance-lockBadge');

        const sceneCache = {}; // sceneName -> { meta, dataView, indexTable }
        const sceneScripts = {}; // sceneName -> Promise of the loading data script
        const sceneDecodes = {}; // sceneName -> Promise of the decoded binary, shared by concurrent calls

        function loadScript(src) {
            return new Promise((resolve, reject) => {
                const s = document.createElement('script');
                s.src = src;
                s.onload = () => resolve();
                s.onerror = () => reject(new Error(`Failed to load ${src}`));
                document.head.appendChild(s);
            });
        }

        // Resolves after the browser had a chance to paint, e.g. the loading text before decoding.
        // The timeout keeps it from stalling where no frames are rendered.
        function nextFrame() {
            return new Promise(resolve => {
                requestAnimationFrame(() => setTimeout(resolve, 0));
                setTimeout(resolve, 100);
            });
        }

        async function switchScene(sceneName) {
            requestedScene = sceneName;

            // Update UI buttons
            root.querySelectorAll('[data-scene]').forEach(btn => {
                const isSelected = btn.dataset.scene === sceneName;
                btn.classList.toggle('button-17-selected', isSelected);
                btn.classList.toggle('button-17', !isSelected);
            });

            const overlay = document.getElementById('variance-loadingOverlay');
            const loadingText = document.getElementById('variance-loadingText');

            if (activeScene === sceneName && sceneCache[sceneName]) {
                overlay.style.display = 'none';
                return;
            }

            overlay.style.display = 'flex';
            const sceneTitle = sceneName.charAt(0).toUpperCase() + sceneName.slice(1);
            loadingText.innerText = `Loading ${sceneTitle}...`;

            try {
                if (!sceneCache[sceneName]) {
                    // Check if data is already in window.SCENE_DATA
                    if (!window.SCENE_DATA || !window.SCENE_DATA[sceneName]) {
                        if (!sceneScripts[sceneName]) {
                            sceneScripts[sceneName] = loadScript(`${DATA_DIR}data_${sceneName}.js`);
                        }
                        await sceneScripts[sceneName];
                    }
                    // Another scene was selected while this one was loading
                    if (requestedScene !== sceneName) return;
                }

                if (!sceneCache[sceneName]) {
                    if (!window.SCENE_DATA || !window.SCENE_DATA[sceneName]) {
                        throw new Error(`Data object for ${sceneName} not found.`);
                    }

                    loadingText.innerText = `Decoding ${sceneTitle}...`;
                    await nextFrame();
                    if (requestedScene !== sceneName) return;
                }

                // Checked again: a concurrent call for the same scene may have decoded it during the await above
                if (!sceneCache[sceneName]) {
                    const sceneObj = window.SCENE_DATA[sceneName];
                    if (!sceneDecodes[sceneName]) sceneDecodes[sceneName] = decodeScene(sceneObj);
                    const arrayBuffer = await sceneDecodes[sceneName];
                    if (!sceneCache[sceneName]) {
                        const numPixels = sceneObj.meta.width * sceneObj.meta.height;
                        sceneCache[sceneName] = {
                            meta: sceneObj.meta,
                            dataView: new DataView(arrayBuffer),
                            indexTable: new Uint32Array(arrayBuffer, 16, numPixels)
                        };
                    }
                    if (requestedScene !== sceneName) return;
                }

                if (activeScene === sceneName) {
                    // Already shown by a concurrent call for the same scene
                    overlay.style.display = 'none';
                    return;
                }

                const cached = sceneCache[sceneName];
                activeScene = sceneName;
                dataView = cached.dataView;
                indexTable = cached.indexTable;
                const meta = cached.meta;

                imgWidth = meta.width;
                imgHeight = meta.height;
                totalPixels = imgWidth * imgHeight;
                splitX = Math.floor(imgWidth / 2);

                canvas.width = imgWidth;
                canvas.height = imgHeight;
                overlayCanvas.width = imgWidth;
                overlayCanvas.height = imgHeight;

                samplePixels0 = new Uint8ClampedArray(totalPixels * 4);
                samplePixels1 = new Uint8ClampedArray(totalPixels * 4);
                compositePixels = new Uint8ClampedArray(totalPixels * 4);

                currentX = meta.max_var_pixel ? meta.max_var_pixel.x : Math.floor(imgWidth / 2);
                currentY = meta.max_var_pixel ? meta.max_var_pixel.y : Math.floor(imgHeight / 2);
                isLocked = false;

                overlay.style.display = 'none';

                generateNewSamples();
                renderCompositeImage();
                inspectPixel(currentX, currentY);

            } catch (err) {
                delete sceneScripts[sceneName]; // allow a retry on the next click
                delete sceneDecodes[sceneName];
                if (requestedScene !== sceneName) return;
                console.error(`Failed to switch to scene ${sceneName}:`, err);
                loadingText.innerText = `Error loading ${sceneName}: ` + err.message;
            }
        }

        // Draw fresh stochastic samples from the distributions for all pixels in current scene
        function generateNewSamples() {
            if (!dataView || !indexTable) return;

            for (let idx = 0; idx < totalPixels; idx++) {
                const offset = indexTable[idx];
                const pIdx = idx * 4;

                // --- Sample Model 0 ---
                const meanR0 = dataView.getUint8(offset + 4);
                const meanG0 = dataView.getUint8(offset + 5);
                const meanB0 = dataView.getUint8(offset + 6);
                const count0 = dataView.getUint8(offset + 7);
                const rand0 = Math.random() * 100;

                let pos = offset + 8;
                let r0 = meanR0, g0 = meanG0, b0 = meanB0;
                let cum0 = 0;
                for (let i = 0; i < count0; i++) {
                    const prob = dataView.getUint8(pos + 4);
                    cum0 += prob;
                    if (rand0 <= cum0) {
                        r0 = dataView.getUint8(pos + 5);
                        g0 = dataView.getUint8(pos + 6);
                        b0 = dataView.getUint8(pos + 7);
                        pos += (count0 - i) * 8; // skip remaining
                        break;
                    }
                    pos += 8;
                }
                samplePixels0[pIdx] = r0;
                samplePixels0[pIdx + 1] = g0;
                samplePixels0[pIdx + 2] = b0;
                samplePixels0[pIdx + 3] = 255;

                // --- Sample Model 1 ---
                const count1Pos = pos;
                const meanR1 = dataView.getUint8(count1Pos + 4);
                const meanG1 = dataView.getUint8(count1Pos + 5);
                const meanB1 = dataView.getUint8(count1Pos + 6);
                const count1 = dataView.getUint8(count1Pos + 7);
                const rand1 = Math.random() * 100;

                pos = count1Pos + 8;
                let r1 = meanR1, g1 = meanG1, b1 = meanB1;
                let cum1 = 0;
                for (let i = 0; i < count1; i++) {
                    const prob = dataView.getUint8(pos + 4);
                    cum1 += prob;
                    if (rand1 <= cum1) {
                        r1 = dataView.getUint8(pos + 5);
                        g1 = dataView.getUint8(pos + 6);
                        b1 = dataView.getUint8(pos + 7);
                        break;
                    }
                    pos += 8;
                }
                samplePixels1[pIdx] = r1;
                samplePixels1[pIdx + 1] = g1;
                samplePixels1[pIdx + 2] = b1;
                samplePixels1[pIdx + 3] = 255;
            }
        }

        // Composite Model 0 (left of splitX) and Model 1 (right of splitX)
        function renderCompositeImage() {
            for (let y = 0; y < imgHeight; y++) {
                const rowStart = y * imgWidth;
                for (let x = 0; x < imgWidth; x++) {
                    const idx = (rowStart + x) * 4;
                    const src = (x < splitX) ? samplePixels0 : samplePixels1;
                    compositePixels[idx] = src[idx];
                    compositePixels[idx + 1] = src[idx + 1];
                    compositePixels[idx + 2] = src[idx + 2];
                    compositePixels[idx + 3] = 255;
                }
            }

            const imgData = new ImageData(compositePixels, imgWidth, imgHeight);
            ctx.putImageData(imgData, 0, 0);

            drawOverlay();
            drawLoupe();
        }

        function drawOverlay() {
            overlayCtx.clearRect(0, 0, imgWidth, imgHeight);

            // 1. Draw Split Divider Line
            overlayCtx.save();
            overlayCtx.beginPath();
            overlayCtx.moveTo(splitX + 0.5, 0);
            overlayCtx.lineTo(splitX + 0.5, imgHeight);
            overlayCtx.lineWidth = 2;
            overlayCtx.strokeStyle = '#ffffff';
            overlayCtx.shadowColor = 'rgba(0,0,0,0.6)';
            overlayCtx.shadowBlur = 4;
            overlayCtx.stroke();
            overlayCtx.restore();

            // 2. Draw Split Handle Knob
            const knobY = imgHeight / 2;
            overlayCtx.save();
            overlayCtx.beginPath();
            overlayCtx.arc(splitX, knobY, 14, 0, Math.PI * 2);
            overlayCtx.fillStyle = '#ffffff';
            overlayCtx.shadowColor = 'rgba(0,0,0,0.4)';
            overlayCtx.shadowBlur = 6;
            overlayCtx.fill();
            overlayCtx.lineWidth = 2.5;
            overlayCtx.strokeStyle = '#21918c'; // --viridis-teal (canvas colors cannot use CSS variables)
            overlayCtx.stroke();

            // Arrows in Knob
            overlayCtx.fillStyle = '#21918c';
            overlayCtx.font = 'bold 11px sans-serif';
            overlayCtx.textAlign = 'center';
            overlayCtx.textBaseline = 'middle';
            overlayCtx.fillText('◀ ▶', splitX, knobY);
            overlayCtx.restore();

            // 3. Inspected pixel, as in the K-buffer explorer: its row and column in yellow and a box around it
            overlayCtx.save();
            overlayCtx.lineWidth = 1;
            overlayCtx.strokeStyle = 'rgba(253, 231, 37, 0.85)'; // viridis yellow
            overlayCtx.beginPath();
            overlayCtx.moveTo(0, currentY + 0.5);
            overlayCtx.lineTo(imgWidth, currentY + 0.5);
            overlayCtx.moveTo(currentX + 0.5, 0);
            overlayCtx.lineTo(currentX + 0.5, imgHeight);
            overlayCtx.stroke();
            overlayCtx.lineWidth = 2;
            overlayCtx.strokeStyle = '#ffffff';
            overlayCtx.strokeRect(currentX - 2, currentY - 2, 5, 5);
            overlayCtx.restore();

            lockBadge.style.display = isLocked ? 'block' : 'none';
        }

        // Magnifies the pixels around the pointer, also while another pixel is locked. Parts outside the image
        // stay black, so the pixel under the pointer is always the one marked in the middle (.loupe-crosshair).
        function drawLoupe() {
            const cropSize = 21;
            const half = Math.floor(cropSize / 2);
            const cx = hoverX !== null ? hoverX : currentX;
            const cy = hoverY !== null ? hoverY : currentY;
            const sx = cx - half, sy = cy - half;
            const x0 = Math.max(0, sx), y0 = Math.max(0, sy);
            const x1 = Math.min(imgWidth, sx + cropSize), y1 = Math.min(imgHeight, sy + cropSize);

            loupeCtx.fillStyle = '#000000';
            loupeCtx.fillRect(0, 0, cropSize, cropSize);
            if (x1 > x0 && y1 > y0) {
                loupeCtx.drawImage(canvas, x0, y0, x1 - x0, y1 - y0, x0 - sx, y0 - sy, x1 - x0, y1 - y0);
            }
        }

        function getPixelData(x, y) {
            if (!dataView || !indexTable) return null;
            if (x < 0 || x >= imgWidth || y < 0 || y >= imgHeight) return null;

            const offset = indexTable[y * imgWidth + x];
            let pos = offset;

            // Model 0
            const var0 = dataView.getFloat32(pos, true); pos += 4;
            const mean0 = [dataView.getUint8(pos), dataView.getUint8(pos+1), dataView.getUint8(pos+2)]; pos += 3;
            const count0 = dataView.getUint8(pos); pos += 1;
            const dist0 = [];
            for (let i = 0; i < count0; i++) {
                const gid = dataView.getInt32(pos, true); pos += 4;
                const prob = dataView.getUint8(pos); pos += 1;
                const color = [dataView.getUint8(pos), dataView.getUint8(pos+1), dataView.getUint8(pos+2)]; pos += 3;
                dist0.push({ gid, prob, color });
            }

            // Model 1
            const var1 = dataView.getFloat32(pos, true); pos += 4;
            const mean1 = [dataView.getUint8(pos), dataView.getUint8(pos+1), dataView.getUint8(pos+2)]; pos += 3;
            const count1 = dataView.getUint8(pos); pos += 1;
            const dist1 = [];
            for (let i = 0; i < count1; i++) {
                const gid = dataView.getInt32(pos, true); pos += 4;
                const prob = dataView.getUint8(pos); pos += 1;
                const color = [dataView.getUint8(pos), dataView.getUint8(pos+1), dataView.getUint8(pos+2)]; pos += 3;
                dist1.push({ gid, prob, color });
            }

            return {
                m0: { variance: var0, meanColor: mean0, distribution: dist0 },
                m1: { variance: var1, meanColor: mean1, distribution: dist1 }
            };
        }

        function inspectPixel(x, y) {
            currentX = x;
            currentY = y;

            drawOverlay();
            drawLoupe();

            const data = getPixelData(x, y);
            if (!data) return;

            // Populate Model 0 with \mathcal{L}_var and Converged Color Circle
            document.getElementById('variance-m0_var_val').innerHTML = `<span class="mathcal">L</span><sub>var</sub> = ${data.m0.variance.toFixed(4)}`;
            const m0MeanRgb = `rgb(${data.m0.meanColor.join(',')})`;
            const m0Circle = document.getElementById('variance-m0_mean_circle');
            m0Circle.style.backgroundColor = m0MeanRgb;
            m0Circle.onmouseenter = (e) => {
                tooltip.style.display = 'block';
                tooltip.innerHTML = `<strong>Converged Color (&lambda;=0):</strong><br>RGB (${data.m0.meanColor.join(', ')})`;
                positionTooltip(e);
            };
            m0Circle.onmousemove = positionTooltip;
            m0Circle.onmouseleave = () => { tooltip.style.display = 'none'; };

            renderVerticalStackedBar('variance-m0_stacked_bar', data.m0.distribution, '&lambda;=0');

            // Populate Model 1 with \mathcal{L}_var and Converged Color Circle
            document.getElementById('variance-m1_var_val').innerHTML = `<span class="mathcal">L</span><sub>var</sub> = ${data.m1.variance.toFixed(4)}`;
            const m1MeanRgb = `rgb(${data.m1.meanColor.join(',')})`;
            const m1Circle = document.getElementById('variance-m1_mean_circle');
            m1Circle.style.backgroundColor = m1MeanRgb;
            m1Circle.onmouseenter = (e) => {
                tooltip.style.display = 'block';
                tooltip.innerHTML = `<strong>Converged Color (&lambda;=2):</strong><br>RGB (${data.m1.meanColor.join(', ')})`;
                positionTooltip(e);
            };
            m1Circle.onmousemove = positionTooltip;
            m1Circle.onmouseleave = () => { tooltip.style.display = 'none'; };

            renderVerticalStackedBar('variance-m1_stacked_bar', data.m1.distribution, '&lambda;=2');
        }

        function renderVerticalStackedBar(containerId, dist, modelLabel) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';

            let totalProb = dist.reduce((acc, it) => acc + it.prob, 0);
            if (totalProb === 0) totalProb = 100;

            dist.forEach(item => {
                const seg = document.createElement('div');
                seg.className = 'v-bar-segment';
                const heightPct = (item.prob / totalProb * 100).toFixed(2);
                seg.style.height = heightPct + '%';
                seg.style.backgroundColor = `rgb(${item.color.join(',')})`;

                seg.onmouseenter = (e) => {
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = `<strong>[${modelLabel}]</strong><br><strong>ID:</strong> ${item.gid >= 0 ? item.gid : 'Background'}<br><strong>Prob:</strong> ${item.prob}%<br><strong>RGB:</strong> (${item.color.join(', ')})`;
                    positionTooltip(e);
                };
                seg.onmousemove = positionTooltip;
                seg.onmouseleave = () => {
                    tooltip.style.display = 'none';
                };

                container.appendChild(seg);
            });
        }

        function positionTooltip(e) {
            tooltip.style.left = (e.clientX + 14) + 'px';
            tooltip.style.top = (e.clientY + 14) + 'px';
        }

        // Touch screens have no Space key, so there the badge stays visible as a pause button
        const canHover = window.matchMedia('(hover: hover)').matches;

        function updatePauseBadge() {
            if (canHover) {
                pauseBadge.style.display = isPaused ? 'block' : 'none';
            } else {
                pauseBadge.textContent = isPaused ? 'Resume' : 'Pause';
                pauseBadge.style.display = 'block';
            }
        }

        function togglePause() {
            isPaused = !isPaused;
            updatePauseBadge();
        }

        function toggleLock() {
            isLocked = !isLocked;
            drawOverlay();
            drawLoupe();
        }

        function selectPixel(x, y) {
            isLocked = true;
            inspectPixel(x, y);
        }

        // --- Pointer Interaction (mouse, touch, pen): Split Slider, Floating Loupe & Pixel Selection ---
        // Mouse: hovering inspects pixels and a click locks one. Touch: a tap or a sideways swipe inspects
        // pixels and locks the last one; vertical swipes still scroll the page (touch-action: pan-y).
        let lastPointerType = 'mouse';
        let touchGesture = null; // { startX, startY, scrubbing } while a finger is down away from the divider

        function getCanvasCoords(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = imgWidth / rect.width;
            const scaleY = imgHeight / rect.height;
            return {
                x: Math.floor((e.clientX - rect.left) * scaleX),
                y: Math.floor((e.clientY - rect.top) * scaleY),
                cssX: e.clientX - rect.left,
                cssY: e.clientY - rect.top
            };
        }

        function isInsideImage(coords) {
            return coords.x >= 0 && coords.x < imgWidth && coords.y >= 0 && coords.y < imgHeight;
        }

        // Grab radius around the divider in CSS pixels, so it stays usable on small screens and for fingers
        function isNearSplit(coords, pointerType) {
            const splitCssX = splitX * canvas.getBoundingClientRect().width / imgWidth;
            return Math.abs(coords.cssX - splitCssX) <= (pointerType === 'mouse' ? 14 : 24);
        }

        function moveLoupe(coords, pointerType) {
            // Update floating loupe position to top-left of the pointer
            const loupeSize = floatingLoupe.offsetWidth;
            let lx = coords.cssX - loupeSize - 14;
            let ly = coords.cssY - loupeSize - 14;

            // Boundary adjustments so loupe stays fully visible on canvas
            if (lx < 6) lx = coords.cssX + 16; // flip to right of cursor if hitting left edge
            if (ly < 6 && pointerType === 'mouse') ly = coords.cssY + 16; // flip below cursor (a finger would cover it)

            floatingLoupe.style.left = lx + 'px';
            floatingLoupe.style.top = ly + 'px';

            if (isInsideImage(coords)) {
                hoverX = coords.x;
                hoverY = coords.y;
                drawLoupe();
            }
        }

        canvasWrapper.addEventListener('pointerenter', (e) => {
            if (e.pointerType === 'mouse') floatingLoupe.style.display = 'block';
        });

        canvasWrapper.addEventListener('pointerleave', (e) => {
            if (e.pointerType === 'mouse') floatingLoupe.style.display = 'none';
        });

        canvasWrapper.addEventListener('pointerdown', (e) => {
            lastPointerType = e.pointerType;
            if (e.target === pauseBadge) return;

            const coords = getCanvasCoords(e);
            // Check if pressing near the vertical split divider
            if (isNearSplit(coords, e.pointerType)) {
                isDraggingSplit = true;
                canvasWrapper.setPointerCapture(e.pointerId);
                e.preventDefault();
            } else if (e.pointerType !== 'mouse') {
                touchGesture = { startX: e.clientX, startY: e.clientY, scrubbing: false };
            }
        });

        canvasWrapper.addEventListener('pointermove', (e) => {
            const coords = getCanvasCoords(e);

            if (e.pointerType === 'mouse') {
                moveLoupe(coords, 'mouse');
                // Update cursor style
                canvasWrapper.style.cursor = isNearSplit(coords, 'mouse') ? 'col-resize' : 'crosshair';
            }

            if (isDraggingSplit) {
                splitX = Math.max(1, Math.min(imgWidth - 2, Math.round(coords.cssX * imgWidth / canvas.getBoundingClientRect().width)));
                renderCompositeImage();
                return;
            }

            if (e.pointerType === 'mouse') {
                // Update inspected pixel if not locked
                if (!isLocked && isInsideImage(coords)) {
                    inspectPixel(coords.x, coords.y);
                }
            } else if (touchGesture) {
                if (!touchGesture.scrubbing) {
                    // Start scrubbing only on a clear sideways move, not on the jitter of a tap or a scroll
                    const dx = Math.abs(e.clientX - touchGesture.startX);
                    const dy = Math.abs(e.clientY - touchGesture.startY);
                    if (dx < 8 || dx < dy) return;
                    touchGesture.scrubbing = true;
                    isLocked = false;
                    floatingLoupe.style.display = 'block';
                }
                moveLoupe(coords, e.pointerType);
                if (isInsideImage(coords)) {
                    inspectPixel(coords.x, coords.y);
                }
            }
        });

        canvasWrapper.addEventListener('pointerup', (e) => {
            if (isDraggingSplit) {
                isDraggingSplit = false;
                return;
            }
            if (touchGesture) {
                // A tap, or the end of a scrub, locks the pixel under the finger
                touchGesture = null;
                floatingLoupe.style.display = 'none';
                const coords = getCanvasCoords(e);
                if (isInsideImage(coords)) {
                    selectPixel(coords.x, coords.y);
                }
            }
        });

        canvasWrapper.addEventListener('pointercancel', () => {
            // The browser took over the gesture, e.g. to scroll the page
            isDraggingSplit = false;
            touchGesture = null;
            floatingLoupe.style.display = 'none';
        });

        canvasWrapper.addEventListener('click', (e) => {
            if (lastPointerType !== 'mouse') return; // touch and pen are handled on pointerup
            const coords = getCanvasCoords(e);
            // Don't select pixel if dragging the split bar
            if (isNearSplit(coords, 'mouse')) return;

            if (isInsideImage(coords)) {
                if (isLocked && currentX === coords.x && currentY === coords.y) {
                    toggleLock(); // unlock if clicking same pixel
                } else {
                    selectPixel(coords.x, coords.y);
                }
            }
        });

        pauseBadge.addEventListener('click', (e) => {
            e.stopPropagation();
            togglePause();
        });
        updatePauseBadge();

        // Spacebar to pause/resume sampling while the pointer is over the widget (elsewhere it scrolls the page)
        root.addEventListener('mouseenter', () => { isPointerOver = true; });
        root.addEventListener('mouseleave', () => { isPointerOver = false; });
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && isPointerOver) {
                e.preventDefault();
                togglePause();
            }
        });

        // Setup scene switcher buttons
        root.querySelectorAll('[data-scene]').forEach(btn => {
            btn.addEventListener('click', () => {
                switchScene(btn.dataset.scene);
            });
        });

        // Load the first scene only once the widget approaches the viewport (each scene is ~80 MB)
        const initialScene = root.querySelector('[data-scene]').dataset.scene;
        new IntersectionObserver((entries, observer) => {
            if (entries.some(entry => entry.isIntersecting)) {
                observer.disconnect();
                if (!requestedScene) switchScene(initialScene);
            }
        }, { rootMargin: '100% 0px' }).observe(root);

        // Resample only while the widget is on screen
        new IntersectionObserver((entries) => {
            isVisible = entries[entries.length - 1].isIntersecting;
        }).observe(root);

        // Start 1-second live resampling timer
        sampleTimer = setInterval(() => {
            if (!isPaused && isVisible && !document.hidden && dataView && indexTable) {
                generateNewSamples();
                renderCompositeImage();
                drawLoupe();
            }
        }, 1000);
    }

    document.addEventListener('DOMContentLoaded', initVarianceWidget);
})();
