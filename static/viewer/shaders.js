// WGSL shaders of the SteadySplats WebGPU viewer, ported from the Vulkan reference renderer (vk_gaussian_splatting:
// threedgs_raster.vert/frag, random.h, spatial_reuse.comp) with the configuration of the paper: 1 sample per pixel,
// per-primitive STBN, history-based spatial reuse (K = 4, |N| = 4, opacity-normalized weights) and the temporal EMA
// with soft history rejection (fixed alpha 0.3).
//
// Conventions: camera space as in COLMAP (x right, y down, z forward), pixel coordinates with y down, WebGPU clip space
// (y up, depth in [0, 1]).

const FRAME = /* wgsl */ `
struct Frame {
  view: mat4x4f,          // world -> camera
  proj: mat4x4f,          // camera -> clip
  invView: mat4x4f,       // camera -> world
  invProj: mat4x4f,       // clip -> camera
  prevViewProj: mat4x4f,  // world -> clip of the previous frame
  camPos: vec4f,          // camera position in world space
  viewport: vec2f,        // render size in pixels
  focal: vec2f,           // focal lengths in pixels
  splatCount: u32,
  frameIdx: u32,
  stbnShift: f32,         // Cranley-Patterson shift of the current 64-frame STBN cycle
  flags: u32,             // 1: spatial reuse, 2: temporal accumulation, 4: white noise, 8: reset history
  shScale: f32,           // SH1 coefficient = (byte - 128) * shScale
  randomSeed: u32,
  restFrames: u32,        // frames since the camera last changed, 0 while it moves or with convergence off
  pad1: u32,
};
@group(0) @binding(0) var<uniform> frame: Frame;

const FLAG_SPATIAL = 1u;
const FLAG_TEMPORAL = 2u;
const FLAG_WHITE_NOISE = 4u;
const FLAG_RESET = 8u;
`;

// ---------------------------------------------------------------------------------------------------------------------
// 1. Preprocess: decode, cull, project, evaluate color; one thread per Gaussian
// ---------------------------------------------------------------------------------------------------------------------

// Decoding of the compact format: 7 words per Gaussian plus a 12-float table per chunk of 256 (see convert_ply.js)
const DECODE_PACKED = /* wgsl */ `
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> chunks: array<f32>;

fn byteOf(w: u32, b: u32) -> f32 { return f32((w >> (8u * b)) & 255u); }
fn shCoef(w: u32, b: u32) -> f32 { return (byteOf(w, b) - 128.0) * frame.shScale; }

fn decodeSplat(i: u32) -> Splat {
  let s = i * 7u;
  let w0 = splats[s]; let w1 = splats[s + 1u]; let w2 = splats[s + 2u]; let w3 = splats[s + 3u];
  let w4 = splats[s + 4u]; let w5 = splats[s + 5u]; let w6 = splats[s + 6u];
  var out: Splat;
  out.opacity = byteOf(w2, 1u) / 255.0;
  // Position and log scale relative to the chunk's ranges
  let c = (i >> 8u) * 12u;
  let pmin = vec3f(chunks[c], chunks[c + 1u], chunks[c + 2u]);
  let pmax = vec3f(chunks[c + 3u], chunks[c + 4u], chunks[c + 5u]);
  out.center = pmin + vec3f(f32(w0 & 0xffffu), f32(w0 >> 16u), f32(w1 & 0xffffu)) / 65535.0 * (pmax - pmin);
  let smin = vec3f(chunks[c + 6u], chunks[c + 7u], chunks[c + 8u]);
  let smax = vec3f(chunks[c + 9u], chunks[c + 10u], chunks[c + 11u]);
  out.scale = exp(smin + vec3f(byteOf(w1, 2u), byteOf(w1, 3u), byteOf(w2, 0u)) / 255.0 * (smax - smin));
  // Smallest-three quaternion (w, x, y, z)
  let big = w6 >> 30u;
  let r3 = (vec3f(f32((w6 >> 20u) & 1023u), f32((w6 >> 10u) & 1023u), f32(w6 & 1023u)) / 1023.0 * 2.0 - 1.0) * 0.7071067811865476;
  let rbig = sqrt(max(0.0, 1.0 - dot(r3, r3)));
  if (big == 0u) { out.rotation = vec4f(rbig, r3); }
  else if (big == 1u) { out.rotation = vec4f(r3.x, rbig, r3.y, r3.z); }
  else if (big == 2u) { out.rotation = vec4f(r3.x, r3.y, rbig, r3.z); }
  else { out.rotation = vec4f(r3, rbig); }
  out.color = vec3f(byteOf(w2, 2u), byteOf(w2, 3u), byteOf(w3, 0u)) / 255.0;
  out.sh0 = vec3f(shCoef(w3, 1u), shCoef(w3, 2u), shCoef(w3, 3u));
  out.sh1 = vec3f(shCoef(w4, 0u), shCoef(w4, 1u), shCoef(w4, 2u));
  out.sh2 = vec3f(shCoef(w4, 3u), shCoef(w5, 0u), shCoef(w5, 1u));
  return out;
}
`;

// Lossless 32-bit layout for validating against the reference renderer (convert_ply.js --float): 23 floats per Gaussian
const DECODE_FLOAT = /* wgsl */ `
@group(0) @binding(1) var<storage, read> splats: array<f32>;
@group(0) @binding(2) var<storage, read> chunks: array<f32>;

fn decodeSplat(i: u32) -> Splat {
  let s = i * 23u;
  var out: Splat;
  out.center = vec3f(splats[s], splats[s + 1u], splats[s + 2u]);
  out.scale = exp(vec3f(splats[s + 3u], splats[s + 4u], splats[s + 5u]));
  out.rotation = normalize(vec4f(splats[s + 6u], splats[s + 7u], splats[s + 8u], splats[s + 9u]));
  out.opacity = splats[s + 10u] + 0.0 * chunks[0];
  out.color = vec3f(splats[s + 11u], splats[s + 12u], splats[s + 13u]);
  out.sh0 = vec3f(splats[s + 14u], splats[s + 15u], splats[s + 16u]);
  out.sh1 = vec3f(splats[s + 17u], splats[s + 18u], splats[s + 19u]);
  out.sh2 = vec3f(splats[s + 20u], splats[s + 21u], splats[s + 22u]);
  return out;
}
`;

const preprocess = (decode) => FRAME + /* wgsl */ `
struct Splat { center: vec3f, scale: vec3f, rotation: vec4f, opacity: f32, color: vec3f, sh0: vec3f, sh1: vec3f, sh2: vec3f };
` + decode + /* wgsl */ `
@group(0) @binding(3) var<storage, read_write> projected: array<vec4u>;  // 2 per Gaussian
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
@group(0) @binding(5) var<storage, read_write> drawArgs: array<atomic<u32>, 4>;

const SH_C1 = 0.4886025119029199;
const SQRT8 = 2.8284271247461903;
const FRUSTUM_DILATION = 0.2;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= frame.splatCount) { return; }
  let sp = decodeSplat(i);
  let opacity = sp.opacity;
  if (opacity < 1.0 / 255.0) { return; }
  let center = sp.center;

  // Frustum culling as in the reference vertex shader
  let viewPos = frame.view * vec4f(center, 1.0);
  let clip = frame.proj * viewPos;
  let lim = (1.0 + FRUSTUM_DILATION) * clip.w;
  if (clip.w <= 0.0 || abs(clip.x) > lim || abs(clip.y) > lim || clip.z < -FRUSTUM_DILATION * clip.w || clip.z > clip.w) { return; }

  // Covariance R S S^T R^T from scale and rotation (w, x, y, z)
  let scale = sp.scale;
  let qw = sp.rotation.x; let qx = sp.rotation.y; let qy = sp.rotation.z; let qz = sp.rotation.w;
  let R = mat3x3f(
    vec3f(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qz * qw), 2.0 * (qx * qz - qy * qw)),
    vec3f(2.0 * (qx * qy - qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qx * qw)),
    vec3f(2.0 * (qx * qz + qy * qw), 2.0 * (qy * qz - qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)));
  let M = mat3x3f(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
  let cov3 = M * transpose(M);

  // Projected covariance (EWA, pixel units) with the 0.3 px low-pass dilation
  let z = viewPos.z;
  let J = mat3x3f(vec3f(frame.focal.x / z, 0.0, 0.0), vec3f(0.0, frame.focal.y / z, 0.0),
                  vec3f(-frame.focal.x * viewPos.x / (z * z), -frame.focal.y * viewPos.y / (z * z), 0.0));
  let W = mat3x3f(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
  let T = J * W;
  let cov2 = T * cov3 * transpose(T);
  let a = cov2[0][0] + 0.3;
  let b = cov2[1][0];
  let d = cov2[1][1] + 0.3;

  // Extent basis (sqrt(8) standard deviations along the eigenvectors), as in threedgsProjectedExtentBasis
  let det = a * d - b * b;
  let half = 0.5 * (a + d);
  let term2 = sqrt(max(0.1, half * half - det));
  let l1 = half + term2;
  let l2 = half - term2;
  if (l2 <= 0.0 || det <= 0.0) { return; }
  var e1 = vec2f(b, l1 - a);
  let e1len = length(e1);
  e1 = select(vec2f(1.0, 0.0), e1 / e1len, e1len > 1e-12);
  let e2 = vec2f(e1.y, -e1.x);
  let basis1 = e1 * min(SQRT8 * sqrt(l1), 2048.0);
  let basis2 = e2 * min(SQRT8 * sqrt(l2), 2048.0);
  let conic = vec3f(d, -b, a) / det;

  // Color: stored base color plus degree-1 SH in the view direction, packed to 8 bits like the reference
  let dir = normalize(center - frame.camPos.xyz);
  let rgb = sp.color + SH_C1 * (-sp.sh0 * dir.y + sp.sh1 * dir.z - sp.sh2 * dir.x);
  let color = pack4x8unorm(vec4f(rgb, opacity));

  let ndc = clip.xyz / clip.w;
  projected[2u * i] = vec4u(bitcast<u32>(ndc.x), bitcast<u32>(ndc.y), bitcast<u32>(ndc.z), color);
  projected[2u * i + 1u] = vec4u(pack2x16float(basis1), pack2x16float(basis2), pack2x16float(conic.xy), pack2x16float(vec2f(conic.z, 0.0)));
  visible[atomicAdd(&drawArgs[1], 1u)] = i;
}
`;

export const PREPROCESS = preprocess(DECODE_PACKED);
export const PREPROCESS_FLOAT = preprocess(DECODE_FLOAT);

// ---------------------------------------------------------------------------------------------------------------------
// 2. Stochastic rasterization: one quad per visible Gaussian; a fragment survives if its threshold is below its
//    opacity, and the depth test keeps the closest surviving one (no sorting, no blending)
// ---------------------------------------------------------------------------------------------------------------------

export const RASTER = FRAME + /* wgsl */ `
@group(0) @binding(1) var<storage, read> projected: array<vec4u>;
@group(0) @binding(2) var<storage, read> visible: array<u32>;
@group(0) @binding(3) var stbn: texture_3d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) fragPos: vec2f,
  @location(1) @interpolate(flat) color: u32,
  @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let id = visible[ii];
  let p0 = projected[2u * id];
  let p1 = projected[2u * id + 1u];
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  let offsetPx = corner.x * unpack2x16float(p1.x) + corner.y * unpack2x16float(p1.y);
  var out: VSOut;
  out.pos = vec4f(bitcast<f32>(p0.x) + offsetPx.x * 2.0 / frame.viewport.x,
                  bitcast<f32>(p0.y) - offsetPx.y * 2.0 / frame.viewport.y,
                  bitcast<f32>(p0.z), 1.0);
  out.fragPos = corner * 2.8284271247461903;
  out.color = p0.w;
  out.id = id;
  return out;
}

// Per-primitive spatio-temporal blue noise (random.h of the reference): R2 offsets into the 128x128x64 volume,
// Cranley-Patterson rotation per 64-frame cycle
fn stbnThreshold(pix: vec2u, id: u32) -> f32 {
  let ox = (id * 3242174889u) >> 25u;
  let oy = (id * 2447445415u) >> 25u;
  let coord = vec3i(i32((pix.x + ox) & 127u), i32((pix.y + oy) & 127u), i32(frame.frameIdx & 63u));
  return fract(textureLoad(stbn, coord, 0).r + frame.stbnShift);
}

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn whiteThreshold(pix: vec2u, id: u32) -> f32 {
  let h = pcg(pcg(pcg(pix.x ^ (pix.y << 16u)) + id) + frame.randomSeed);
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

struct FSOut {
  @location(0) color: vec4f,
  @location(1) id: u32,
};

@fragment
fn fs(in: VSOut) -> FSOut {
  let A = dot(in.fragPos, in.fragPos);
  if (A > 10.0) { discard; }
  let c = unpack4x8unorm(in.color);
  let opacity = exp(-0.5 * A) * c.a;
  if (opacity <= 1.0 / 255.0) { discard; }
  if (opacity < 1.0) {
    let pix = vec2u(in.pos.xy);
    var u: f32;
    if ((frame.flags & FLAG_WHITE_NOISE) != 0u) { u = whiteThreshold(pix, in.id); } else { u = stbnThreshold(pix, in.id); }
    if (!(u < opacity)) { discard; }
  }
  return FSOut(vec4f(c.rgb, 1.0), in.id);
}
`;

// ---------------------------------------------------------------------------------------------------------------------
// 3. Spatial reuse with the K-buffer, then temporal accumulation; 16x16 pixels per workgroup with a 1 px apron in
//    workgroup memory (spatial_reuse.comp of the reference, paper configuration only)
// ---------------------------------------------------------------------------------------------------------------------

export const REUSE = FRAME + /* wgsl */ `
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var idTex: texture_2d<u32>;
@group(0) @binding(3) var depthTex: texture_depth_2d;
@group(0) @binding(4) var<storage, read> projected: array<vec4u>;
@group(0) @binding(5) var<storage, read> kIn: array<vec2u>;          // 4 per pixel
@group(0) @binding(6) var<storage, read_write> kOut: array<vec2u>;
@group(0) @binding(7) var histTex: texture_2d<f32>;                   // rgb: previous output, a: accumulated mass
@group(0) @binding(8) var histSampler: sampler;
@group(0) @binding(9) var outTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var<storage, read_write> restAccum: array<vec4f>;  // average since the camera stopped

const K = 4u;
const REST_SPATIAL_FRAMES = 8u;   // frames at rest that contribute spatial estimates to the average, then raw samples
const REST_BLEND = 8.0;           // the rest average takes over with weight n / (n + REST_BLEND)
const EMPTY_ID = 0x3FFFFFFFu;     // K-buffer entry x: splat id (30 bits); bits 30-31 of entry 0: write index
const NO_SPLAT = 0xFFFFFFFFu;     // id texture: background
const EMPTY_DATA = 0xFFFF0000u;   // entry y: depth (f16, low 16 bits) | 4-bit occlusion ratios R, B, L, T (high 16 bits)
const TILE = 16u;
const EXTENT = 18u;
const STRIDE = 19u;               // padded row stride against bank conflicts
const CAP = 342u;                 // EXTENT * STRIDE
const SLICE = 343u;               // K-major slices with one pad entry

var<workgroup> sColor: array<u32, 342>;
var<workgroup> sId: array<u32, 342>;
var<workgroup> sGauss: array<vec4u, 342>;   // conic (f16 x3), center relative to the tile (f16 x2), depth (f16), opacity (f32)
var<workgroup> sKId: array<u32, 1372>;
var<workgroup> sKData: array<u32, 1372>;
var<workgroup> sKWrite: array<u32, 342>;

fn f16bits(v: f32) -> u32 { return pack2x16float(vec2f(v, 0.0)) & 0xffffu; }

fn loadTexel(loadIndex: u32, tileOrigin: vec2i, size: vec2i) {
  let ty = loadIndex / EXTENT;
  let tx = loadIndex - ty * EXTENT;
  let gp = tileOrigin + vec2i(i32(tx), i32(ty));
  let li = ty * STRIDE + tx;
  if (all(gp >= vec2i(0)) && all(gp < size)) {
    let id = textureLoad(idTex, gp, 0).r;
    sId[li] = id;
    sColor[li] = pack4x8unorm(textureLoad(colorTex, gp, 0));
    let depth = textureLoad(depthTex, gp, 0);
    if (id != NO_SPLAT) {
      let p0 = projected[2u * id];
      let p1 = projected[2u * id + 1u];
      let screen = vec2f((bitcast<f32>(p0.x) + 1.0) * 0.5 * frame.viewport.x, (1.0 - bitcast<f32>(p0.y)) * 0.5 * frame.viewport.y);
      let rel = screen - vec2f(tileOrigin);
      let conicXY = unpack2x16float(p1.z);
      let conicZ = unpack2x16float(p1.w).x;
      let opacity = f32(p0.w >> 24u) / 255.0;
      sGauss[li] = vec4u(pack2x16float(conicXY), pack2x16float(vec2f(conicZ, rel.x)), pack2x16float(vec2f(rel.y, depth)), bitcast<u32>(opacity));
    } else {
      sGauss[li] = vec4u(0u);
    }
    let base = u32(gp.y * size.x + gp.x) * K;
    for (var k = 0u; k < K; k++) {
      let e = kIn[base + k];
      sKId[k * SLICE + li] = e.x & EMPTY_ID;
      sKData[k * SLICE + li] = e.y;
      if (k == 0u) { sKWrite[li] = e.x >> 30u; }
    }
  } else {
    sId[li] = NO_SPLAT;
    sColor[li] = 0u;
    sGauss[li] = vec4u(0u);
    for (var k = 0u; k < K; k++) {
      sKId[k * SLICE + li] = EMPTY_ID;
      sKData[k * SLICE + li] = EMPTY_DATA;
    }
    sKWrite[li] = 0u;
  }
}

struct G2D { conic: vec3f, center: vec2f, depth: f32, opacity: f32 };

fn unpackGauss(li: u32, tileOrigin: vec2f) -> G2D {
  let p = sGauss[li];
  let a = unpack2x16float(p.x);
  let b = unpack2x16float(p.y);
  let c = unpack2x16float(p.z);
  return G2D(vec3f(a.x, a.y, b.x), vec2f(b.y, c.x) + tileOrigin, c.y, bitcast<f32>(p.w));
}

fn evalGauss(g: G2D, pixel: vec2f) -> f32 {
  let d = pixel - g.center;
  let power = -0.5 * (g.conic.x * d.x * d.x + g.conic.z * d.y * d.y) - g.conic.y * d.x * d.y;
  return select(0.0, exp(power) * g.opacity, g.opacity > 0.0);
}

// Visibility of a candidate sample from neighbor n at the center pixel: product of the occlusion ratios of the
// neighbor's K-buffer entries in front of it (dir: 1 left, 2 right, 3 top, 4 bottom neighbor)
fn reuseVisibility(candDepth16: u32, candId: u32, nIdx: u32, dir: u32, reset: bool) -> f32 {
  if (reset) { return 1.0; }
  // The ratio toward the center pixel: R for the left neighbor, L for the right one, B for the top, T for the bottom
  var shift = 0u;
  if (dir == 2u) { shift = 8u; } else if (dir == 3u) { shift = 4u; } else if (dir == 4u) { shift = 12u; }
  var vis = 1.0;
  for (var k = 0u; k < K; k++) {
    if (vis <= 1e-4) { break; }
    let hid = sKId[k * SLICE + nIdx];
    if (hid != EMPTY_ID && hid != (candId & EMPTY_ID)) {
      let data = sKData[k * SLICE + nIdx];
      if ((data & 0xffffu) < candDepth16) {
        vis *= f32((data >> (16u + shift)) & 15u) / 15.0;
      }
    }
  }
  return vis;
}

fn ycocg(c: vec3f) -> vec3f {
  let co = (c.r - c.b) * 0.5;
  let t = (c.r + c.b) * 0.25;
  let g2 = c.g * 0.5;
  return vec3f(g2 + t, co, g2 - t);
}

fn rgbFromYcocg(c: vec3f) -> vec3f {
  let t = c.x - c.z;
  return vec3f(t + c.y, c.x + c.z, t - c.y);
}

// 5-tap Catmull-Rom history fetch built from bilinear taps
fn catmullRom(uv: vec2f, texSize: vec2f) -> vec4f {
  let inv = 1.0 / texSize;
  let sp = uv * texSize;
  let tc = floor(sp - 0.5) + 0.5;
  let f = sp - tc;
  let f2 = f * f;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f2 * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f2 * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let tc12 = tc + w2 / max(w12, vec2f(1e-5));
  let tc0 = tc - 1.0;
  let tc3 = tc + 2.0;
  let wTop = w12.x * w0.y;
  let wLeft = w0.x * w12.y;
  let wCenter = w12.x * w12.y;
  let wRight = w3.x * w12.y;
  let wBottom = w12.x * w3.y;
  let c = textureSampleLevel(histTex, histSampler, vec2f(tc12.x, tc0.y) * inv, 0.0) * wTop
        + textureSampleLevel(histTex, histSampler, vec2f(tc0.x, tc12.y) * inv, 0.0) * wLeft
        + textureSampleLevel(histTex, histSampler, tc12 * inv, 0.0) * wCenter
        + textureSampleLevel(histTex, histSampler, vec2f(tc3.x, tc12.y) * inv, 0.0) * wRight
        + textureSampleLevel(histTex, histSampler, vec2f(tc12.x, tc3.y) * inv, 0.0) * wBottom;
  return max(c / max(wTop + wLeft + wCenter + wRight + wBottom, 1e-5), vec4f(0.0));
}

fn hasNan(v: vec3f) -> bool { return any(v != v) || any(abs(v) > vec3f(1e30)); }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let size = vec2i(textureDimensions(idTex));
  let tileOrigin = vec2i(wid.xy) * i32(TILE) - vec2i(1);
  let tileOriginF = vec2f(tileOrigin);
  let linear = lid.y * TILE + lid.x;
  loadTexel(linear, tileOrigin, size);
  if (linear < 68u) { loadTexel(linear + 256u, tileOrigin, size); }
  workgroupBarrier();

  let pos = vec2i(gid.xy);
  if (any(pos >= size)) { return; }
  let cIdx = (lid.y + 1u) * STRIDE + lid.x + 1u;
  let centerPixel = vec2f(pos) + 0.5;
  let currentDepth = textureLoad(depthTex, pos, 0);
  let currentId = sId[cIdx];
  let currentColor = unpack4x8unorm(sColor[cIdx]);
  let reset = (frame.flags & FLAG_RESET) != 0u;

  // Reprojection of the current sample into the previous frame
  let uv = centerPixel / frame.viewport;
  var camPt = frame.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, currentDepth, 1.0);
  camPt = camPt / camPt.w;
  let world = frame.invView * camPt;
  let prevClip = frame.prevViewProj * world;
  let prevNdc = prevClip.xy / max(prevClip.w, 1e-8);
  let prevUV = vec2f(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
  let margin = 2.0 / frame.viewport;
  let historyValid = all(prevUV >= margin) && all(prevUV <= vec2f(1.0) - margin);
  let motionPixels = length((uv - prevUV) * frame.viewport);
  var historyMass = 0.0;
  if (historyValid) { historyMass = textureSampleLevel(histTex, histSampler, prevUV, 0.0).a; }
  let resetHistory = reset || !historyValid || historyMass < 0.5;
  let resetPacked = reset || !historyValid;

  // Opacity of the current sample at the pixel and occlusion ratios toward the 4 neighbors
  var g = G2D(vec3f(0.0), vec2f(0.0), 0.0, 0.0);
  var ratios = vec4f(1.0);  // R, L, B, T
  if (currentId != NO_SPLAT) {
    g = unpackGauss(cIdx, tileOriginF);
    let alpha = evalGauss(g, centerPixel);
    let invDenom = 1.0 / ((1.0 - alpha) + 1e-8);
    ratios = min(vec4f(1.0 - evalGauss(g, centerPixel + vec2f(1.0, 0.0)), 1.0 - evalGauss(g, centerPixel + vec2f(-1.0, 0.0)),
                       1.0 - evalGauss(g, centerPixel + vec2f(0.0, 1.0)), 1.0 - evalGauss(g, centerPixel + vec2f(0.0, -1.0))) * invDenom,
                 vec4f(100.0));
  }

  // Spatial reuse of the 4 neighbors' samples, weighted by their opacity ratio and K-buffer visibility
  var accum = currentColor;
  var sumW = 1.0;
  var sumW2 = 1.0;
  if ((frame.flags & FLAG_SPATIAL) != 0u) {
    let offsets = array<vec2i, 5>(vec2i(0, 0), vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1));
    for (var i = 1u; i < 5u; i++) {
      let np = pos + offsets[i];
      if (any(np < vec2i(0)) || any(np >= size)) { continue; }
      let nIdx = u32(i32(cIdx) + offsets[i].y * i32(STRIDE) + offsets[i].x);
      let nid = sId[nIdx];
      let nColor = unpack4x8unorm(sColor[nIdx]);
      var w: f32;
      if (nid == NO_SPLAT) {
        w = reuseVisibility(f16bits(65500.0), NO_SPLAT, nIdx, i, resetPacked);
        if (w <= 1e-4) { continue; }
      } else {
        let ng = unpackGauss(nIdx, tileOriginF);
        w = reuseVisibility(f16bits(ng.depth), nid, nIdx, i, resetPacked);
        // alpha_n(center) / alpha_n(neighbor) from the quadratic form's change over one pixel
        let dn = (vec2f(np) + 0.5) - ng.center;
        let gx = ng.conic.x * dn.x + ng.conic.y * dn.y;
        let gy = ng.conic.z * dn.y + ng.conic.y * dn.x;
        var dq: f32;
        if (i == 1u) { dq = -gx - 0.5 * ng.conic.x; }
        else if (i == 2u) { dq = gx - 0.5 * ng.conic.x; }
        else if (i == 3u) { dq = -gy - 0.5 * ng.conic.z; }
        else { dq = gy - 0.5 * ng.conic.z; }
        w *= exp(clamp(dq, -10.0, 10.0));
      }
      accum += nColor * w;
      sumW += w;
      sumW2 += w * w;
    }
  }

  // Append the current sample to this pixel's K-buffer (ring of the K most recent distinct samples)
  {
    var ids: array<u32, 4>;
    var datas: array<u32, 4>;
    var writeIndex = 0u;
    for (var k = 0u; k < K; k++) {
      ids[k] = select(sKId[k * SLICE + cIdx], EMPTY_ID, resetPacked);
      datas[k] = select(sKData[k * SLICE + cIdx], EMPTY_DATA, resetPacked);
    }
    if (!resetPacked) { writeIndex = sKWrite[cIdx] % K; }
    let isBackground = currentId == NO_SPLAT;
    var duplicate = false;
    for (var k = 0u; k < K; k++) {
      if (!isBackground && ids[k] != EMPTY_ID && ids[k] == (currentId & EMPTY_ID)) { duplicate = true; }
    }
    if (!duplicate) {
      if (isBackground) {
        ids[writeIndex] = EMPTY_ID;
        datas[writeIndex] = EMPTY_DATA;
      } else {
        let q = vec4u(clamp(ratios * 15.0 + 0.5, vec4f(0.0), vec4f(15.0)));  // R, L, B, T
        ids[writeIndex] = currentId & EMPTY_ID;
        datas[writeIndex] = f16bits(currentDepth) | ((q.x | (q.z << 4u) | (q.y << 8u) | (q.w << 12u)) << 16u);
      }
      writeIndex = (writeIndex + 1u) % K;
    }
    let base = u32(pos.y * size.x + pos.x) * K;
    for (var k = 0u; k < K; k++) {
      kOut[base + k] = vec2u(ids[k] | select(0u, writeIndex << 30u, k == 0u), datas[k]);
    }
  }

  let spatial = accum / max(sumW, 1e-8);
  let essSpatial = (sumW * sumW) / max(sumW2, 1e-6);

  if ((frame.flags & FLAG_TEMPORAL) == 0u) {
    textureStore(outTex, pos, vec4f(spatial.rgb, essSpatial));
    return;
  }

  // Temporal accumulation: history clipped against the 3x3 neighborhood of raw samples in YCoCg, with a box that
  // narrows under motion; fixed alpha 0.3, soft rejection by the clip distance
  var m1 = vec3f(0.0);
  var m2 = vec3f(0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let y = ycocg(unpack4x8unorm(sColor[u32(i32(cIdx) + dy * i32(STRIDE) + dx)]).rgb);
      m1 += y;
      m2 += y * y;
    }
  }
  let mu = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3f(0.0)));
  let velocity = saturate(motionPixels / 4.0);
  let gamma = mix(4.0, 1.0, velocity);
  let halfExtent = max(gamma * sigma, vec3f(0.01, 0.005, 0.005));

  var history = spatial.rgb;
  if (historyValid && !resetHistory) {
    if (motionPixels < 0.005) { history = textureLoad(histTex, pos, 0).rgb; }
    else { history = catmullRom(prevUV, frame.viewport).rgb; }
  }
  if (hasNan(history)) { history = spatial.rgb; }

  var alpha = 1.0;
  var mass = essSpatial;
  if (historyValid && !resetHistory) {
    let v = ycocg(history) - mu;
    let maUnit = max(max(abs(v.x) / max(halfExtent.x, 1e-3), abs(v.y) / max(halfExtent.y, 1e-3)), abs(v.z) / max(halfExtent.z, 1e-3));
    alpha = 0.3;
    mass = 1.0;
    if (maUnit > 1.0) { alpha = mix(1.0, 0.3, 1.0 / maUnit); }
  }
  var result = mix(history, spatial.rgb, alpha);
  if (hasNan(result)) { result = spatial.rgb; mass = essSpatial; }

  // Convergence at rest: the frames since the camera stopped are averaged in a buffer of their own, so nothing from
  // the motion phase (ghosting) survives in it. The first frames contribute spatial estimates (low noise), later ones
  // raw samples (unbiased), so the slight blur of the spatial reuse fades as REST_SPATIAL_FRAMES / n. The average takes
  // over with weight n / (n + REST_BLEND), and the blend is stored as history, so that motion starts from it.
  if (frame.restFrames > 0u) {
    let n = f32(frame.restFrames);
    let sample = select(currentColor.rgb, spatial.rgb, frame.restFrames <= REST_SPATIAL_FRAMES);
    let idx = u32(pos.y * size.x + pos.x);
    var avg = sample;
    if (frame.restFrames > 1u) { let prev = restAccum[idx].rgb; avg = prev + (sample - prev) / n; }
    restAccum[idx] = vec4f(avg, 1.0);
    result = mix(result, avg, n / (n + REST_BLEND));
  }
  textureStore(outTex, pos, vec4f(result, mass));
}
`;

// ---------------------------------------------------------------------------------------------------------------------
// 4. Display: draws the output texture to the canvas
// ---------------------------------------------------------------------------------------------------------------------

export const DISPLAY = /* wgsl */ `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: VSOut;
  out.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(p.x, 1.0 - p.y);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(image, imageSampler, in.uv, 0.0).rgb, 1.0);
}
`;
