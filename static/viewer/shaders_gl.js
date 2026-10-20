// GLSL ES 3.00 shaders of the WebGL 2 fallback (webgl.js): a port of shaders.js with the same math and the same
// 16-bit and 8-bit quantizations, so that both backends render the same image.
//
// Differences forced by WebGL 2 (no compute shaders, no storage buffers):
// - The preprocessing runs in the vertex shader of the rasterization (per quad corner); culled Gaussians become
//   degenerate quads instead of being left out of a compacted list.
// - The rasterization writes the surviving Gaussian's conic and center per pixel to an extra target, which replaces
//   the per-Gaussian buffer the WebGPU reuse pass reads.
// - Reuse and temporal accumulation run in one full-screen fragment pass; neighbors come from texel fetches instead of
//   workgroup memory, and the K-buffer lives in two RGBA32UI textures per parity.
// - Images are stored with rows from top to bottom as in WebGPU (the rasterization flips y, the display flips back),
//   so pixel coordinates, STBN lookups and reprojection are identical.

const HEADER = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp sampler3D;

layout(std140) uniform FrameBlock {
  mat4 view;          // world -> camera
  mat4 proj;          // camera -> clip
  mat4 invView;       // camera -> world
  mat4 invProj;       // clip -> camera
  mat4 prevViewProj;  // world -> clip of the previous frame
  vec4 camPos;        // camera position in world space
  vec2 viewport;      // render size in pixels
  vec2 focal;         // focal lengths in pixels
  uint splatCount;
  uint frameIdx;
  float stbnShift;    // Cranley-Patterson shift of the current 64-frame STBN cycle
  uint flags;         // 1: spatial reuse, 2: temporal accumulation, 4: white noise, 8: reset history
  float shScale;      // SH1 coefficient = (byte - 128) * shScale
  uint randomSeed;
  uint restFrames;    // frames since the camera last changed, 0 while it moves or with convergence off
  uint pad1;
} frame;

const uint FLAG_SPATIAL = 1u;
const uint FLAG_TEMPORAL = 2u;
const uint FLAG_WHITE_NOISE = 4u;
const uint FLAG_RESET = 8u;

vec4 unpack4x8(uint v) {
  return vec4(float(v & 255u), float((v >> 8u) & 255u), float((v >> 16u) & 255u), float(v >> 24u)) / 255.0;
}
`;

// ---------------------------------------------------------------------------------------------------------------------
// 1. Rasterization with the preprocessing in the vertex shader
// ---------------------------------------------------------------------------------------------------------------------

export const RASTER_VS = HEADER + /* glsl */ `
uniform usampler2D splatsA;   // words 0-3 of the 7 per Gaussian (layout in shaders.js, DECODE_PACKED)
uniform usampler2D splatsB;   // words 4-6
uniform sampler2D chunks;     // 3 texels (12 floats) per chunk of 256: position min, max, log-scale min, max

flat out uint vColor;
flat out uint vId;
flat out uvec4 vGauss;        // conic (f16 x3), center in NDC (f32 x2)
out vec2 vFragPos;

const float SH_C1 = 0.4886025119029199;
const float SQRT8 = 2.8284271247461903;
const float FRUSTUM_DILATION = 0.2;

float byteOf(uint w, uint b) { return float((w >> (8u * b)) & 255u); }
float shCoef(uint w, uint b) { return (byteOf(w, b) - 128.0) * frame.shScale; }
vec2 f16(vec2 v) { return unpackHalf2x16(packHalf2x16(v)); }

uint pack4x8(vec4 v) {
  uvec4 q = uvec4(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
  return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

vec4 chunkTexel(uint c, int k) {
  int t = int(c) * 3 + k;
  int width = textureSize(chunks, 0).x;
  return texelFetch(chunks, ivec2(t % width, t / width), 0);
}

void cull() {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // outside the depth range: no fragments
  vColor = 0u;
  vId = 0u;
  vGauss = uvec4(0u);
  vFragPos = vec2(0.0);
}

void main() {
  uint i = uint(gl_InstanceID);
  int width = textureSize(splatsA, 0).x;
  ivec2 tc = ivec2(gl_InstanceID % width, gl_InstanceID / width);
  uvec4 wa = texelFetch(splatsA, tc, 0);
  uvec4 wb = texelFetch(splatsB, tc, 0);
  uint w0 = wa.x; uint w1 = wa.y; uint w2 = wa.z; uint w3 = wa.w;
  uint w4 = wb.x; uint w5 = wb.y; uint w6 = wb.z;

  float opacity = byteOf(w2, 1u) / 255.0;
  if (opacity < 1.0 / 255.0) { cull(); return; }
  // Position and log scale relative to the chunk's ranges
  uint c = i >> 8u;
  vec4 t0 = chunkTexel(c, 0);
  vec4 t1 = chunkTexel(c, 1);
  vec4 t2 = chunkTexel(c, 2);
  vec3 pmin = t0.xyz;
  vec3 pmax = vec3(t0.w, t1.xy);
  vec3 smin = vec3(t1.zw, t2.x);
  vec3 smax = t2.yzw;
  vec3 center = pmin + vec3(float(w0 & 0xffffu), float(w0 >> 16u), float(w1 & 0xffffu)) / 65535.0 * (pmax - pmin);
  vec3 scale = exp(smin + vec3(byteOf(w1, 2u), byteOf(w1, 3u), byteOf(w2, 0u)) / 255.0 * (smax - smin));
  // Smallest-three quaternion (w, x, y, z)
  uint big = w6 >> 30u;
  vec3 r3 = (vec3(float((w6 >> 20u) & 1023u), float((w6 >> 10u) & 1023u), float(w6 & 1023u)) / 1023.0 * 2.0 - 1.0) * 0.7071067811865476;
  float rbig = sqrt(max(0.0, 1.0 - dot(r3, r3)));
  vec4 rotation;
  if (big == 0u) { rotation = vec4(rbig, r3); }
  else if (big == 1u) { rotation = vec4(r3.x, rbig, r3.y, r3.z); }
  else if (big == 2u) { rotation = vec4(r3.x, r3.y, rbig, r3.z); }
  else { rotation = vec4(r3, rbig); }
  vec3 baseColor = vec3(byteOf(w2, 2u), byteOf(w2, 3u), byteOf(w3, 0u)) / 255.0;
  vec3 sh0 = vec3(shCoef(w3, 1u), shCoef(w3, 2u), shCoef(w3, 3u));
  vec3 sh1 = vec3(shCoef(w4, 0u), shCoef(w4, 1u), shCoef(w4, 2u));
  vec3 sh2 = vec3(shCoef(w4, 3u), shCoef(w5, 0u), shCoef(w5, 1u));

  // Frustum culling as in the reference vertex shader
  vec4 viewPos = frame.view * vec4(center, 1.0);
  vec4 clip = frame.proj * viewPos;
  float lim = (1.0 + FRUSTUM_DILATION) * clip.w;
  if (clip.w <= 0.0 || abs(clip.x) > lim || abs(clip.y) > lim || clip.z < -FRUSTUM_DILATION * clip.w || clip.z > clip.w) { cull(); return; }

  // Covariance R S S^T R^T from scale and rotation (w, x, y, z)
  float qw = rotation.x; float qx = rotation.y; float qy = rotation.z; float qz = rotation.w;
  mat3 R = mat3(
    vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qz * qw), 2.0 * (qx * qz - qy * qw)),
    vec3(2.0 * (qx * qy - qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qx * qw)),
    vec3(2.0 * (qx * qz + qy * qw), 2.0 * (qy * qz - qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)));
  mat3 M = mat3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
  mat3 cov3 = M * transpose(M);

  // Projected covariance (EWA, pixel units) with the 0.3 px low-pass dilation
  float z = viewPos.z;
  mat3 J = mat3(vec3(frame.focal.x / z, 0.0, 0.0), vec3(0.0, frame.focal.y / z, 0.0),
                vec3(-frame.focal.x * viewPos.x / (z * z), -frame.focal.y * viewPos.y / (z * z), 0.0));
  mat3 W = mat3(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
  mat3 T = J * W;
  mat3 cov2 = T * cov3 * transpose(T);
  float a = cov2[0][0] + 0.3;
  float b = cov2[1][0];
  float d = cov2[1][1] + 0.3;

  // Extent basis (sqrt(8) standard deviations along the eigenvectors), as in threedgsProjectedExtentBasis
  float det = a * d - b * b;
  float halfTrace = 0.5 * (a + d);
  float term2 = sqrt(max(0.1, halfTrace * halfTrace - det));
  float l1 = halfTrace + term2;
  float l2 = halfTrace - term2;
  if (l2 <= 0.0 || det <= 0.0) { cull(); return; }
  vec2 e1 = vec2(b, l1 - a);
  float e1len = length(e1);
  e1 = e1len > 1e-12 ? e1 / e1len : vec2(1.0, 0.0);
  vec2 e2 = vec2(e1.y, -e1.x);
  vec2 basis1 = f16(e1 * min(SQRT8 * sqrt(l1), 2048.0));  // stored as f16 in WebGPU
  vec2 basis2 = f16(e2 * min(SQRT8 * sqrt(l2), 2048.0));
  vec3 conic = vec3(d, -b, a) / det;

  // Color: stored base color plus degree-1 SH in the view direction, packed to 8 bits like the reference
  vec3 dir = normalize(center - frame.camPos.xyz);
  vec3 rgb = baseColor + SH_C1 * (-sh0 * dir.y + sh1 * dir.z - sh2 * dir.x);

  vec3 ndc = clip.xyz / clip.w;
  if (ndc.z < 0.0) { cull(); return; }  // WebGPU clips depths below 0 (the quad has the center's depth throughout)

  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float(gl_VertexID >> 1) * 2.0 - 1.0);
  vec2 offsetPx = corner.x * basis1 + corner.y * basis2;
  vec2 p = vec2(ndc.x + offsetPx.x * 2.0 / frame.viewport.x, ndc.y - offsetPx.y * 2.0 / frame.viewport.y);
  // Rows from top to bottom as in WebGPU (y flipped), depth in [0, 1] as in WebGPU
  gl_Position = vec4(p.x, -p.y, ndc.z * 2.0 - 1.0, 1.0);
  vFragPos = corner * SQRT8;
  vColor = pack4x8(vec4(rgb, opacity));
  vId = i;
  vGauss = uvec4(packHalf2x16(conic.xy), packHalf2x16(vec2(conic.z, 0.0)), floatBitsToUint(ndc.x), floatBitsToUint(ndc.y));
}
`;

export const RASTER_FS = HEADER + /* glsl */ `
uniform sampler3D stbn;

flat in uint vColor;
flat in uint vId;
flat in uvec4 vGauss;
in vec2 vFragPos;

layout(location = 0) out vec4 outColor;   // rgb, alpha: the Gaussian's opacity (for the reuse pass)
layout(location = 1) out uint outId;
layout(location = 2) out uvec4 outGauss;

// Per-primitive spatio-temporal blue noise (random.h of the reference): R2 offsets into the 128x128x64 volume,
// Cranley-Patterson rotation per 64-frame cycle
float stbnThreshold(uvec2 pix, uint id) {
  uint ox = (id * 3242174889u) >> 25u;
  uint oy = (id * 2447445415u) >> 25u;
  ivec3 coord = ivec3(int((pix.x + ox) & 127u), int((pix.y + oy) & 127u), int(frame.frameIdx & 63u));
  return fract(texelFetch(stbn, coord, 0).r + frame.stbnShift);
}

uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float whiteThreshold(uvec2 pix, uint id) {
  uint h = pcg(pcg(pcg(pix.x ^ (pix.y << 16u)) + id) + frame.randomSeed);
  return float(h >> 8u) * (1.0 / 16777216.0);
}

void main() {
  float A = dot(vFragPos, vFragPos);
  if (A > 10.0) { discard; }
  vec4 c = unpack4x8(vColor);
  float opacity = exp(-0.5 * A) * c.a;
  if (opacity <= 1.0 / 255.0) { discard; }
  if (opacity < 1.0) {
    uvec2 pix = uvec2(gl_FragCoord.xy);
    float u = (frame.flags & FLAG_WHITE_NOISE) != 0u ? whiteThreshold(pix, vId) : stbnThreshold(pix, vId);
    if (!(u < opacity)) { discard; }
  }
  outColor = c;
  outId = vId;
  outGauss = vGauss;
}
`;

// ---------------------------------------------------------------------------------------------------------------------
// 2. Spatial reuse with the K-buffer, temporal accumulation and the rest average, per pixel (shaders.js, REUSE)
// ---------------------------------------------------------------------------------------------------------------------

export const FULLSCREEN_VS = /* glsl */ `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const REUSE_FS = HEADER + /* glsl */ `
uniform sampler2D colorTex;    // rgb, alpha: the Gaussian's opacity
uniform usampler2D idTex;
uniform sampler2D depthTex;
uniform usampler2D gaussTex;   // conic (f16 x3), center in NDC (f32 x2)
uniform usampler2D kInA;       // K-buffer entries 0 and 1: (id, data, id, data); bits 30-31 of id 0: write index
uniform usampler2D kInB;       // entries 2 and 3
uniform sampler2D histTex;     // rgb: previous output, a: accumulated mass (bilinear)
uniform sampler2D restIn;      // average since the camera stopped

layout(location = 0) out vec4 outColor;
layout(location = 1) out uvec4 outKA;
layout(location = 2) out uvec4 outKB;
layout(location = 3) out vec4 outRest;

const uint K = 4u;
const uint REST_SPATIAL_FRAMES = 8u;   // frames at rest that contribute spatial estimates to the average, then raw samples
const float REST_BLEND = 8.0;          // the rest average takes over with weight n / (n + REST_BLEND)
const uint EMPTY_ID = 0x3FFFFFFFu;     // K-buffer entry x: splat id (30 bits)
const uint NO_SPLAT = 0xFFFFFFFFu;     // id texture: background
const uint EMPTY_DATA = 0xFFFF0000u;   // entry y: depth (f16, low 16 bits) | 4-bit occlusion ratios R, B, L, T (high 16 bits)

ivec2 imgSize;
vec2 tileOriginF;  // origin of the pixel's 16x16 tile minus one pixel, as in the WebGPU workgroups (center quantization)

uint f16bits(float v) { return packHalf2x16(vec2(v, 0.0)) & 0xffffu; }
float f16(float v) { return unpackHalf2x16(packHalf2x16(vec2(v, 0.0))).x; }
bool inImage(ivec2 p) { return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, imgSize)); }
vec4 colorAt(ivec2 p) { return inImage(p) ? texelFetch(colorTex, p, 0) : vec4(0.0); }

struct G2D { vec3 conic; vec2 center; float depth; float opacity; };

// The surviving Gaussian at pixel p, quantized like the WebGPU workgroup memory
G2D gaussAt(ivec2 p) {
  if (texelFetch(idTex, p, 0).r == NO_SPLAT) { return G2D(vec3(0.0), tileOriginF, 0.0, 0.0); }
  uvec4 q = texelFetch(gaussTex, p, 0);
  vec2 ndc = uintBitsToFloat(q.zw);
  vec2 screen = vec2((ndc.x + 1.0) * 0.5 * frame.viewport.x, (1.0 - ndc.y) * 0.5 * frame.viewport.y);
  vec2 rel = unpackHalf2x16(packHalf2x16(screen - tileOriginF));
  return G2D(vec3(unpackHalf2x16(q.x), unpackHalf2x16(q.y).x), rel + tileOriginF, f16(texelFetch(depthTex, p, 0).r),
             texelFetch(colorTex, p, 0).a);
}

float evalGauss(G2D g, vec2 pixel) {
  vec2 d = pixel - g.center;
  float power = -0.5 * (g.conic.x * d.x * d.x + g.conic.z * d.y * d.y) - g.conic.y * d.x * d.y;
  return g.opacity > 0.0 ? exp(power) * g.opacity : 0.0;
}

// K-buffer of pixel p: ids, data and write index
void kBufferAt(ivec2 p, out uvec4 ids, out uvec4 datas, out uint writeIndex) {
  if (!inImage(p)) {
    ids = uvec4(EMPTY_ID);
    datas = uvec4(EMPTY_DATA);
    writeIndex = 0u;
    return;
  }
  uvec4 a = texelFetch(kInA, p, 0);
  uvec4 b = texelFetch(kInB, p, 0);
  ids = uvec4(a.x, a.z, b.x, b.z) & EMPTY_ID;
  datas = uvec4(a.y, a.w, b.y, b.w);
  writeIndex = a.x >> 30u;
}

// Visibility of a candidate sample from neighbor n at the center pixel: product of the occlusion ratios of the
// neighbor's K-buffer entries in front of it (dir: 1 left, 2 right, 3 top, 4 bottom neighbor)
float reuseVisibility(uint candDepth16, uint candId, uvec4 hIds, uvec4 hDatas, uint dir, bool reset) {
  if (reset) { return 1.0; }
  // The ratio toward the center pixel: R for the left neighbor, L for the right one, B for the top, T for the bottom
  uint shift = 0u;
  if (dir == 2u) { shift = 8u; } else if (dir == 3u) { shift = 4u; } else if (dir == 4u) { shift = 12u; }
  float vis = 1.0;
  for (uint k = 0u; k < K; k++) {
    if (vis <= 1e-4) { break; }
    uint hid = hIds[int(k)];
    if (hid != EMPTY_ID && hid != (candId & EMPTY_ID)) {
      uint data = hDatas[int(k)];
      if ((data & 0xffffu) < candDepth16) {
        vis *= float((data >> (16u + shift)) & 15u) / 15.0;
      }
    }
  }
  return vis;
}

vec3 ycocg(vec3 c) {
  float co = (c.r - c.b) * 0.5;
  float t = (c.r + c.b) * 0.25;
  float g2 = c.g * 0.5;
  return vec3(g2 + t, co, g2 - t);
}

// 5-tap Catmull-Rom history fetch built from bilinear taps
vec4 catmullRom(vec2 uv, vec2 texSize) {
  vec2 inv = 1.0 / texSize;
  vec2 sp = uv * texSize;
  vec2 tc = floor(sp - 0.5) + 0.5;
  vec2 f = sp - tc;
  vec2 f2 = f * f;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f2 * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f2 * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 tc12 = tc + w2 / max(w12, vec2(1e-5));
  vec2 tc0 = tc - 1.0;
  vec2 tc3 = tc + 2.0;
  float wTop = w12.x * w0.y;
  float wLeft = w0.x * w12.y;
  float wCenter = w12.x * w12.y;
  float wRight = w3.x * w12.y;
  float wBottom = w12.x * w3.y;
  vec4 c = textureLod(histTex, vec2(tc12.x, tc0.y) * inv, 0.0) * wTop
         + textureLod(histTex, vec2(tc0.x, tc12.y) * inv, 0.0) * wLeft
         + textureLod(histTex, tc12 * inv, 0.0) * wCenter
         + textureLod(histTex, vec2(tc3.x, tc12.y) * inv, 0.0) * wRight
         + textureLod(histTex, vec2(tc12.x, tc3.y) * inv, 0.0) * wBottom;
  return max(c / max(wTop + wLeft + wCenter + wRight + wBottom, 1e-5), vec4(0.0));
}

bool hasNan(vec3 v) { return any(isnan(v)) || any(greaterThan(abs(v), vec3(1e30))); }

void main() {
  ivec2 pos = ivec2(gl_FragCoord.xy);
  imgSize = textureSize(idTex, 0);
  tileOriginF = vec2((pos / 16) * 16 - 1);
  vec2 centerPixel = vec2(pos) + 0.5;
  float currentDepth = texelFetch(depthTex, pos, 0).r;
  uint currentId = texelFetch(idTex, pos, 0).r;
  vec4 currentColor = texelFetch(colorTex, pos, 0);
  bool reset = (frame.flags & FLAG_RESET) != 0u;

  // Reprojection of the current sample into the previous frame
  vec2 uv = centerPixel / frame.viewport;
  vec4 camPt = frame.invProj * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, currentDepth, 1.0);
  camPt = camPt / camPt.w;
  vec4 world = frame.invView * camPt;
  vec4 prevClip = frame.prevViewProj * world;
  vec2 prevNdc = prevClip.xy / max(prevClip.w, 1e-8);
  vec2 prevUV = vec2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
  vec2 margin = 2.0 / frame.viewport;
  bool historyValid = all(greaterThanEqual(prevUV, margin)) && all(lessThanEqual(prevUV, vec2(1.0) - margin));
  float motionPixels = length((uv - prevUV) * frame.viewport);
  float historyMass = 0.0;
  if (historyValid) { historyMass = textureLod(histTex, prevUV, 0.0).a; }
  bool resetHistory = reset || !historyValid || historyMass < 0.5;
  bool resetPacked = reset || !historyValid;

  // Opacity of the current sample at the pixel and occlusion ratios toward the 4 neighbors
  vec4 ratios = vec4(1.0);  // R, L, B, T
  if (currentId != NO_SPLAT) {
    G2D g = gaussAt(pos);
    float alpha = evalGauss(g, centerPixel);
    float invDenom = 1.0 / ((1.0 - alpha) + 1e-8);
    ratios = min(vec4(1.0 - evalGauss(g, centerPixel + vec2(1.0, 0.0)), 1.0 - evalGauss(g, centerPixel + vec2(-1.0, 0.0)),
                      1.0 - evalGauss(g, centerPixel + vec2(0.0, 1.0)), 1.0 - evalGauss(g, centerPixel + vec2(0.0, -1.0))) * invDenom,
                 vec4(100.0));
  }

  // Spatial reuse of the 4 neighbors' samples, weighted by their opacity ratio and K-buffer visibility
  vec4 accum = currentColor;
  float sumW = 1.0;
  float sumW2 = 1.0;
  if ((frame.flags & FLAG_SPATIAL) != 0u) {
    const ivec2 offsets[5] = ivec2[5](ivec2(0, 0), ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    for (uint i = 1u; i < 5u; i++) {
      ivec2 np = pos + offsets[int(i)];
      if (!inImage(np)) { continue; }
      uint nid = texelFetch(idTex, np, 0).r;
      vec4 nColor = texelFetch(colorTex, np, 0);
      uvec4 hIds; uvec4 hDatas; uint hWrite;
      kBufferAt(np, hIds, hDatas, hWrite);
      float w;
      if (nid == NO_SPLAT) {
        w = reuseVisibility(f16bits(65500.0), NO_SPLAT, hIds, hDatas, i, resetPacked);
        if (w <= 1e-4) { continue; }
      } else {
        G2D ng = gaussAt(np);
        w = reuseVisibility(f16bits(ng.depth), nid, hIds, hDatas, i, resetPacked);
        // alpha_n(center) / alpha_n(neighbor) from the quadratic form's change over one pixel
        vec2 dn = (vec2(np) + 0.5) - ng.center;
        float gx = ng.conic.x * dn.x + ng.conic.y * dn.y;
        float gy = ng.conic.z * dn.y + ng.conic.y * dn.x;
        float dq;
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
    uvec4 ids; uvec4 datas; uint writeIndex;
    kBufferAt(pos, ids, datas, writeIndex);
    if (resetPacked) { ids = uvec4(EMPTY_ID); datas = uvec4(EMPTY_DATA); writeIndex = 0u; }
    writeIndex = writeIndex % K;
    bool isBackground = currentId == NO_SPLAT;
    bool duplicate = false;
    for (uint k = 0u; k < K; k++) {
      if (!isBackground && ids[int(k)] != EMPTY_ID && ids[int(k)] == (currentId & EMPTY_ID)) { duplicate = true; }
    }
    if (!duplicate) {
      if (isBackground) {
        ids[int(writeIndex)] = EMPTY_ID;
        datas[int(writeIndex)] = EMPTY_DATA;
      } else {
        uvec4 q = uvec4(clamp(ratios * 15.0 + 0.5, vec4(0.0), vec4(15.0)));  // R, L, B, T
        ids[int(writeIndex)] = currentId & EMPTY_ID;
        datas[int(writeIndex)] = f16bits(currentDepth) | ((q.x | (q.z << 4u) | (q.y << 8u) | (q.w << 12u)) << 16u);
      }
      writeIndex = (writeIndex + 1u) % K;
    }
    outKA = uvec4(ids[0] | (writeIndex << 30u), datas[0], ids[1], datas[1]);
    outKB = uvec4(ids[2], datas[2], ids[3], datas[3]);
  }

  vec4 spatial = accum / max(sumW, 1e-8);
  float essSpatial = (sumW * sumW) / max(sumW2, 1e-6);

  if ((frame.flags & FLAG_TEMPORAL) == 0u) {
    outColor = vec4(spatial.rgb, essSpatial);
    outRest = vec4(0.0);
    return;
  }

  // Temporal accumulation: history clipped against the 3x3 neighborhood of raw samples in YCoCg, with a box that
  // narrows under motion; fixed alpha 0.3, soft rejection by the clip distance
  vec3 m1 = vec3(0.0);
  vec3 m2 = vec3(0.0);
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec3 y = ycocg(colorAt(pos + ivec2(dx, dy)).rgb);
      m1 += y;
      m2 += y * y;
    }
  }
  vec3 mu = m1 / 9.0;
  vec3 sigma = sqrt(max(m2 / 9.0 - mu * mu, vec3(0.0)));
  float velocity = clamp(motionPixels / 4.0, 0.0, 1.0);
  float gamma = mix(4.0, 1.0, velocity);
  vec3 halfExtent = max(gamma * sigma, vec3(0.01, 0.005, 0.005));

  vec3 history = spatial.rgb;
  if (historyValid && !resetHistory) {
    if (motionPixels < 0.005) { history = texelFetch(histTex, pos, 0).rgb; }
    else { history = catmullRom(prevUV, frame.viewport).rgb; }
  }
  if (hasNan(history)) { history = spatial.rgb; }

  float alpha = 1.0;
  float mass = essSpatial;
  if (historyValid && !resetHistory) {
    vec3 v = ycocg(history) - mu;
    float maUnit = max(max(abs(v.x) / max(halfExtent.x, 1e-3), abs(v.y) / max(halfExtent.y, 1e-3)), abs(v.z) / max(halfExtent.z, 1e-3));
    alpha = 0.3;
    mass = 1.0;
    if (maUnit > 1.0) { alpha = mix(1.0, 0.3, 1.0 / maUnit); }
  }
  vec3 result = mix(history, spatial.rgb, alpha);
  if (hasNan(result)) { result = spatial.rgb; mass = essSpatial; }

  // Convergence at rest (see shaders.js): the frames since the camera stopped are averaged in a texture of their own,
  // spatial estimates first, raw samples later; the average takes over with weight n / (n + REST_BLEND)
  outRest = vec4(0.0);
  if (frame.restFrames > 0u) {
    float n = float(frame.restFrames);
    vec3 sampleRgb = frame.restFrames <= REST_SPATIAL_FRAMES ? spatial.rgb : currentColor.rgb;
    vec3 avg = sampleRgb;
    if (frame.restFrames > 1u) { vec3 prev = texelFetch(restIn, pos, 0).rgb; avg = prev + (sampleRgb - prev) / n; }
    outRest = vec4(avg, 1.0);
    result = mix(result, avg, n / (n + REST_BLEND));
  }
  outColor = vec4(result, mass);
}
`;

// ---------------------------------------------------------------------------------------------------------------------
// 3. Display: draws the output texture to the canvas (rows flipped back to the bottom-up default framebuffer)
// ---------------------------------------------------------------------------------------------------------------------

export const DISPLAY_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D image;
in vec2 vUv;
out vec4 outColor;
void main() {
  outColor = vec4(texture(image, vec2(vUv.x, 1.0 - vUv.y)).rgb, 1.0);
}
`;
