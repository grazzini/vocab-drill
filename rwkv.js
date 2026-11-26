/* rwkv.js: RWKV spaced-repetition model (anki-rwkv "two-head" family: RWKV-Curve + RWKV-Instant) in plain JS.
 *
 * Runs as a Web Worker (messages below) and loads in Node for tests (module.exports).
 * Ported from the MIT NumPy reference of JSchoreels/anki-rwkv (reference/small_model_recurrent.py and
 * INTEGRATION.md of the rwkv-family-2026-11-28 release). The weights are NOT part of this repo: they are
 * fetched at run time from a pinned commit of the Anki fork JSchoreels/anki (personal research use).
 *
 * Worker messages
 *   {type:"init", size:"tiny"|"small", weights?:"int8"|"f32", baseUrl?}
 *       -> {type:"ready", size, modelId, bytes}
 *   {type:"replay", log, dayStartHour, tzOffsetMinutes, instant?}
 *       -> {type:"curves", curves:{[cardKey]: params}, preds:[...], predsInstant?:[...], ms}
 *   {type:"append", entries, instant?}          (after a replay; same reply shape, only changed cards)
 *   any failure -> {type:"error", message}
 * log entry: [t (unix s, answer time), wordId, card (0|1), rating 1..4, ms, flag?]; flag 1 = synthetic, skipped.
 * tzOffsetMinutes: minutes to ADD to UTC to get local time (JST = +540; that is -new Date().getTimezoneOffset()).
 * params: {w:[3], s:[3], d:[3], at}: R(t) = sum w_i (1 + t/s_i)^-d_i, t = seconds since `at`.
 */
"use strict";
(function (root) {

// ---------------------------------------------------------------- pinned model files
const RWKV_PIN = "36c4f39d7a9880063656981b7d1f35dd23a38007"; // JSchoreels/anki main, 2026-11-26
const RWKV_BASE = "https://raw.githubusercontent.com/JSchoreels/anki/" + RWKV_PIN + "/qt/aqt/rwkv_inference/";
const RWKV_MODELS = {
  // Small two-head: show-time two-head cool-down (SM118), architecture "w64", 331,511 parameters
  small: {
    id: "sm118", dir: "sm118",
    files: {
      int8: { name: "weights.int8.bin", bytes: 436391, sha256: "12d3c2d658738e6132a05deadb4014f482fa9d42da2c64d0f0c7112f3f7672c6" },
      f32: { name: "weights.f32.bin", bytes: 1351553, sha256: "379168c80e8d546ebac892adda49e9a012b71c92b44cf8c9f94d4202f2530c3c" },
    },
    // stream: [n_layers, layers without a channel mixer]; streams = card, deck, note, preset, user
    streams: [[2, [1]], [4, [1, 2]], [1, []], [3, [0, 1, 2]], [3, [0, 1, 2]]],
    steps: [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [3, 1], [4, 1], [1, 2], [3, 2], [4, 2], [1, 3]],
  },
  // Tiny two-head: six-layer Tiny two-head model (SM130, "sm130th"), architecture "w64-c1-d2p1u1", 185,057 parameters
  tiny: {
    id: "sm130th", dir: "sm130th",
    files: {
      int8: { name: "weights.int8.bin", bytes: 238585, sha256: "cef6ac7391eb611ae36046b324b7e189693c1982b42f4cf4212569ab3f765220" },
      f32: { name: "weights.f32.bin", bytes: 752409, sha256: "4d64bc9c1dc2432e016596ade4d3575933ad74bbbad3cc4672977bc871c7c232" },
    },
    streams: [[1, []], [2, [1]], [1, []], [1, [0]], [1, [0]]],
    steps: [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [1, 1]],
  },
};
// Shared by both models (architecture.json "constants" and widths)
const D = 64, H = 4, K = 16, NF = 92, FC = 128;
const LN_EPS = 1e-5, GN_EPS = 0.00064, NORM_EPS = 1e-12, SOFTPLUS_T = 20, PROB_EPS = 1e-5, POW_CLAMP = 25, S_EPS = 1e-7;
const MAX_DURATION_MS = 60000; // Anki's default "maximum answer seconds" (the dataset caps durations there)

// ---------------------------------------------------------------- recall curve (self-contained; copy to the page)
function rwkvRecall(p, nowSeconds) {
  const t = Math.max(nowSeconds - p.at, 1);
  let r = 0;
  for (let i = 0; i < p.w.length; i++) r += p.w[i] * Math.exp(-p.d[i] * Math.log1p(t / (1e-7 + p.s[i])));
  const raw = 1e-5 + (1 - 2e-5) * r;
  return 1 / (1 + Math.exp(-(Math.log(raw / (1 - raw)) + 1e-5)));
}
function rwkvDue(p, target) {
  // first time (unix s) at which rwkvRecall(p, time) <= target; R is non-increasing in t
  const MAX = 100 * 365 * 86400;
  const R = (t) => rwkvRecall(p, p.at + t);
  if (R(1) <= target) return p.at + 1;
  let lo = 1, hi = 2;
  while (hi < MAX && R(hi) > target) { lo = hi; hi *= 2; }
  if (hi >= MAX) { hi = MAX; if (R(hi) > target) return p.at + MAX; }
  for (let i = 0; i < 60 && hi - lo > 1; i++) {
    const mid = Math.sqrt(lo * hi);
    if (R(mid) > target) lo = mid; else hi = mid;
  }
  return p.at + Math.ceil(hi);
}

// ---------------------------------------------------------------- weights container
function parseWeights(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  const text = (a, n) => { let s = ""; for (let i = 0; i < n; i++) s += String.fromCharCode(u8[a + i]); return s; };
  let pos, int8;
  if (text(0, 16) === "ARWKVWEIGHTS8PC1") { pos = 16; int8 = true; }
  else if (text(0, 13) === "ARWKVWEIGHTS1") { pos = 13; int8 = false; }
  else throw new Error("rwkv: unknown weights magic");
  const count = dv.getUint32(pos, true); pos += 4;
  const out = {};
  const dec = new TextDecoder();
  for (let n = 0; n < count; n++) {
    const len = dv.getUint16(pos, true); pos += 2;
    const name = dec.decode(u8.subarray(pos, pos + len)); pos += len;
    const kind = int8 ? u8[pos++] : 0;
    const rank = u8[pos++];
    const dims = [];
    for (let r = 0; r < rank; r++) { dims.push(dv.getUint32(pos, true)); pos += 4; }
    const size = dims.reduce((a, b) => a * b, 1);
    let arr;
    if (kind === 1) {
      const rows = dims[0], cols = size / rows;
      const scale = new Float32Array(buffer.slice(pos, pos + 4 * rows)); pos += 4 * rows;
      const codes = new Int8Array(buffer, pos, size); pos += size;
      arr = new Float32Array(size);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) arr[r * cols + c] = codes[r * cols + c] * scale[r];
    } else if (kind === 0) {
      arr = new Float32Array(buffer.slice(pos, pos + 4 * size)); pos += 4 * size;
    } else throw new Error("rwkv: bad tensor kind");
    out[name] = arr;
  }
  if (pos !== u8.length) throw new Error("rwkv: trailing bytes in weights");
  return out;
}

// ---------------------------------------------------------------- math helpers (Float64 scratch, Float32 weights)
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
function matvec(W, b, x, out, nOut, nIn) {
  for (let o = 0, r = 0; o < nOut; o++, r += nIn) {
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, i = 0;
    for (; i + 7 < nIn; i += 8) {
      const q = r + i;
      s0 += W[q] * x[i]; s1 += W[q + 1] * x[i + 1]; s2 += W[q + 2] * x[i + 2]; s3 += W[q + 3] * x[i + 3];
      s4 += W[q + 4] * x[i + 4]; s5 += W[q + 5] * x[i + 5]; s6 += W[q + 6] * x[i + 6]; s7 += W[q + 7] * x[i + 7];
    }
    for (; i < nIn; i++) s0 += W[r + i] * x[i];
    out[o] = (b ? b[o] : 0) + (((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
  }
}
function layerNorm(x, w, b, out, n) {
  let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
  const inv = 1 / Math.sqrt(v + LN_EPS);
  for (let i = 0; i < n; i++) out[i] = (x[i] - m) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------- model
function buildModel(size, tensors) {
  const cfg = RWKV_MODELS[size];
  const T = (name) => { const t = tensors[name]; if (!t) throw new Error("rwkv: missing tensor " + name); return t; };
  const streams = cfg.streams.map(([nLayers, stripped], i) => {
    const layers = [];
    let off = 0;
    for (let l = 0; l < nLayers; l++) {
      const p = `rwkv_modules.${i}.blocks.${l}.time_mixer.`;
      const c = `rwkv_modules.${i}.blocks.${l}.channel_mixer.`;
      const mixer = !stripped.includes(l);
      const L = {
        ln_w: T(p + "layer_norm.weight"), ln_b: T(p + "layer_norm.bias"), lerp: T(p + "rkvdag_lerp"),
        bonus: T(p + "bonus"), Wr: T(p + "W_r.weight"), Wk: T(p + "W_k.weight"), Wv: T(p + "W_v.weight"), Wo: T(p + "W_o.weight"),
        ks_w: T(p + "k_scale_linear.weight"), ks_b: T(p + "k_scale_linear.bias"),
        vs_w: T(p + "v_scale_linear.weight"), vs_b: T(p + "v_scale_linear.bias"),
        vA: l ? T(p + "v_lora_simple.A.weight") : null, vB: l ? T(p + "v_lora_simple.B_and_lamb.weight") : null,
        vBb: l ? T(p + "v_lora_simple.B_and_lamb.bias") : null,
        aA: T(p + "a_lora_simple.A.weight"), aB: T(p + "a_lora_simple.B_and_lamb.weight"), aBb: T(p + "a_lora_simple.B_and_lamb.bias"),
        dA: T(p + "d_lora_mlp.A.weight"), dB: T(p + "d_lora_mlp.B_and_lamb.weight"), dBb: T(p + "d_lora_mlp.B_and_lamb.bias"),
        gA: T(p + "lora_A_g.weight"), gB: T(p + "lora_B_g.weight"),
        gn_w: T(p + "out_group_norm.weight"), gn_b: T(p + "out_group_norm.bias"),
        mixer,
        c_ln_w: mixer ? T(c + "layer_norm.weight") : null, c_ln_b: mixer ? T(c + "layer_norm.bias") : null,
        c_lerp: mixer ? T(c + "lerp_k") : null, c_Wk: mixer ? T(c + "W_k.weight") : null, c_Wv: mixer ? T(c + "W_v.weight") : null,
        vRank: l ? T(p + "v_lora_simple.A.weight").length / D : 0,
        aRank: T(p + "a_lora_simple.A.weight").length / D,
        dRank: T(p + "d_lora_mlp.A.weight").length / D,
        gRank: T(p + "lora_A_g.weight").length / D,
        cDim: mixer ? T(c + "W_k.weight").length / D : 0,
        offShift: off, offS: off + D, offC: off + D + H * K * K,
      };
      off += D + H * K * K + (mixer ? D : 0);
      layers.push(L);
    }
    return { layers, stateSize: off };
  });
  return {
    size, id: cfg.id, steps: cfg.steps, streams,
    tmpState: new Float32Array(streams.reduce((a, s) => a + s.stateSize, 0)), // a query row's working copy
    f0_w: T("features2card.0.weight"), f0_b: T("features2card.0.bias"),
    f2_w: T("features2card.2.weight"), f2_b: T("features2card.2.bias"),
    f3_w: T("features2card.3.weight"), f3_b: T("features2card.3.bias"),
    pre_w: T("prehead_norm.weight"), pre_b: T("prehead_norm.bias"),
    hw0_w: T("head_w.0.weight"), hw0_b: T("head_w.0.bias"),
    hw2_w: T("head_w.2.weight"), hw2_b: T("head_w.2.bias"),
    hw4_w: T("head_w.4.weight"), hw4_b: T("head_w.4.bias"),
    gw_w: T("gru_w_weight"), gw_b: T("gru_w_bias"), gs_w: T("gru_s_weight"), gs_b: T("gru_s_bias"),
    gd_w: T("gru_d_weight"), gd_b: T("gru_d_bias"),
    hp0_w: T("head_p.0.weight"), hp0_b: T("head_p.0.bias"), pl_w: T("p_linear.weight"), pl_b: T("p_linear.bias"),
  };
}

// scratch buffers (one model runs at a time per thread)
const F64 = (n) => new Float64Array(n);
const sc = {
  f: F64(NF), h128: F64(FC), h128b: F64(FC), x: F64(D), xl: F64(D), mix: F64(8 * D),
  r: F64(D), k: F64(D), v: F64(D), a: F64(D), g: F64(D), w: F64(D), tmp: F64(FC), lo: F64(16), lo2: F64(D),
  ks: F64(H), vs: F64(H), kd: F64(D), vv: F64(D), kb: F64(D), out: F64(D), y: F64(D),
  v0: [F64(D), F64(D), F64(D), F64(D), F64(D)],
  heads: { w: F64(3), s: F64(3), d: F64(3), p: F64(4), pImm: 0 },
};

// One RWKV-7 time mixer (+ channel mixer) on entity state `st` (Float32Array) for layer L, in place.
function runLayer(L, layerIdx, st, has, v0) {
  const x = sc.x, xl = sc.xl, mix = sc.mix;
  layerNorm(x, L.ln_w, L.ln_b, xl, D);
  const sh = L.offShift;
  for (let j = 0; j < 8; j++) {
    const lw = L.lerp, base = j * D;
    for (let c = 0; c < D; c++) { const a = xl[c], b = has ? st[sh + c] : a; mix[base + c] = a + lw[base + c] * (b - a); }
  }
  const m = (j) => mix.subarray(j * D, j * D + D);
  matvec(L.Wr, null, m(0), sc.r, D, D);
  matvec(L.Wk, null, m(1), sc.k, D, D);
  matvec(L.ks_w, L.ks_b, m(6), sc.ks, H, D);
  matvec(L.vs_w, L.vs_b, m(7), sc.vs, H, D);
  for (let h = 0; h < H; h++) { sc.ks[h] = sigmoid(sc.ks[h]); sc.vs[h] = sigmoid(sc.vs[h]); }
  const v = sc.v;
  matvec(L.Wv, null, m(2), v, D, D);
  if (layerIdx === 0) v0.set(v);
  else {
    matvec(L.vA, null, m(2), sc.lo, L.vRank, D);
    matvec(L.vB, L.vBb, sc.lo, sc.lo2, D, L.vRank);
    for (let c = 0; c < D; c++) { const t = sigmoid(sc.lo2[c]); v[c] = v[c] + t * (v0[c] - v[c]); }
  }
  matvec(L.aA, null, m(4), sc.lo, L.aRank, D);
  matvec(L.aB, L.aBb, sc.lo, sc.a, D, L.aRank);
  for (let c = 0; c < D; c++) sc.a[c] = sigmoid(sc.a[c]);
  matvec(L.gA, null, m(5), sc.lo, L.gRank, D);
  for (let i = 0; i < L.gRank; i++) sc.lo[i] = sigmoid(sc.lo[i]);
  matvec(L.gB, null, sc.lo, sc.g, D, L.gRank);
  matvec(L.dA, null, m(3), sc.lo, L.dRank, D);
  for (let i = 0; i < L.dRank; i++) sc.lo[i] = Math.tanh(sc.lo[i]);
  matvec(L.dB, L.dBb, sc.lo, sc.w, D, L.dRank);
  for (let c = 0; c < D; c++) {
    const nd = -sc.w[c];
    const sp = nd > SOFTPLUS_T ? nd : Math.log1p(Math.exp(nd));
    sc.w[c] = Math.exp(-Math.exp(-0.5 - sp));
  }
  const r = sc.r, k = sc.k, a = sc.a, w = sc.w, kd = sc.kd, vv = sc.vv, kb = sc.kb, out = sc.out, bonus = L.bonus;
  const S0 = L.offS;
  for (let h = 0; h < H; h++) {
    const o = h * K;
    let nk = 0, nv = 0;
    for (let j = 0; j < K; j++) { nk += k[o + j] * k[o + j]; nv += v[o + j] * v[o + j]; }
    const fk = sc.ks[h] / Math.max(Math.sqrt(nk), NORM_EPS), fv = sc.vs[h] / Math.max(Math.sqrt(nv), NORM_EPS);
    let bon = 0;
    for (let j = 0; j < K; j++) {
      kd[o + j] = k[o + j] * fk; vv[o + j] = v[o + j] * fv; kb[o + j] = kd[o + j] * a[o + j];
      bon += r[o + j] * bonus[o + j] * kb[o + j];
    }
    // S'[i][j] = S[i][j] w[j] + (vv[i] - sum_j' S[i][j'] kd[j']) kb[j];  out[i] = sum_j S'[i][j] r[j]
    for (let i = 0; i < K; i++) {
      const row = S0 + (o + i) * K;
      let sk = 0;
      if (has) for (let j = 0; j < K; j++) sk += st[row + j] * kd[o + j];
      const coef = vv[o + i] - sk;
      let acc = 0;
      for (let j = 0; j < K; j++) {
        const s = (has ? st[row + j] * w[o + j] : 0) + coef * kb[o + j];
        st[row + j] = s;
        acc += Math.fround(s) * r[o + j];
      }
      out[o + i] = acc;
    }
    // group norm over this head, then bonus
    let mean = 0; for (let i = 0; i < K; i++) mean += out[o + i]; mean /= K;
    let vr = 0; for (let i = 0; i < K; i++) { const d = out[o + i] - mean; vr += d * d; } vr /= K;
    const inv = 1 / Math.sqrt(vr + GN_EPS);
    for (let i = 0; i < K; i++) {
      const c = o + i;
      sc.y[c] = sc.g[c] * ((out[c] - mean) * inv * L.gn_w[c] + L.gn_b[c] + bon * vv[c]);
    }
  }
  matvec(L.Wo, null, sc.y, out, D, D);
  for (let c = 0; c < D; c++) { x[c] += out[c]; st[sh + c] = xl[c]; }
  if (L.mixer) {
    layerNorm(x, L.c_ln_w, L.c_ln_b, xl, D);
    const cs = L.offC, lk = L.c_lerp, mk = sc.mix;
    for (let c = 0; c < D; c++) { const a2 = xl[c], b2 = has ? st[cs + c] : a2; mk[c] = a2 + lk[c] * (b2 - a2); }
    matvec(L.c_Wk, null, mk, sc.tmp, L.cDim, D);
    for (let i = 0; i < L.cDim; i++) { const t = sc.tmp[i] > 0 ? sc.tmp[i] : 0; sc.tmp[i] = t * t; }
    matvec(L.c_Wv, null, sc.tmp, out, D, L.cDim);
    for (let c = 0; c < D; c++) { x[c] += out[c]; st[cs + c] = xl[c]; }
  }
}

// One review row. features: Float64Array(92). ents: 5 entity objects {st: Float32Array, has: bool}.
// skip = true: a query row (no state is changed). Fills sc.heads.
function step(model, features, ents, skip) {
  const f = sc.f; f.set(features); f[22] = 0;
  matvec(model.f0_w, model.f0_b, f, sc.h128, FC, NF);
  for (let i = 0; i < FC; i++) { const t = sc.h128[i]; sc.h128[i] = t * sigmoid(t); }
  layerNorm(sc.h128, model.f2_w, model.f2_b, sc.h128b, FC);
  matvec(model.f3_w, model.f3_b, sc.h128b, sc.x, D, FC);
  for (let i = 0; i < D; i++) { const t = sc.x[i]; sc.x[i] = t * sigmoid(t); }
  // a skipped row works on copies of the stream states (pending updates are thrown away)
  let work = ents;
  if (skip) {
    work = []; let off = 0;
    for (let i = 0; i < 5; i++) {
      const n = model.streams[i].stateSize;
      const view = model.tmpState.subarray(off, off + n); off += n;
      if (ents[i].has) view.set(ents[i].st);
      work.push({ st: view, has: ents[i].has });
    }
  }
  for (const [i, l] of model.steps) runLayer(model.streams[i].layers[l], l, work[i].st, work[i].has, sc.v0[i]);
  if (!skip) for (let i = 0; i < 5; i++) ents[i].has = true;
  // heads
  const hx = sc.xl; layerNorm(sc.x, model.pre_w, model.pre_b, hx, D);
  matvec(model.hw0_w, model.hw0_b, hx, sc.out, D, D);
  for (let i = 0; i < D; i++) if (sc.out[i] < 0) sc.out[i] = 0;
  layerNorm(sc.out, model.hw2_w, model.hw2_b, sc.y, D);
  matvec(model.hw4_w, model.hw4_b, sc.y, sc.h128, FC, D);
  const hd = sc.heads;
  matvec(model.gw_w, model.gw_b, sc.h128, hd.w, 3, FC);
  matvec(model.gs_w, model.gs_b, sc.h128, hd.s, 3, FC);
  matvec(model.gd_w, model.gd_b, sc.h128, hd.d, 3, FC);
  const mx = Math.max(hd.w[0], hd.w[1], hd.w[2]);
  let z = 0; for (let i = 0; i < 3; i++) { hd.w[i] = Math.exp(hd.w[i] - mx); z += hd.w[i]; }
  for (let i = 0; i < 3; i++) {
    hd.w[i] /= z;
    hd.s[i] = Math.exp(Math.min(Math.max(hd.s[i], -POW_CLAMP), POW_CLAMP));
    hd.d[i] = Math.exp(Math.min(Math.max(hd.d[i], -POW_CLAMP), POW_CLAMP));
  }
  matvec(model.hp0_w, model.hp0_b, hx, sc.h128b, FC, D);
  for (let i = 0; i < FC; i++) if (sc.h128b[i] < 0) sc.h128b[i] = 0;
  matvec(model.pl_w, model.pl_b, sc.h128b, hd.p, 4, FC);
  const pm = Math.max(hd.p[0], hd.p[1], hd.p[2], hd.p[3]);
  let pz = 0; for (let i = 0; i < 4; i++) { hd.p[i] = Math.exp(hd.p[i] - pm); pz += hd.p[i]; }
  for (let i = 0; i < 4; i++) hd.p[i] /= pz;
  hd.pImm = Math.min(Math.max(1 - hd.p[0], PROB_EPS), 1 - PROB_EPS);
  return hd;
}
function curveParams(hd, at) {
  return { w: [hd.w[0], hd.w[1], hd.w[2]], s: [hd.s[0], hd.s[1], hd.s[2]], d: [hd.d[0], hd.d[1], hd.d[2]], at };
}

// ---------------------------------------------------------------- 92 input features (INTEGRATION.md 6.4-6.6, 6.10)
const Lg = (x) => (x === -1 ? 0 : Math.log(1 + 1e-5 + x));
const TAU = 2 * Math.PI;
const PERIODS = [3, 7, 30, 100, 365, 3650, 36500];
const mod = (a, n) => ((a % n) + n) % n;

// Fixed random entity code (values in {-1.5,-0.5,0.5,1.5}) from a string key: FNV-1a seed + mulberry32.
function entityCode(key, n) {
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  let s = h >>> 0;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) & 3) - 1.5;
  }
  return out;
}

// Stateful feature builder: the per-user counters of the dataset pipeline. Days are day_offset (0 = user's first day).
class FeatureState {
  constructor() {
    this.th = 0; this.newTotal = 0; this.lastRowDay = null;
    this.curDay = null; this.newToday = 0; this.revToday = 0;
    this.cards = new Map();
  }
  _fill(f, c, day, es, ed, cumD, cumS, dur, rating, query, dd, newSince, revSince, newToday, revToday, codes) {
    f[0] = (Lg(ed) - 1.51) / 1.62;
    f[1] = (Lg(cumD) - 2.14) / 2.25;
    f[2] = (Lg(es) - 9.96) / 5.21;
    const a1 = TAU * mod(es, 86400) / 86400; f[3] = Math.sin(a1); f[4] = Math.cos(a1);
    f[5] = (Lg(cumS) - 10.86) / 5.80;
    const a2 = TAU * mod(cumS, 86400) / 86400; f[6] = Math.sin(a2); f[7] = Math.cos(a2);
    f[8] = query ? 0 : (Math.log(10 + dur) - 8.9) / 1.07;
    for (let i = 0; i < 4; i++) f[9 + i] = !query && rating === i + 1 ? 1 : 0;
    f[13] = f[14] = f[15] = 0;
    f[16] = Math.log(Math.log(Math.E + dd));
    f[17] = (mod(day, 7) - 3) / 3;
    f[18] = (Math.log(3 + newSince) - 2.945) / 2.011;
    f[19] = (Math.log(3 + revSince) - 4.64) / 2.59;
    f[20] = (Math.log(3 + newToday) - 2.55) / 1.41;
    f[21] = (Math.log(3 + revToday) - 4.59) / 1.30;
    f[22] = 0; f[23] = query ? 1 : 0;
    if (codes) {
      let o = 24;
      for (const part of codes) for (let i = 0; i < part.length; i++) f[o++] = part[i];
      for (const p of PERIODS) {
        const q = TAU / p;
        f[o++] = Math.sin(q * mod(day, p)); f[o++] = Math.cos(q * mod(day, p));
        f[o++] = Math.sin(q * mod(c.firstDay, p)); f[o++] = Math.cos(q * mod(c.firstDay, p));
      }
    }
  }
  _today(day) { return this.curDay === day ? [this.newToday, this.revToday] : [0, 0]; }
  // Query row ("card shown now"): no mutation. showSeconds = floor(max(now - last answer, 0)).
  query(f, key, day, showSeconds, codes) {
    const c = this.cards.get(key);
    if (!c) throw new Error("rwkv: query on a new card");
    const ed = day - c.lastDay;
    const [nt, rt] = this._today(day);
    this._fill(f, c, day, showSeconds, ed, c.cumD + ed, c.cumS + showSeconds, 0, 0, true,
      this.lastRowDay === null ? 0 : day - this.lastRowDay, this.newTotal - c.newAtLast, Math.max(0, this.th - c.lastTh), nt, rt, codes);
  }
  // Update row (the answer). elapsedSeconds = answer-to-answer seconds (ignored on a first review).
  update(f, key, day, elapsedSeconds, rating, durationMs, codes) {
    let c = this.cards.get(key);
    const first = !c;
    if (this.curDay !== day) { this.curDay = day; this.newToday = 0; this.revToday = 0; }
    const dd = this.lastRowDay === null ? 0 : day - this.lastRowDay; // days since the previous review of any card
    if (first) {
      c = { lastDay: day, firstDay: day, cumD: -1, cumS: -1, lastTh: 0, newAtLast: 0 };
      this.cards.set(key, c);
      this.newToday++; this.newTotal++;
      this._fill(f, c, day, -1, -1, -1, -1, durationMs, rating, false, dd, 0, 0, this.newToday, this.revToday, codes);
    } else {
      const ed = day - c.lastDay;
      c.cumD += ed; c.cumS += elapsedSeconds;
      this._fill(f, c, day, elapsedSeconds, ed, c.cumD, c.cumS, durationMs, rating, false, dd,
        this.newTotal - c.newAtLast, Math.max(0, this.th - c.lastTh), this.newToday, this.revToday, codes);
    }
    this.revToday++; this.th++;
    this.lastRowDay = day; c.lastDay = day; c.lastTh = this.th; c.newAtLast = this.newTotal;
  }
}

// ---------------------------------------------------------------- engine: the app's log -> curves
const STREAM_KEYS = ["card", "deck", "note", "preset", "user"];
class Engine {
  constructor(model) { this.model = model; this.reset(0, 0); }
  reset(dayStartHour, tzOffsetMinutes) {
    this.dayShift = (tzOffsetMinutes || 0) * 60 - (dayStartHour || 0) * 3600;
    this.firstDayIndex = null; this.fs = new FeatureState();
    this.ent = STREAM_KEYS.map(() => new Map());
    this.cards = new Map(); // cardKey -> {lastT, curve}
    this.codes = new Map();
    this.durLogSum = 0; this.durN = 0; this.lastT = -Infinity;
    this.feat = new Float64Array(NF);
  }
  _entity(i, key) {
    let e = this.ent[i].get(key);
    if (!e) { e = { st: new Float32Array(this.model.streams[i].stateSize), has: false }; this.ent[i].set(key, e); }
    return e;
  }
  _code(kind, key, n) {
    const k = kind + ":" + key;
    let c = this.codes.get(k);
    if (!c) { c = entityCode(k, n); this.codes.set(k, c); }
    return c;
  }
  _duration(ms) {
    // known durations: capped at 60 s; unknown (0): the running geometric mean of the known ones (8 s before any)
    if (ms > 0) {
      const d = Math.min(ms, MAX_DURATION_MS);
      this.durLogSum += Math.log(d); this.durN++;
      return d;
    }
    return this.durN ? Math.round(Math.exp(this.durLogSum / this.durN)) : 8000;
  }
  // entries: log rows; returns {changed:Set, preds, predsInstant}
  process(entries, instant) {
    const n = entries.length;
    const order = [...Array(n).keys()].sort((a, b) => entries[a][0] - entries[b][0] || a - b);
    const preds = new Array(n).fill(null);
    const predsInstant = instant ? new Array(n).fill(null) : null;
    const changed = new Set();
    const f = this.feat;
    for (const idx of order) {
      const e = entries[idx];
      const [t, id, card, rating, ms, flag] = e;
      if (flag === 1 || !(rating >= 1 && rating <= 4)) continue;
      const cardKey = id + "#" + card;
      const dayIndex = Math.floor((t + this.dayShift) / 86400);
      if (this.firstDayIndex === null) this.firstDayIndex = dayIndex;
      const day = dayIndex - this.firstDayIndex;
      const dur = this._duration(ms);
      const codes = [this._code("c", cardKey, 12), this._code("n", id, 12), this._code("d", "deck", 8), this._code("p", "preset", 8)];
      const ents = [this._entity(0, cardKey), this._entity(1, "deck"), this._entity(2, id), this._entity(3, "preset"), this._entity(4, "user")];
      let c = this.cards.get(cardKey);
      let es = 0;
      if (c) {
        es = Math.max(0, t - c.lastT);
        const show = Math.floor(Math.max(es - dur / 1000, 0));
        preds[idx] = rwkvRecall(c.curve, c.curve.at + show);
        if (instant) {
          this.fs.query(f, cardKey, day, show, codes);
          predsInstant[idx] = step(this.model, f, ents, true).pImm;
        }
      }
      this.fs.update(f, cardKey, day, es, rating, dur, codes);
      const hd = step(this.model, f, ents, false);
      if (!c) { c = { lastT: t, curve: null }; this.cards.set(cardKey, c); }
      c.lastT = t; c.curve = curveParams(hd, t);
      changed.add(cardKey);
      if (t > this.lastT) this.lastT = t;
    }
    return { changed, preds, predsInstant };
  }
  curves(keys) {
    const out = {};
    for (const k of keys || this.cards.keys()) out[k] = this.cards.get(k).curve;
    return out;
  }
}

// ---------------------------------------------------------------- loading
function hex(buf) { return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join(""); }
function loadFromBuffer(size, buffer) {
  if (!RWKV_MODELS[size]) throw new Error("rwkv: unknown size " + size);
  const model = buildModel(size, parseWeights(buffer));
  model.bytes = buffer.byteLength;
  return model;
}
async function loadModel(size, opts) {
  opts = opts || {};
  const cfg = RWKV_MODELS[size];
  if (!cfg) throw new Error("rwkv: unknown size " + size);
  const file = cfg.files[opts.weights || "int8"];
  const url = (opts.baseUrl || RWKV_BASE) + cfg.dir + "/" + file.name;
  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error("rwkv: fetch " + url + " -> " + res.status);
  const buf = await res.arrayBuffer();
  if (root.crypto && root.crypto.subtle) {
    const got = hex(await root.crypto.subtle.digest("SHA-256", buf));
    if (got !== file.sha256) throw new Error("rwkv: sha256 mismatch for " + url);
  }
  return loadFromBuffer(size, buf);
}

// ---------------------------------------------------------------- worker glue
const api = {
  RWKV_PIN, RWKV_BASE, RWKV_MODELS, rwkvRecall, rwkvDue, parseWeights, buildModel, loadModel, loadFromBuffer,
  step, curveParams, FeatureState, Engine, entityCode,
};
if (typeof module !== "undefined" && module.exports) module.exports = api;
else root.RWKV = api;

if (typeof self !== "undefined" && typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  let engine = null, model = null;
  self.onmessage = async (ev) => {
    const msg = ev.data || {};
    try {
      if (msg.type === "init") {
        model = await loadModel(msg.size || "tiny", { weights: msg.weights, baseUrl: msg.baseUrl });
        engine = null;
        self.postMessage({ type: "ready", size: model.size, modelId: model.id, bytes: model.bytes });
      } else if (msg.type === "replay" || msg.type === "append") {
        if (!model) throw new Error("rwkv: not initialised");
        const t0 = performance.now();
        if (msg.type === "replay" || !engine) {
          engine = new Engine(model);
          engine.reset(msg.dayStartHour ?? 5, msg.tzOffsetMinutes ?? 0);
        }
        const r = engine.process(msg.type === "replay" ? msg.log : msg.entries, !!msg.instant);
        const reply = { type: "curves", curves: engine.curves(r.changed), preds: r.preds, ms: performance.now() - t0 };
        if (r.predsInstant) reply.predsInstant = r.predsInstant;
        self.postMessage(reply);
      }
    } catch (err) {
      self.postMessage({ type: "error", message: String((err && err.message) || err) });
    }
  };
}
})(typeof globalThis !== "undefined" ? globalThis : this);
