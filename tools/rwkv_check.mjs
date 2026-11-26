#!/usr/bin/env node
// Validate rwkv.js against the anki-rwkv package test vectors, and benchmark it.
//
//   node tools/rwkv_check.mjs check <package dir> [<package dir> ...] [--rows N]
//       package dir = an unzipped rwkv-{small,tiny}-two-head package of release rwkv-family-2026-11-28
//       (gh release download rwkv-family-2026-11-28 -R JSchoreels/anki-rwkv -p 'rwkv-*-two-head.zip').
//       Runs both test-vector users with fp32 and int8 weights (recurrent form, query + update rows) and
//       prints max |difference| of p_curve (label time + 6 sweep times), p_imm, p_rating; then checks the
//       feature builder (features 0..21, 23) against the vectors' raw_* columns. Exit code 1 on failure.
//   node tools/rwkv_check.mjs bench <weights dir or URL base> [--sizes tiny,small] [--n 10000,50000,100000] [--instant]
//       Replays synthetic logs (~3000 words x 2 cards) through Engine and times it.
//       <weights dir> holds sm118/ and sm130th/ (as in the fork); omit to fetch from the pinned GitHub commit.
//   node tools/rwkv_check.mjs dumplog <n> <out.json>      write the synthetic log used by bench (for the WebKit page)
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// rwkv.js is a classic script (worker); load it as CommonJS whatever package.json "type" is in scope
const R = (() => { const m = { exports: {} }; new Function("module", fs.readFileSync(path.join(ROOT, "rwkv.js"), "utf8"))(m); return m.exports; })();

// ---------------------------------------------------------------- .npz reader (zip + .npy)
function readNpz(file) {
  const buf = fs.readFileSync(file);
  let eocd = buf.length - 22;
  while (eocd >= 0 && buf.readUInt32LE(eocd) !== 0x06054b50) eocd--;
  if (eocd < 0) throw new Error("not a zip: " + file);
  const entries = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const out = {};
  for (let n = 0; n < entries; n++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error("bad central directory");
    const method = buf.readUInt16LE(p + 10), csize = buf.readUInt32LE(p + 20);
    const nlen = buf.readUInt16LE(p + 28), xlen = buf.readUInt16LE(p + 30), clen = buf.readUInt16LE(p + 32);
    const local = buf.readUInt32LE(p + 42);
    const name = buf.toString("utf8", p + 46, p + 46 + nlen);
    p += 46 + nlen + xlen + clen;
    const lnl = buf.readUInt16LE(local + 26), lxl = buf.readUInt16LE(local + 28);
    const raw = buf.subarray(local + 30 + lnl + lxl, local + 30 + lnl + lxl + csize);
    const data = method === 8 ? zlib.inflateRawSync(raw) : method === 0 ? raw : null;
    if (!data) throw new Error("zip method " + method);
    out[name.replace(/\.npy$/, "")] = parseNpy(data);
  }
  return out;
}
function parseNpy(b) {
  const major = b[6];
  const hl = major === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const ho = major === 1 ? 10 : 12;
  const header = b.toString("latin1", ho, ho + hl);
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const body = b.subarray(ho + hl);
  const ab = body.buffer.slice(body.byteOffset, body.byteOffset + body.length);
  const T = { "<f4": Float32Array, "<f8": Float64Array, "<i8": BigInt64Array, "|u1": Uint8Array, "|i1": Int8Array }[descr];
  if (!T) return { descr, shape, data: null };
  return { descr, shape, data: new T(ab) };
}

// ---------------------------------------------------------------- check
function runVectors(model, v, rows) {
  const n = Math.min(rows ?? Infinity, v.features.shape[0]);
  const streams = ["card_id", "deck_id", "note_id", "preset_id", "user_id"];
  const stores = streams.map(() => new Map());
  const sweep = Array.from(v.sweep_seconds.data);
  const pre = "fp32";
  const maxd = { p_curve: 0, p_curve_sweep: 0, p_imm: 0, p_rating: 0 };
  const f = new Float64Array(92);
  return (prefix) => {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 92; j++) f[j] = v.features.data[i * 92 + j];
      const ents = streams.map((s, k) => {
        const key = v["id_" + s].data[i].toString();
        let e = stores[k].get(key);
        if (!e) { e = { st: new Float32Array(model.streams[k].stateSize), has: false }; stores[k].set(key, e); }
        return e;
      });
      const hd = R.step(model, f, ents, v.skip.data[i] === 1);
      const params = R.curveParams(hd, 0);
      const upd = (k, val) => { if (!(val <= maxd[k])) maxd[k] = Number.isNaN(val) ? Infinity : val; };
      upd("p_curve", Math.abs(R.rwkvRecall(params, v.label_elapsed_seconds.data[i]) - v[prefix + "_p_curve"].data[i]));
      for (let j = 0; j < sweep.length; j++)
        upd("p_curve_sweep", Math.abs(R.rwkvRecall(params, sweep[j]) - v[prefix + "_p_curve_sweep"].data[i * sweep.length + j]));
      upd("p_imm", Math.abs(hd.pImm - v[prefix + "_p_imm"].data[i]));
      for (let j = 0; j < 4; j++) upd("p_rating", Math.abs(hd.p[j] - v[prefix + "_p_rating"].data[i * 4 + j]));
    }
    return { n, maxd };
  };
}

function checkFeatures(v) {
  // Rebuild features 0..21 and 23 from the raw columns with FeatureState (22 is zeroed by the model,
  // 24..91 are random codes / random calendar baselines in the vectors).
  const fs_ = new R.FeatureState();
  const n = v.features.shape[0];
  const f = new Float64Array(92);
  let worst = 0, worstAt = "";
  for (let i = 0; i < n; i++) {
    const key = v.id_card_id.data[i].toString();
    const day = v.raw_day_offset.data[i];
    if (v.is_query.data[i]) fs_.query(f, key, day, v.raw_elapsed_seconds.data[i], null);
    else fs_.update(f, key, day, v.raw_elapsed_seconds.data[i], v.raw_rating.data[i], v.raw_duration.data[i], null);
    for (let j = 0; j < 24; j++) {
      if (j === 22) continue;
      const d = Math.abs(f[j] - v.features.data[i * 92 + j]);
      if (d > worst) { worst = d; worstAt = `row ${i} feature ${j}: ${f[j]} vs ${v.features.data[i * 92 + j]}`; }
    }
  }
  return { worst, worstAt };
}

async function check(dirs, rows) {
  let failed = false;
  for (const dir of dirs) {
    const arch = JSON.parse(fs.readFileSync(path.join(dir, "architecture.json"), "utf8"));
    const size = arch.architecture === "w64" ? "small" : arch.architecture === "w64-c1-d2p1u1" ? "tiny" : null;
    if (!size) throw new Error("unexpected architecture " + arch.architecture);
    const cfg = R.RWKV_MODELS[size];
    const sameSpec = JSON.stringify(arch.layer_steps) === JSON.stringify(cfg.steps) &&
      arch.streams.every((s, i) => s.n_layers === cfg.streams[i][0] &&
        JSON.stringify(s.channel_mixer_stripped_layers) === JSON.stringify(cfg.streams[i][1]));
    console.log(`${path.basename(dir)}: ${size} (${cfg.id}), embedded spec matches architecture.json: ${sameSpec}`);
    failed ||= !sameSpec;
    const users = fs.readdirSync(path.join(dir, "testvectors")).filter((f) => /^user_\d+\.npz$/.test(f)).sort();
    for (const u of users) {
      const v = readNpz(path.join(dir, "testvectors", u));
      for (const [prefix, file, tol] of [["fp32", "weights.f32.bin", 1e-5], ["int8", "weights.int8.bin", 1e-5]]) {
        const buf = fs.readFileSync(path.join(dir, file));
        const model = R.loadFromBuffer(size, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
        const t0 = performance.now();
        const { n, maxd } = runVectors(model, v, rows)(prefix);
        const ms = performance.now() - t0;
        const worst = Math.max(...Object.values(maxd));
        const ok = worst < tol;
        failed ||= !ok;
        console.log(`  ${ok ? "PASS" : "FAIL"} ${u} ${prefix} rows ${n}: max|d| curve ${maxd.p_curve.toExponential(1)} ` +
          `sweep ${maxd.p_curve_sweep.toExponential(1)} imm ${maxd.p_imm.toExponential(1)} rating ${maxd.p_rating.toExponential(1)} ` +
          `(${(ms / n).toFixed(3)} ms/row)`);
      }
      const fc = checkFeatures(v);
      const ok = fc.worst < 1e-5;
      failed ||= !ok;
      console.log(`  ${ok ? "PASS" : "FAIL"} ${u} feature builder, features 0..21,23: max|d| ${fc.worst.toExponential(1)}` +
        (ok ? "" : "  worst " + fc.worstAt));
    }
  }
  console.log("RESULT:", failed ? "FAIL" : "PASS");
  process.exit(failed ? 1 : 0);
}

// ---------------------------------------------------------------- synthetic log + bench
function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export function syntheticLog(total, seed = 1) {
  // ~3000 words x 2 cards, 20 new words/day, interval doubling-ish, reviews spread over the waking hours of JST.
  const rand = rng(seed);
  const log = [];
  const cards = []; // {id, card, due, ivl}
  let nextWord = 0;
  const t0 = 1767225600 + 9 * 3600; // 2026-01-01 09:00 UTC = 18:00 JST
  for (let day = 0; log.length < total; day++) {
    const dayStart = t0 + day * 86400;
    const todays = [];
    for (let i = 0; i < 20 && nextWord < 3000; i++, nextWord++) {
      const id = `w${nextWord}|r${nextWord}`;
      for (const c of [0, 1]) { const card = { id, card: c, due: day, ivl: 0 }; cards.push(card); todays.push(card); }
    }
    for (const c of cards) if (c.due <= day && c.ivl > 0) todays.push(c);
    let t = dayStart;
    for (const c of todays) {
      if (log.length >= total) break;
      t += 5 + Math.floor(rand() * 20);
      const p = rand();
      const rating = p < 0.12 ? 1 : p < 0.2 ? 2 : p < 0.95 ? 3 : 4;
      const ms = 1500 + Math.floor(rand() * 9000);
      log.push([t, c.id, c.card, rating, ms]);
      if (rating === 1) {
        c.ivl = 1;
        if (log.length < total) { // same-day relearning step ~10 minutes later
          log.push([t + 600, c.id, c.card, 3, 1500 + Math.floor(rand() * 5000)]);
        }
      } else c.ivl = Math.max(1, Math.round((c.ivl || 1) * (rating === 2 ? 1.2 : rating === 3 ? 2.5 : 3.5)));
      c.due = day + c.ivl;
    }
  }
  log.sort((a, b) => a[0] - b[0]);
  return log.slice(0, total);
}

async function loadModelFor(size, src) {
  const cfg = R.RWKV_MODELS[size];
  if (!src) return R.loadModel(size);
  if (/^https?:/.test(src)) return R.loadModel(size, { baseUrl: src.replace(/\/?$/, "/") });
  const buf = fs.readFileSync(path.join(src, cfg.dir, cfg.files.int8.name));
  return R.loadFromBuffer(size, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
}

async function bench(src, sizes, ns, instant) {
  const rows = [];
  for (const size of sizes) {
    const model = await loadModelFor(size, src);
    for (const n of ns) {
      const log = syntheticLog(n);
      const eng = new R.Engine(model);
      eng.reset(5, 540);
      const t0 = performance.now();
      const r = eng.process(log, instant);
      const ms = performance.now() - t0;
      const cards = eng.cards.size;
      rows.push({ size, model: model.id, reviews: n, cards, seconds: +(ms / 1000).toFixed(2), msPerReview: +(ms / n).toFixed(3) });
      console.log(JSON.stringify(rows.at(-1)));
      if (n === ns[ns.length - 1]) {
        // incremental: one more review appended to a warm engine
        const t1 = performance.now();
        const reps = 200;
        for (let i = 0; i < reps; i++) eng.process([[log.at(-1)[0] + 60 * (i + 1), log[i][1], log[i][2], 3, 4000]], false);
        console.log(JSON.stringify({ size, appendOneMs: +((performance.now() - t1) / reps).toFixed(3) }));
        const bytes = [0, 1, 2, 3, 4].map((i) => eng.ent[i].size * model.streams[i].stateSize * 4);
        console.log(JSON.stringify({ size, stateBytesF32: { card: bytes[0], note: bytes[2], perCard: model.streams[0].stateSize * 4, perNote: model.streams[2].stateSize * 4 } }));
        const preds = r.preds.filter((x) => x !== null);
        console.log(JSON.stringify({ size, predsMean: +(preds.reduce((a, b) => a + b, 0) / preds.length).toFixed(4) }));
      }
    }
  }
  return rows;
}

// ---------------------------------------------------------------- main
const [cmd, ...rest] = process.argv.slice(2);
const opt = (name, def) => { const i = rest.indexOf(name); if (i < 0) return def; const v = rest[i + 1]; rest.splice(i, 2); return v; };
const flag = (name) => { const i = rest.indexOf(name); if (i < 0) return false; rest.splice(i, 1); return true; };
if (cmd === "check") {
  const rows = opt("--rows", null);
  await check(rest, rows ? +rows : undefined);
} else if (cmd === "bench") {
  const sizes = opt("--sizes", "tiny,small").split(",");
  const ns = opt("--n", "10000,50000,100000").split(",").map(Number);
  const instant = flag("--instant");
  await bench(rest[0], sizes, ns, instant);
} else if (cmd === "dumplog") {
  fs.writeFileSync(rest[1], JSON.stringify(syntheticLog(+rest[0])));
} else {
  console.log("usage: node tools/rwkv_check.mjs check <package dir>... | bench [weights dir] | dumplog <n> <out.json>");
  process.exit(2);
}
