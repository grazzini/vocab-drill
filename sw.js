/* Offline support. The app itself is fetched network-first so a new build shows up on the
   next open; audio, stroke drawings and fonts never change at their URLs, so they are kept
   once seen. Everything else (sync, translation, the explainer, local TTS) goes straight
   to the network. */
const SHELL = "shell-v2", KEEP = "keep-v1";
const SHELL_FILES = ["./", "sentences.js", "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png"];
const KEEP_ORIGINS = ["https://raw.githubusercontent.com", "https://fonts.googleapis.com", "https://fonts.gstatic.com"];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(SHELL).then(cache => cache.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys()
    .then(names => Promise.all(names.filter(n => n !== SHELL && n !== KEEP).map(n => caches.delete(n))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", event => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    if (url.pathname.includes("/audio/")) event.respondWith(audio(req));
    else event.respondWith(networkFirst(req));
  } else if (KEEP_ORIGINS.includes(url.origin)) event.respondWith(cacheFirst(req));
});

async function networkFirst(req) {
  const cache = await caches.open(SHELL);
  const key = req.mode === "navigate" ? "./" : req;
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(key, res.clone());
    return res;
  } catch (err) {
    return (await cache.match(key, { ignoreSearch: true })) || Promise.reject(err);
  }
}

async function cacheFirst(req) {
  const cache = await caches.open(KEEP);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === "opaque") cache.put(req, res.clone());
  return res;
}

/* Media elements ask for byte ranges (Safari insists), so keep the whole file and slice it. */
async function audio(req) {
  const cache = await caches.open(KEEP);
  let full = await cache.match(req.url);
  if (!full) {
    const res = await fetch(req.url);
    if (!res.ok) return res;
    await cache.put(req.url, res.clone());
    full = res;
  }
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.get("range") || "");
  if (!range) return full;
  const blob = await full.blob();
  const start = range[1] ? Number(range[1]) : Math.max(0, blob.size - Number(range[2]));
  const end = range[1] && range[2] ? Math.min(Number(range[2]), blob.size - 1) : blob.size - 1;
  return new Response(blob.slice(start, end + 1), {
    status: 206,
    headers: {
      "content-type": full.headers.get("content-type") || "audio/mp4",
      "content-range": `bytes ${start}-${end}/${blob.size}`,
      "content-length": String(end - start + 1),
      "accept-ranges": "bytes",
    },
  });
}
