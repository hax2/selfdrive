#!/usr/bin/env node
/**
 * server.js
 * High-performance, zero-dependency Node.js HTTP server for TTFM Visualizer.
 * Serves the modern UI, REST API endpoints, and real benchmark image artifacts.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = process.env.PORT || 3000;
const ROOT_DIR = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(ROOT_DIR, 'docs');
const PUBLIC_DIR = fs.existsSync(DOCS_DIR) ? DOCS_DIR : path.join(__dirname, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.pdf': 'application/pdf',
  '.ico': 'image/x-icon',
  '.webp': 'image/webp'
};

// Safe media directory aliases
const MEDIA_MOUNTS = {
  '/media/look_here': path.join(ROOT_DIR, 'look here'),
  '/media/reports': path.join(ROOT_DIR, 'reports'),
  '/media/outputs': path.join(ROOT_DIR, 'outputs'),
  '/media/data_processed_blue_green': path.join(ROOT_DIR, 'data_processed_blue_green'),
  '/media/realtime_blue_green_results': path.join(ROOT_DIR, 'realtime_blue_green_results'),
  '/media/rod_suite_results': path.join(ROOT_DIR, 'rod_suite_results'),
};

function sendFile(res, filePath, contentType) {
  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`404 Not Found: ${path.basename(filePath)}`);
      return;
    }

    const stream = fs.createReadStream(filePath);
    res.writeHead(200, {
      'Content-Type': contentType,
      'Content-Length': stats.size,
      'Cache-Control': 'public, max-age=3600'
    });
    stream.pipe(res);
  });
}

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url);
  let pathname = decodeURIComponent(parsedUrl.pathname);

  // Security: Prevent directory traversal
  if (pathname.includes('..')) {
    res.writeHead(403, { 'Content-Type': 'text/plain' });
    res.end('403 Forbidden');
    return;
  }

  // API Route: Benchmark Data
  if (pathname === '/api/data') {
    const dataPath = path.join(PUBLIC_DIR, 'data', 'benchmark_data.json');
    sendFile(res, dataPath, 'application/json; charset=utf-8');
    return;
  }

  // API Route: Health check
  if (pathname === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', time: new Date().toISOString() }));
    return;
  }

  // Media mounts
  for (const [mountPrefix, targetDir] of Object.entries(MEDIA_MOUNTS)) {
    if (pathname.startsWith(mountPrefix)) {
      const relPath = pathname.slice(mountPrefix.length).replace(/^\/+/, '');
      const filePath = path.join(targetDir, relPath);
      const ext = path.extname(filePath).toLowerCase();
      const mime = MIME_TYPES[ext] || 'application/octet-stream';
      sendFile(res, filePath, mime);
      return;
    }
  }

  // Frontend Static Files
  let localPath = pathname === '/' ? path.join(PUBLIC_DIR, 'index.html') : path.join(PUBLIC_DIR, pathname);
  const ext = path.extname(localPath).toLowerCase();

  // If requesting directory or no extension, fallback to index.html for SPA feel
  if (!ext) {
    localPath = path.join(PUBLIC_DIR, 'index.html');
  }

  const contentType = MIME_TYPES[path.extname(localPath).toLowerCase()] || 'text/plain';
  sendFile(res, localPath, contentType);
});

server.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(` TTFM Off-Road Traversability Web Visualizer Running `);
  console.log(` Local URL: http://localhost:${PORT}`);
  console.log(` Health Check: http://localhost:${PORT}/api/health`);
  console.log(` Benchmark Data: http://localhost:${PORT}/api/data`);
  console.log(`=======================================================`);
});
