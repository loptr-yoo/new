import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(process.cwd());

function getArgValue(flag) {
  const idx = process.argv.indexOf(flag);
  if (idx === -1) return null;
  return process.argv[idx + 1] || null;
}

const colorFilterRaw = getArgValue('--color');
const colorFilter = colorFilterRaw ? colorFilterRaw.toLowerCase() : null;

const defaultRoots = [
  path.join(ROOT, 'frontend', 'src'),
  path.join(ROOT, 'backend'),
  path.join(ROOT, 'scripts'),
];

const includeExt = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs',
  '.py',
  '.json',
  '.css',
  '.html',
  '.md',
]);

function isTextFile(filePath) {
  return includeExt.has(path.extname(filePath).toLowerCase());
}

function walk(dir, out) {
  if (!fs.existsSync(dir)) return;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    if (e.name === 'node_modules' || e.name === '.git' || e.name === '.venv') continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.isFile() && isTextFile(p)) out.push(p);
  }
}

function padRight(s, n) {
  if (s.length >= n) return s;
  return s + ' '.repeat(n - s.length);
}

function main() {
  const files = [];
  for (const r of defaultRoots) walk(r, files);

  const hexRe = /#[0-9a-fA-F]{6}\b/g;
  const counts = new Map();
  const occ = new Map();

  for (const file of files) {
    let text;
    try {
      text = fs.readFileSync(file, 'utf8');
    } catch {
      continue;
    }
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(hexRe);
      if (!m) continue;
      for (const c of m) {
        const hex = c.toLowerCase();
        if (colorFilter && hex !== colorFilter) continue;
        counts.set(hex, (counts.get(hex) || 0) + 1);
        const arr = occ.get(hex) || [];
        arr.push({ file, line: i + 1, text: line.trim() });
        occ.set(hex, arr);
      }
    }
  }

  if (colorFilter) {
    const list = occ.get(colorFilter) || [];
    if (list.length === 0) {
      process.stdout.write(`No matches for ${colorFilter}\n`);
      return;
    }
    for (const it of list) {
      const rel = path.relative(ROOT, it.file);
      process.stdout.write(`${rel}:${it.line}: ${it.text}\n`);
    }
    process.stdout.write(`\nTotal: ${list.length}\n`);
    return;
  }

  const rows = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  if (rows.length === 0) {
    process.stdout.write('No hex colors found.\n');
    return;
  }

  const maxHex = Math.max(...rows.map(r => r[0].length), 7);
  const maxCount = Math.max(...rows.map(r => String(r[1]).length), 5);
  process.stdout.write(`${padRight('COLOR', maxHex)}  ${padRight('COUNT', maxCount)}  TOP LOCATIONS\n`);
  process.stdout.write(`${'-'.repeat(maxHex)}  ${'-'.repeat(maxCount)}  -------------\n`);

  for (const [hex, cnt] of rows) {
    const list = occ.get(hex) || [];
    const topFiles = [];
    const perFile = new Map();
    for (const it of list) {
      const rel = path.relative(ROOT, it.file);
      perFile.set(rel, (perFile.get(rel) || 0) + 1);
    }
    for (const [f, c] of Array.from(perFile.entries()).sort((a, b) => b[1] - a[1]).slice(0, 3)) {
      topFiles.push(`${f} (${c})`);
    }
    process.stdout.write(`${padRight(hex, maxHex)}  ${padRight(String(cnt), maxCount)}  ${topFiles.join(', ')}\n`);
  }

  process.stdout.write('\nUse: npm run colors:refs -- --color #RRGGBB\n');
}

main();

