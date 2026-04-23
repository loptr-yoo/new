import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

/* global process */

const ROOT = path.resolve(process.cwd());
const SCHEME_JSON = path.join(ROOT, 'src', 'theme', 'buildingColorScheme.json');
const VARS_CSS = path.join(ROOT, 'src', 'theme', 'building-vars.css');

const args = new Set(process.argv.slice(2));
const fix = args.has('--fix');

function stableStringify(v) {
  if (v == null) return 'null';
  if (typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(',')}]`;
  const keys = Object.keys(v).sort();
  return `{${keys.map(k => `${JSON.stringify(k)}:${stableStringify(v[k])}`).join(',')}}`;
}

function sha256Hex(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

function isHex6(v) {
  return typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v);
}

function fail(msg) {
  process.stderr.write(`${msg}\n`);
  process.exitCode = 1;
}

function loadScheme() {
  const txt = fs.readFileSync(SCHEME_JSON, 'utf8');
  return JSON.parse(txt);
}

function writeScheme(next) {
  fs.writeFileSync(SCHEME_JSON, JSON.stringify(next, null, 2) + '\n', 'utf8');
}

function writeVarsCss(scheme) {
  const lines = [':root {'];
  lines.push(`  --bldg-fallback: ${scheme.semanticColors.fallback};`);
  // Export all unique colors as CSS vars
  const allColors = new Set();
  for (const c of Object.values(scheme.parkingElementColors)) allColors.add(c);
  for (const c of Object.values(scheme.floorplanElementColors)) allColors.add(c);
  lines.push('}');
  fs.writeFileSync(VARS_CSS, lines.join('\n') + '\n', 'utf8');
}

// ============ Load & validate ============

const scheme = loadScheme();

const requiredTop = ['version', 'fingerprint', 'semanticColors', 'parkingElementColors', 'floorplanElementColors'];
for (const k of requiredTop) {
  if (!(k in scheme)) fail(`Missing required field: ${k}`);
}

if (!scheme.semanticColors?.fallback) fail('semanticColors must include "fallback".');
if (!isHex6(scheme.semanticColors.fallback)) fail(`Invalid fallback hex: ${scheme.semanticColors.fallback}`);

// Validate per-element colors
function validateElementColors(sceneName, mapping) {
  for (const [k, hex] of Object.entries(mapping)) {
    if (!isHex6(hex)) fail(`[${sceneName}] Invalid hex for "${k}": ${hex}`);
  }
}

validateElementColors('parking', scheme.parkingElementColors);
validateElementColors('floorplan', scheme.floorplanElementColors);

// Cross-scene consistency: shared element types must have same color
const parkingKeys = new Set(Object.keys(scheme.parkingElementColors));
const floorKeys = new Set(Object.keys(scheme.floorplanElementColors));
const sharedKeys = [...parkingKeys].filter(k => floorKeys.has(k));

for (const k of sharedKeys) {
  const pColor = scheme.parkingElementColors[k];
  const fColor = scheme.floorplanElementColors[k];
  if (pColor.toLowerCase() !== fColor.toLowerCase()) {
    fail(`Cross-scene mismatch for "${k}": parking=${pColor}, floorplan=${fColor}`);
  }
}

if (sharedKeys.length > 0) {
  process.stderr.write(`[INFO] Cross-scene shared elements (${sharedKeys.length}): ${sharedKeys.join(', ')}\n`);
}

// ============ Fingerprint ============

const fingerprintBase = {
  version: scheme.version,
  semanticColors: scheme.semanticColors,
  parkingElementColors: scheme.parkingElementColors,
  floorplanElementColors: scheme.floorplanElementColors,
};
const computed = sha256Hex(stableStringify(fingerprintBase));

if (scheme.fingerprint !== computed) {
  if (fix) {
    const next = { ...scheme, fingerprint: computed };
    writeScheme(next);
    process.stderr.write(`[FIX] Fingerprint updated: ${computed}\n`);
  } else {
    fail(`Fingerprint mismatch. Expected: ${computed} Current: ${scheme.fingerprint}. Run: npm run colors:fix`);
  }
}

writeVarsCss(scheme);

if (process.exitCode) process.exit(process.exitCode);
