import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const inputPath = args[0];
const outputPath = args[1];
if (!inputPath || !outputPath) {
  throw new Error('Usage: node scripts/render-layout.mjs <input.json> <output.svg>');
}

const layout = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const width = layout.width || 800;
const height = layout.height || 600;

const colors = {
  ground: '#334155',
  driving_lane: '#1e293b',
  wall: '#f1f5f9',
  slope: '#c026d3',
  entrance: '#15803d',
  exit: '#b91c1c',
  parking_space: '#3b82f6',
  ground_line: '#facc15',
  guidance_sign: '#f59e0b',
  pillar: '#94a3b8'
};

const shapes = layout.elements.map(el => {
  const fill = colors[el.type] || '#ff00ff';
  const poly = el.polygon;
  if (Array.isArray(poly) && poly.length >= 3) {
    const pts = poly.map(p => `${p[0]},${p[1]}`).join(' ');
    return `<polygon points="${pts}" fill="${fill}" stroke="none" />`;
  }

  const w = el.width || 0;
  const h = el.height || 0;
  let x = el.x || 0;
  let y = el.y || 0;
  const anchor = el.anchor || 'min';
  if (anchor === 'center') {
    x = x - w / 2;
    y = y - h / 2;
  }

  if (el.type === 'ground_line') {
    return `<line x1="${x}" y1="${y}" x2="${x + w}" y2="${y + h}" stroke="${fill}" stroke-width="2" stroke-dasharray="8,8" />`;
  }

  const rot = el.rotation || 0;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const transform = rot ? ` transform="rotate(${rot}, ${cx}, ${cy})"` : '';
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}"${transform} />`;
});

const svg = [
  `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  `<rect x="0" y="0" width="${width}" height="${height}" fill="#0f172a" />`,
  shapes.join('\n'),
  `</svg>`
].join('\n');

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, svg, 'utf8');
console.log(`SVG written: ${outputPath}`);
