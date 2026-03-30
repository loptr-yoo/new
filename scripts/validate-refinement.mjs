import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const root = process.cwd();
const coarsePath = args[0] || path.join(root, 'tests', 'fixtures', 'coarse_layout.json');
const patchPath = args[1] || path.join(root, 'tests', 'fixtures', 'refinement_patch.json');

const coarse = JSON.parse(fs.readFileSync(coarsePath, 'utf8'));
const patch = JSON.parse(fs.readFileSync(patchPath, 'utf8'));

const structural = new Set(['wall', 'driving_lane', 'ground', 'slope', 'entrance', 'exit']);

const byId = new Map(coarse.elements.map(e => [e.id, { ...e }]));
const deletedIds = Array.isArray(patch.deleted_ids) ? patch.deleted_ids : [];
deletedIds.forEach(id => {
  const existing = byId.get(id);
  if (existing && structural.has(existing.type)) {
    throw new Error(`结构元素被删除: ${id}`);
  }
  byId.delete(id);
});

const modified = Array.isArray(patch.modified_elements) ? patch.modified_elements : [];
modified.forEach(p => {
  const id = String(p.id || '');
  const existing = byId.get(id);
  if (!existing) return;
  if (structural.has(existing.type)) {
    throw new Error(`结构元素被修改: ${id}`);
  }
  Object.assign(existing, p);
});

const newElements = Array.isArray(patch.new_elements) ? patch.new_elements : [];
let newIndex = 0;
newElements.forEach(ne => {
  const id = ne.id || `new_${newIndex++}`;
  byId.set(id, { id, ...ne });
});

const refined = {
  width: coarse.width,
  height: coarse.height,
  elements: Array.from(byId.values())
};

const width = refined.width;
const height = refined.height;
const covered = new Uint8Array(width * height);

const markRect = (x, y, w, h) => {
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(width, Math.ceil(x + w));
  const y1 = Math.min(height, Math.ceil(y + h));
  for (let yy = y0; yy < y1; yy++) {
    const row = yy * width;
    for (let xx = x0; xx < x1; xx++) {
      covered[row + xx] = 1;
    }
  }
};

refined.elements.forEach(el => {
  if (['wall', 'driving_lane', 'ground'].includes(el.type)) {
    markRect(el.x, el.y, el.width, el.height);
  }
});

let uncovered = 0;
for (let i = 0; i < covered.length; i++) {
  if (!covered[i]) uncovered++;
}
const holeRatio = uncovered / covered.length;
if (holeRatio > 0.001) {
  throw new Error(`空洞面积超标: ${(holeRatio * 100).toFixed(3)}%`);
}

const structuralChanges = [];
coarse.elements.forEach(e => {
  if (!structural.has(e.type)) return;
  const after = refined.elements.find(r => r.id === e.id);
  if (!after) structuralChanges.push(e.id);
  else if (e.x !== after.x || e.y !== after.y || e.width !== after.width || e.height !== after.height) {
    structuralChanges.push(e.id);
  }
});

if (structuralChanges.length > 0) {
  throw new Error(`结构元素被改变: ${structuralChanges.join(', ')}`);
}

console.log('OK: skeleton unchanged, void ratio <= 0.1%');
