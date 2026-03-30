import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const root = process.cwd();
const beforePath = args[0] || path.join(root, 'tests', 'fixtures', 'coarse_layout.json');
const afterPath = args[1] || path.join(root, 'artifacts', 'refined_layout.json');

const before = JSON.parse(fs.readFileSync(beforePath, 'utf8'));
const after = JSON.parse(fs.readFileSync(afterPath, 'utf8'));

const byId = new Map(after.elements.map(e => [e.id, e]));
const structural = new Set(['wall', 'driving_lane', 'ground', 'slope', 'entrance', 'exit']);

let changed = 0;
let structuralChanged = 0;
let missing = 0;
before.elements.forEach(e => {
  const a = byId.get(e.id);
  if (!a) {
    missing++;
    if (structural.has(e.type)) structuralChanged++;
    return;
  }
  if (e.x !== a.x || e.y !== a.y || e.width !== a.width || e.height !== a.height) {
    changed++;
    if (structural.has(e.type)) structuralChanged++;
  }
});

const added = after.elements.filter(e => !before.elements.some(b => b.id === e.id)).length;
console.log(JSON.stringify({ changed, structuralChanged, missing, added }, null, 2));
