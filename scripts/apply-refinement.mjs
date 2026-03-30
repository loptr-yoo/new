import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const root = process.cwd();
const coarsePath = args[0] || path.join(root, 'tests', 'fixtures', 'coarse_layout.json');
const patchPath = args[1] || path.join(root, 'tests', 'fixtures', 'refinement_patch.json');
const outputPath = args[2] || path.join(root, 'artifacts', 'refined_layout.json');

const coarse = JSON.parse(fs.readFileSync(coarsePath, 'utf8'));
const patch = JSON.parse(fs.readFileSync(patchPath, 'utf8'));

const byId = new Map(coarse.elements.map(e => [e.id, { ...e }]));
const deletedIds = Array.isArray(patch.deleted_ids) ? patch.deleted_ids : [];
deletedIds.forEach(id => byId.delete(id));

const modified = Array.isArray(patch.modified_elements) ? patch.modified_elements : [];
modified.forEach(p => {
  const id = String(p.id || '');
  const existing = byId.get(id);
  if (!existing) return;
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

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(refined, null, 2), 'utf8');
console.log(`Refined JSON written: ${outputPath}`);
