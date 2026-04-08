import { BuildingData, LayoutElement, ParkingLayout } from '../types';

function toNumber(v: any, fallback: number): number {
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function toString(v: any, fallback: string): string {
  return typeof v === 'string' ? v : (v == null ? fallback : String(v));
}

function sanitizeElement(input: any): LayoutElement | null {
  if (!input || typeof input !== 'object') return null;

  const id = toString((input as any).id, '');
  const type = toString((input as any).type ?? (input as any).t, '');
  if (!type) return null;

  const x = toNumber((input as any).x, 0);
  const y = toNumber((input as any).y, 0);
  const width = toNumber((input as any).width ?? (input as any).w, 0);
  const height = toNumber((input as any).height ?? (input as any).h, 0);

  const rotationRaw = (input as any).rotation;
  const rotation = rotationRaw == null ? undefined : toNumber(rotationRaw, 0);

  const label = (input as any).label ?? (input as any).l;
  const subType = (input as any).subType;
  const forward = (input as any).forward;

  const out: LayoutElement = {
    id,
    type,
    x,
    y,
    width,
    height,
  };

  if (rotation != null) out.rotation = rotation;
  if (typeof label === 'string') out.label = label;
  if (typeof subType === 'string') out.subType = subType;
  if (Array.isArray(forward) && forward.length === 3) out.forward = [Number(forward[0]), Number(forward[1]), Number(forward[2])] as any;

  return out;
}

export function sanitizeLayout(input: any): ParkingLayout {
  const width = toNumber(input?.width, 800);
  const height = toNumber(input?.height, 600);
  const sceneId = typeof input?.sceneId === 'string' ? input.sceneId : undefined;

  const rawElements = Array.isArray(input?.elements) ? input.elements : [];
  const elements: LayoutElement[] = [];
  for (const e of rawElements) {
    const se = sanitizeElement(e);
    if (se) elements.push(se);
  }

  return { width, height, elements, ...(sceneId ? { sceneId } : {}) };
}

export function sanitizeBuildingData(input: any): BuildingData {
  const blueprintRaw = Array.isArray(input?.blueprint) ? input.blueprint : [];
  const blueprint: LayoutElement[] = [];
  for (const e of blueprintRaw) {
    const se = sanitizeElement(e);
    if (se) blueprint.push(se);
  }

  const floorsRaw = input?.floors && typeof input.floors === 'object' ? input.floors : {};
  const floors: Record<string, ParkingLayout> = {};
  for (const [k, v] of Object.entries(floorsRaw)) {
    floors[String(k)] = sanitizeLayout(v);
  }

  return { blueprint, floors };
}

