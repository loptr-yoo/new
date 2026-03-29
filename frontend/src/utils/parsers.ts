import { ParkingLayout, ElementType, LayoutElement } from '../types';

interface UserInputJSON {
  canvas_size: { width: number; height: number };
  parking?: Array<{ id: string; bbox: [number, number, number, number] }>;
  walls?: Array<{ id: string; bbox: [number, number, number, number] }>;
  entrances?: Array<{ id: string; bbox: [number, number, number, number]; type?: string }>;
}

export function parseCustomLayout(json: any): ParkingLayout {
  if (!json.canvas_size || typeof json.canvas_size.width !== 'number') {
    throw new Error('JSON missing "canvas_size"');
  }

  const input = json as UserInputJSON;
  const elements: LayoutElement[] = [];

  const convertBBox = (id: string, type: ElementType, bbox: [number, number, number, number], label?: string) => {
    const width = Math.abs(bbox[2] - bbox[0]);
    const height = Math.abs(bbox[3] - bbox[1]);
    elements.push({
      id,
      type,
      x: bbox[0],
      y: bbox[1],
      width,
      height,
      rotation: 0,
      label
    });
  };

  input.parking?.forEach(p => convertBBox(p.id, ElementType.PARKING_SPACE, p.bbox, p.id));
  input.walls?.forEach(w => convertBBox(w.id, ElementType.WALL, w.bbox, w.id));
  input.entrances?.forEach(e => {
    const type = e.type === 'out' ? ElementType.EXIT : ElementType.ENTRANCE;
    convertBBox(e.id, type, e.bbox, e.type?.toUpperCase());
  });

  return {
    width: input.canvas_size.width,
    height: input.canvas_size.height,
    elements
  };
}