import rawScheme from './buildingColorScheme.json';
import { ElementStyle, ElementType } from '../types';

export type BuildingColorScheme = {
  version: string;
  fingerprint: string;
  semanticColors: { fallback: string };
  parkingElementColors: Record<string, string>;
  floorplanElementColors: Record<string, string>;
};

export const BUILDING_COLOR_SCHEME: BuildingColorScheme = rawScheme as any;

export const COLOR_SCHEME_VERSION = BUILDING_COLOR_SCHEME.version;
export const COLOR_SCHEME_FINGERPRINT = BUILDING_COLOR_SCHEME.fingerprint;

export const FALLBACK_COLOR = BUILDING_COLOR_SCHEME.semanticColors.fallback;

export function createParkingSceneStyles(): Record<string, ElementStyle> {
  const m = BUILDING_COLOR_SCHEME.parkingElementColors;
  const styles: Record<string, ElementStyle> = {};

  for (const [k, color] of Object.entries(m)) {
    if (k === ElementType.LANE_LINE) {
      styles[k] = { fill: 'none', opacity: 1, stroke: color, strokeWidth: 1.5 };
    } else {
      styles[k] = { fill: color, opacity: 1 };
    }
  }

  return styles;
}

export function createFloorPlanSceneStyles(): Record<string, ElementStyle> {
  const m = BUILDING_COLOR_SCHEME.floorplanElementColors;
  const styles: Record<string, ElementStyle> = {};
  for (const [k, color] of Object.entries(m)) {
    styles[k] = { fill: color, opacity: 1 };
  }
  return styles;
}
