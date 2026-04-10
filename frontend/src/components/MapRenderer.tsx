import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as d3 from 'd3';
import { ElementType } from '../types';
import { useStore } from '../store';
import { DEFAULT_SCENE_ID, SCENE_REGISTRY } from '../utils/sceneRegistry';
import { FALLBACK_COLOR } from '../theme/buildingColorScheme';

const ELEMENT_STYLES: Record<string, { fill: string; opacity: number; stroke?: string; strokeWidth?: number }> = {
  [ElementType.GROUND]: { fill: '#1e293b', opacity: 1 },   // 统一为 floor_slab 色
  [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },
  [ElementType.PARKING_SPACE]: { fill: '#3b82f6', opacity: 1 },
  [ElementType.SIDEWALK]: { fill: '#1e293b', opacity: 1 },
  [ElementType.RAMP]: { fill: '#c026d3', opacity: 1 },
  [ElementType.PILLAR]: { fill: '#94a3b8', opacity: 1 },
  [ElementType.WALL]: { fill: '#475569', opacity: 1 },   // 统一为 exterior_wall 色
  [ElementType.ENTRANCE]: { fill: '#15803d', opacity: 1 },
  [ElementType.EXIT]: { fill: '#b91c1c', opacity: 1 },
  [ElementType.STAIRCASE]: { fill: '#b45309', opacity: 1 },
  [ElementType.ELEVATOR]: { fill: '#06b6d4', opacity: 1 },
  [ElementType.CHARGING_STATION]: { fill: '#22c55e', opacity: 1 },
  [ElementType.GUIDANCE_SIGN]: { fill: '#f59e0b', opacity: 1 },
  [ElementType.SAFE_EXIT]: { fill: '#10b981', opacity: 1 },
  [ElementType.SPEED_BUMP]: { fill: '#eab308', opacity: 1 },
  [ElementType.FIRE_EXTINGUISHER]: { fill: '#ef4444', opacity: 1 },
  [ElementType.LANE_LINE]: { fill: 'none', opacity: 1, stroke: '#facc15', strokeWidth: 1.5 },
  [ElementType.CONVEX_MIRROR]: { fill: '#38bdf8', opacity: 1 },
  // V2 Building 元素
  'floor_slab': { fill: '#1e293b', opacity: 1 },
  'corridor': { fill: '#cbd5e1', opacity: 0.7 },
  'exterior_wall': { fill: '#475569', opacity: 1 },
  'partition_wall': { fill: '#64748b', opacity: 1 },
  'door': { fill: '#e2e8f0', opacity: 1 },
  'window': { fill: '#7dd3fc', opacity: 0.8 },
};

const BACKGROUND_COLOR = '#020617';
const warnedMissingStyleTypes = new Set<string>();

function resolveBackgroundColor(scene: any, styles: any): string {
  const cfg = scene?.background;
  if (!cfg) return BACKGROUND_COLOR;
  if (cfg.mode === 'fixed') return cfg.color || BACKGROUND_COLOR;
  if (cfg.mode === 'styleKey') {
    const key = cfg.styleKey;
    return (key && styles?.[key]?.fill) || cfg.fallbackColor || BACKGROUND_COLOR;
  }
  return BACKGROUND_COLOR;
}

function isOrthogonalPolygon(polygon: [number, number][]): boolean {
  const eps = 1e-6;
  for (let i = 0; i < polygon.length - 1; i++) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[i + 1];
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    if (dx > eps && dy > eps) return false;
  }
  return true;
}

export interface MapRendererHandle {
  downloadJpg: () => void;
  downloadJson: () => void;
}

const MapRenderer = forwardRef<MapRendererHandle>((props, ref) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { layout, violations, activeSceneId } = useStore();

  useImperativeHandle(ref, () => ({
    downloadJpg: () => {
      if (!svgRef.current || !layout) return;
      const svgNode = svgRef.current;
      const zoomGroup = d3.select(svgNode).select("g.main-group");
      const prevTransform = zoomGroup.attr("transform");
      zoomGroup.attr("transform", null); 
      svgNode.setAttribute("width", layout.width.toString());
      svgNode.setAttribute("height", layout.height.toString());
      svgNode.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);

      const serializer = new XMLSerializer();
      let svgString = serializer.serializeToString(svgNode);
      if (!svgString.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {
        svgString = svgString.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }
      
      if (prevTransform) zoomGroup.attr("transform", prevTransform);

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = layout.width;
        canvas.height = layout.height;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.imageSmoothingEnabled = false; 
          const exportSceneKey = layout.sceneId || activeSceneId;
          const exportScene = SCENE_REGISTRY[exportSceneKey] || SCENE_REGISTRY[DEFAULT_SCENE_ID];
          const exportStyles = (exportScene?.styles && Object.keys(exportScene.styles).length > 0) ? exportScene.styles : ELEMENT_STYLES;
          ctx.fillStyle = resolveBackgroundColor(exportScene as any, exportStyles as any);
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          const jpgUrl = canvas.toDataURL("image/jpeg", 0.98);
          const link = document.createElement("a");
          link.download = `parking_map_${Date.now()}.jpg`;
          link.href = jpgUrl;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      };
      img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgString)));
    },
    downloadJson: () => {
      if (!layout) return;
      const dataStr = JSON.stringify(layout, null, 2);
      const blob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `parking_layout_${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  }));

  useEffect(() => {
    if (!layout || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); 

    const { width, height, elements } = layout;
    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .attr("width", "100%")
       .attr("height", "100%")
       .style("shape-rendering", "crispEdges"); 
    
    const mainGroup = svg.append("g").attr("class", "main-group");

    const sceneKey = layout.sceneId || activeSceneId;
    const activeScene = SCENE_REGISTRY[sceneKey] || SCENE_REGISTRY[DEFAULT_SCENE_ID];
    const baseStyles = (activeScene?.styles && Object.keys(activeScene.styles).length > 0) ? activeScene.styles : ELEMENT_STYLES;
    const sceneStyles = baseStyles as any;
    
    const defaultZOrder = [
        'floor_slab',
        ElementType.GROUND,
        'corridor',
        ElementType.ROAD,
        ElementType.SIDEWALK,
        ElementType.RAMP,
        ElementType.PARKING_SPACE,
        'exterior_wall',
        'partition_wall',
        ElementType.WALL,
        ElementType.PILLAR,
        ElementType.STAIRCASE,
        ElementType.ELEVATOR,
        'door',
        'window',
        ElementType.LANE_LINE,
        ElementType.SPEED_BUMP,
        ElementType.SAFE_EXIT,
        ElementType.FIRE_EXTINGUISHER,
        ElementType.GUIDANCE_SIGN,
        ElementType.ENTRANCE,
        ElementType.EXIT,
        ElementType.CHARGING_STATION,
        ElementType.CONVEX_MIRROR,
    ];
    const zOrder = (activeScene?.zOrder && activeScene.zOrder.length > 0) ? activeScene.zOrder : defaultZOrder;
    
    const sortedElements = [...elements].sort((a, b) => {
      const idxA = zOrder.indexOf(a.type as ElementType);
      const idxB = zOrder.indexOf(b.type as ElementType);
      // 未知类型（如 living_room, bedroom）放在 corridor 和 wall 之间
      const defaultIdx = zOrder.indexOf('exterior_wall') - 0.5;
      return (idxA === -1 ? defaultIdx : idxA) - (idxB === -1 ? defaultIdx : idxB);
    });

    const bgColor = resolveBackgroundColor(activeScene as any, sceneStyles);

    mainGroup.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", width).attr("height", height)
      .attr("fill", bgColor)
      .attr("opacity", 1);

    mainGroup.selectAll("g.element")
      .data(sortedElements)
      .enter()
      .append("g")
      .attr("class", "element")
      .attr("transform", d => `translate(${d.x}, ${d.y})`)
      .each(function(this: any, d) {
        const g = d3.select(this);
        const rawType = d.type as string;
        const normalizedType = (activeScene?.elementNormalization?.[rawType] || rawType) as string;
        const style = sceneStyles[normalizedType] || sceneStyles[rawType] || { fill: FALLBACK_COLOR, opacity: 1 };
        if (import.meta.env?.DEV && !(sceneStyles[normalizedType] || sceneStyles[rawType])) {
          const key = `${sceneKey}:${normalizedType}`;
          if (!warnedMissingStyleTypes.has(key)) {
            warnedMissingStyleTypes.add(key);
            console.warn(`Missing style mapping for type "${normalizedType}" in scene "${sceneKey}". Using FALLBACK_COLOR.`);
          }
        }
        const customDrawer = activeScene?.customDrawers?.[normalizedType] || activeScene?.customDrawers?.[rawType];
        const w = d.width;
        const h = d.height;
        const cx = w / 2;
        const cy = h / 2;

        if (customDrawer) {
          customDrawer(g, d as any, style as any, { layout, violations } as any);
          return;
        }

        if (normalizedType === ElementType.LANE_LINE) {
             const isVertical = h > w;
             g.append("line")
               .attr("x1", isVertical ? cx : 0).attr("y1", isVertical ? 0 : cy)
               .attr("x2", isVertical ? cx : w).attr("y2", isVertical ? h : cy)
               .attr("stroke", style.stroke || FALLBACK_COLOR).attr("stroke-width", style.strokeWidth || 1.5).attr("stroke-dasharray", "8,8")
               .attr("stroke-opacity", style.opacity ?? 1)
               .style("shape-rendering", "geometricPrecision"); 
        } 
        else if (normalizedType === ElementType.SIDEWALK) {
            g.append("rect")
              .attr("width", w).attr("height", h)
              .attr("fill", style.fill)
              .attr("opacity", style.opacity ?? 1);
        }
        else if (normalizedType === ElementType.SPEED_BUMP) {
             const cx = d.x + w / 2;
             const cy = d.y + h / 2;
             const parentRoad = layout?.elements.find(r => 
                 r.type === ElementType.ROAD && 
                 cx >= r.x && cx <= r.x + r.width &&
                 cy >= r.y && cy <= r.y + r.height
             );
 
             let renderW = w;
             let renderH = h;
             let offsetX = 0;
             let offsetY = 0;
 
             if (parentRoad) {
                 const isRoadHorizontal = parentRoad.width > parentRoad.height;
                 const isBumpHorizontal = w > h;
                 if (isRoadHorizontal === isBumpHorizontal) {
                     renderW = h;
                     renderH = w;
                     offsetX = (w - renderW) / 2;
                     offsetY = (h - renderH) / 2;
                 }
             }
 
             g.append("rect")
               .attr("x", offsetX)
               .attr("y", offsetY)
               .attr("width", renderW)
               .attr("height", renderH)
               .attr("fill", style.fill)
               .attr("opacity", style.opacity ?? 1)
               .attr("rx", 2);
        }
        else if (normalizedType === ElementType.GUIDANCE_SIGN) {
            g.append("rect").attr("width", w).attr("height", h).attr("fill", style.fill).attr("opacity", style.opacity ?? 1).attr("rx", 2);
            const s = Math.min(w, h) * 0.7;
            const rot = d.rotation || 0;
            g.append("path")
             .attr("d", `M ${cx - s/4} ${cy - s/2} L ${cx + s/2} ${cy} L ${cx - s/4} ${cy + s/2} M ${cx + s/2} ${cy} L ${cx - s/2} ${cy}`)
             .attr("stroke", "#ffffff").attr("fill", "none").attr("stroke-width", 2)
             .attr("stroke-linecap", "round").attr("stroke-linejoin", "round")
             .attr("transform", `rotate(${rot}, ${cx}, ${cy})`)
             .style("shape-rendering", "geometricPrecision");
        } 
        else if (d.polygon && d.polygon.length >= 3) {
            const structuralTypes = new Set<string>([
              'floor_slab',
              'corridor',
              'wall',
              'exterior_wall',
              'partition_wall',
              'door',
              'window',
              ElementType.PILLAR,
              ElementType.ELEVATOR,
              ElementType.STAIRCASE,
              ElementType.WALL,
            ]);
            const isStructural =
              structuralTypes.has(normalizedType) ||
              structuralTypes.has(rawType) ||
              rawType === 'exterior_wall' ||
              rawType === 'partition_wall';
            const isWall =
              rawType === 'exterior_wall' ||
              rawType === 'partition_wall' ||
              normalizedType === 'wall' ||
              normalizedType === ElementType.WALL;

            const points = d.polygon.map(([px, py]: [number, number]) => `${px - d.x},${py - d.y}`).join(' ');
            const poly = g.append("polygon")
              .attr("points", points)
              .attr("fill", style.fill)
              .attr("opacity", style.opacity ?? 1);

            if (!isStructural) {
              poly
                .attr("stroke", "#334155")
                .attr("stroke-width", 0.2)
                .attr("stroke-opacity", 0.35);
            } else {
              poly.attr("stroke", "none");
            }

            const orthogonal = isOrthogonalPolygon(d.polygon);
            poly.style("shape-rendering", orthogonal ? "crispEdges" : "geometricPrecision");

            if (isWall) {
              poly.attr("fill", "#334155").attr("opacity", 1).style("shape-rendering", "crispEdges");
            }
        }
        else {
            const rect = g.append("rect")
              .attr("width", w).attr("height", h)
              .attr("fill", style.fill)
              .attr("opacity", style.opacity ?? 1)
              .attr("transform", d.rotation ? `rotate(${d.rotation}, ${cx}, ${cy})` : null);
            if (d.type === ElementType.PILLAR) rect.attr("rx", 4);
        }

        if (violations.some(v => v.elementId === d.id)) {
            g.append("rect").attr("width", w).attr("height", h).attr("fill", "none").attr("stroke", "#ef4444").attr("stroke-width", 2).style("stroke-dasharray", "4,2");
        }
      });

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 15])
      .on("zoom", (event) => mainGroup.attr("transform", event.transform));
    svg.call(zoom);

    if (svgRef.current?.parentElement) {
        const { clientWidth: pw, clientHeight: ph } = svgRef.current.parentElement;
        const scale = Math.min(pw / width, ph / height) * 0.9;
        svg.call(zoom.transform, d3.zoomIdentity.translate((pw - width * scale) / 2, (ph - height * scale) / 2).scale(scale));
    }
  }, [layout, violations, activeSceneId]);

  return (
    <div className="w-full h-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800 shadow-inner relative">
       <svg ref={svgRef} className="block cursor-grab active:cursor-grabbing w-full h-full" />
    </div>
  );
});

export default MapRenderer;
