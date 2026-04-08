import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as d3 from 'd3';
import { ElementType } from '../types';
import { useStore } from '../store';
import { DEFAULT_SCENE_ID, SCENE_REGISTRY } from '../utils/sceneRegistry';
import { FALLBACK_COLOR } from '../theme/buildingColorScheme';

const ELEMENT_STYLES: Record<string, { fill: string; opacity: number; stroke?: string; strokeWidth?: number }> = {
  [ElementType.GROUND]: { fill: '#334155', opacity: 1 }, 
  [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },   
  [ElementType.PARKING_SPACE]: { fill: '#3b82f6', opacity: 1 },
  [ElementType.SIDEWALK]: { fill: '#1e293b', opacity: 1 }, 
  [ElementType.RAMP]: { fill: '#c026d3', opacity: 1 },
  [ElementType.PILLAR]: { fill: '#94a3b8', opacity: 1 },
  [ElementType.WALL]: { fill: '#f1f5f9', opacity: 1 },
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
  [ElementType.CONVEX_MIRROR]: { fill: '#38bdf8', opacity: 1 }
};

const BACKGROUND_COLOR = '#020617';
const warnedMissingStyleTypes = new Set<string>();

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
          ctx.fillStyle = BACKGROUND_COLOR; 
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
        ElementType.WALL, 
        ElementType.GROUND, 
        ElementType.ROAD, 
        ElementType.RAMP,
        ElementType.SIDEWALK, 
        ElementType.PARKING_SPACE, 
        ElementType.LANE_LINE, 
        ElementType.SPEED_BUMP, 
        ElementType.PILLAR, 
        ElementType.STAIRCASE,
        ElementType.ELEVATOR,
        ElementType.SAFE_EXIT,
        ElementType.FIRE_EXTINGUISHER,
        ElementType.GUIDANCE_SIGN
    ];
    const zOrder = (activeScene?.zOrder && activeScene.zOrder.length > 0) ? activeScene.zOrder : defaultZOrder;
    
    const sortedElements = [...elements].sort((a, b) => {
      const idxA = zOrder.indexOf(a.type as ElementType);
      const idxB = zOrder.indexOf(b.type as ElementType);
      return (idxA === -1 ? 999 : idxA) - (idxB === -1 ? 999 : idxB);
    });

    mainGroup.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", width).attr("height", height)
      .attr("fill", BACKGROUND_COLOR)
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
