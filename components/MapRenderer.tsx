import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as d3 from 'd3';
import { ElementType } from '../types';
import { useStore } from '../store';

const ELEMENT_STYLES: Record<string, { fill: string; opacity: number }> = {
  [ElementType.GROUND]: { fill: '#475569', opacity: 1 },
  [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },
  [ElementType.PARKING_SPACE]: { fill: '#3b82f6', opacity: 1 },
  [ElementType.SIDEWALK]: { fill: '#cbd5e1', opacity: 1 }, 
  [ElementType.RAMP]: { fill: '#c026d3', opacity: 1 },
  [ElementType.PILLAR]: { fill: '#94a3b8', opacity: 1 },
  [ElementType.WALL]: { fill: '#f1f5f9', opacity: 1 },
  [ElementType.ENTRANCE]: { fill: '#15803d', opacity: 1 },
  [ElementType.EXIT]: { fill: '#b91c1c', opacity: 1 },
  [ElementType.STAIRCASE]: { fill: '#7e22ce', opacity: 1 },
  [ElementType.ELEVATOR]: { fill: '#0284c7', opacity: 1 },
  [ElementType.CHARGING_STATION]: { fill: '#84cc16', opacity: 1 },
  [ElementType.GUIDANCE_SIGN]: { fill: '#d97706', opacity: 1 },
  [ElementType.SAFE_EXIT]: { fill: '#0d9488', opacity: 1 },
  [ElementType.SPEED_BUMP]: { fill: '#fbbf24', opacity: 1 },
  [ElementType.FIRE_EXTINGUISHER]: { fill: '#ef4444', opacity: 1 },
  [ElementType.LANE_LINE]: { fill: 'none', opacity: 1 },
  [ElementType.CONVEX_MIRROR]: { fill: '#f97316', opacity: 1 },
};

export interface MapRendererHandle {
  downloadJpg: () => void;
}

const MapRenderer = forwardRef<MapRendererHandle>((_, ref) => {
  const { layout, violations } = useStore();
  const svgRef = useRef<SVGSVGElement>(null);

  useImperativeHandle(ref, () => ({
    downloadJpg: () => {
      if (!svgRef.current || !layout) return;

      const svgNode = svgRef.current;
      const zoomGroup = d3.select(svgNode).select("g");
      
      // Save state
      const prevTransform = zoomGroup.attr("transform");
      const prevWidth = svgNode.getAttribute("width");
      const prevHeight = svgNode.getAttribute("height");
      const prevViewBox = svgNode.getAttribute("viewBox");

      // Prepare for snapshot: Reset zoom and set exact dimensions
      zoomGroup.attr("transform", null); 
      svgNode.setAttribute("width", layout.width.toString());
      svgNode.setAttribute("height", layout.height.toString());
      svgNode.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);

      // Serialize
      const serializer = new XMLSerializer();
      let svgString = serializer.serializeToString(svgNode);
      
      // Restore state immediately
      if (prevTransform) zoomGroup.attr("transform", prevTransform);
      if (prevWidth) svgNode.setAttribute("width", prevWidth); else svgNode.removeAttribute("width");
      if (prevHeight) svgNode.setAttribute("height", prevHeight); else svgNode.removeAttribute("height");
      if (prevViewBox) svgNode.setAttribute("viewBox", prevViewBox); else svgNode.removeAttribute("viewBox");

      // Fix namespaces for standalone SVG usage
      if (!svgString.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
        svgString = svgString.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
      }

      // Convert to JPG
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = layout.width;
        canvas.height = layout.height;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#0f172a"; // Background color
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          
          const jpgUrl = canvas.toDataURL("image/jpeg", 0.9);
          const link = document.createElement("a");
          link.download = `parking_layout_semantic_${Date.now()}.jpg`;
          link.href = jpgUrl;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      };
      // Handle unicode chars in SVG string
      img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgString)));
    }
  }));

  useEffect(() => {
    if (!svgRef.current || !layout) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous

    // 1. Zoom Group
    const zoomGroup = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on("zoom", (e) => zoomGroup.attr("transform", e.transform));
    svg.call(zoom);

    // 2. Background
    zoomGroup.append("rect")
      .attr("width", layout.width || 800)
      .attr("height", layout.height || 600)
      .attr("fill", "#475569")
      .attr("shape-rendering", "crispEdges"); // Background can be crisp

    // 3. Sort Elements
    const zOrder = [ElementType.GROUND, ElementType.ROAD, ElementType.LANE_LINE, ElementType.SIDEWALK, ElementType.PARKING_SPACE, ElementType.CHARGING_STATION, ElementType.RAMP, ElementType.WALL, ElementType.PILLAR, ElementType.ENTRANCE, ElementType.EXIT];
    const sorted = [...layout.elements].sort((a, b) => {
        const ia = zOrder.indexOf(a.type as ElementType), ib = zOrder.indexOf(b.type as ElementType);
        return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });

    // 4. Render
    // CENTROID-BASED TRANSFORM: Translate to center -> Rotate. 
    // All internal shapes must be drawn relative to (0,0) as center (i.e., x = -width/2, y = -height/2).
    const groups = zoomGroup.selectAll("g.el")
      .data(sorted)
      .enter()
      .append("g")
      .attr("transform", d => `translate(${d.x + d.width/2}, ${d.y + d.height/2}) rotate(${d.rotation || 0})`);

    groups.each(function(d) {
        const g = d3.select(this);
        const style = ELEMENT_STYLES[d.type] || { fill: '#ccc', opacity: 1 };
        const isError = violations.some(v => v.elementId === d.id);
        const color = isError ? '#ef4444' : style.fill;

        // Half-width and Half-height for centering logic
        const hw = d.width / 2;
        const hh = d.height / 2;

        // Common attribute to prevent anti-aliasing artifacts on rotations
        const renderAttr = { "shape-rendering": "geometricPrecision" };

        // Custom Rendering for Ground Line
        if (d.type === ElementType.LANE_LINE) {
            const isVertical = d.height > d.width;
            if (isVertical) {
                 // Vertical line centered at x=0, stretching from -hh to hh
                 g.append("line")
                 .attr("x1", 0).attr("y1", -hh).attr("x2", 0).attr("y2", hh)
                 .attr("stroke", "#facc15").attr("stroke-width", 2).attr("stroke-dasharray", "8,8")
                 .attr(renderAttr as any);
            } else {
                 // Horizontal line centered at y=0, stretching from -hw to hw
                 g.append("line")
                 .attr("x1", -hw).attr("y1", 0).attr("x2", hw).attr("y2", 0)
                 .attr("stroke", "#facc15").attr("stroke-width", 2).attr("stroke-dasharray", "8,8")
                 .attr(renderAttr as any);
            }
            return;
        }

        // Custom Rendering for Sidewalk
        if (d.type === ElementType.SIDEWALK) {
            // Background rect from -hw, -hh
            g.append("rect")
             .attr("x", -hw).attr("y", -hh)
             .attr("width", d.width).attr("height", d.height)
             .attr("fill", "#1e293b").attr("opacity", 1)
             .attr(renderAttr as any);

            const isVertical = d.height > d.width;
            const stripeCount = 3;
            
            if (isVertical) {
                const h = d.height / (stripeCount * 2 + 1);
                for(let i=0; i<stripeCount; i++) {
                    g.append("rect")
                     .attr("x", -hw)
                     .attr("y", -hh + h + i * 2 * h) // Offset from top (-hh)
                     .attr("width", d.width)
                     .attr("height", h)
                     .attr("fill", "#cbd5e1")
                     .attr(renderAttr as any);
                }
            } else {
                const w = d.width / (stripeCount * 2 + 1);
                for(let i=0; i<stripeCount; i++) {
                     g.append("rect")
                     .attr("x", -hw + w + i * 2 * w) // Offset from left (-hw)
                     .attr("y", -hh)
                     .attr("width", w)
                     .attr("height", d.height)
                     .attr("fill", "#cbd5e1")
                     .attr(renderAttr as any);
                }
            }
            return;
        }

        // Standard Rectangular Elements
        g.append("rect")
         .attr("x", -hw)
         .attr("y", -hh)
         .attr("width", d.width)
         .attr("height", d.height)
         .attr("fill", color)
         .attr("stroke", isError ? "red" : "none")
         .attr("stroke-width", isError ? 2 : 0)
         .attr("rx", (d.type === ElementType.PILLAR) ? 4 : 0)
         .attr(renderAttr as any);
        
        // Guidance Arrow
        if (d.type === ElementType.GUIDANCE_SIGN) {
            const s = Math.min(d.width, d.height) * 0.8;
            // Center is 0,0
            const cx = 0;
            const cy = 0; 
            g.append("path")
             .attr("d", `M ${cx-s/2} ${cy+s/4} L ${cx} ${cy-s/2} L ${cx+s/2} ${cy+s/4} M ${cx} ${cy-s/2} L ${cx} ${cy+s/2}`)
             .attr("stroke", "white").attr("fill", "none").attr("stroke-width", 2)
             .attr("stroke-linecap", "round")
             .attr("stroke-linejoin", "round")
             .attr(renderAttr as any);
        }
        
        // Charging Station
        if (d.type === ElementType.CHARGING_STATION) {
             const cx = 0, cy = 0;
             g.append("text")
              .attr("x", cx).attr("y", cy + 2) // Adjusted y for visual centering
              .attr("text-anchor", "middle")
              .attr("alignment-baseline", "middle")
              .attr("fill", "white")
              .attr("font-size", Math.min(d.width, d.height) * 0.8)
              .attr("font-weight", "bold")
              .text("⚡")
              .attr(renderAttr as any);
        }
    });

    // Auto-Fit Logic
    if (layout.width > 0 && svgRef.current?.parentElement) {
        const { clientWidth: pw, clientHeight: ph } = svgRef.current.parentElement;
        const scale = Math.min(pw / layout.width, ph / layout.height) * 0.95;
        const tx = (pw - layout.width * scale) / 2;
        const ty = (ph - layout.height * scale) / 2;
        svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    }
  }, [layout, violations]);

  return (
    <div className="w-full h-full bg-slate-950 overflow-hidden relative border border-slate-700 rounded-lg">
      <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing"></svg>
    </div>
  );
});

export default MapRenderer;