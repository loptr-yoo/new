import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import * as d3 from 'd3';
import { ElementType, LayoutElement } from '../types';
import { useStore } from '../store';

const ELEMENT_STYLES: Record<string, { fill: string; opacity: number }> = {
  [ElementType.GROUND]: { fill: '#334155', opacity: 1 }, 
  [ElementType.ROAD]: { fill: '#1e293b', opacity: 1 },   
  [ElementType.PARKING_SPACE]: { fill: '#3b82f6', opacity: 0.9 },
  [ElementType.SIDEWALK]: { fill: '#1e293b', opacity: 1 }, 
  [ElementType.RAMP]: { fill: '#c026d3', opacity: 1 },
  [ElementType.PILLAR]: { fill: '#94a3b8', opacity: 1 },
  [ElementType.WALL]: { fill: '#f1f5f9', opacity: 1 },
  [ElementType.ENTRANCE]: { fill: '#15803d', opacity: 1 },
  [ElementType.EXIT]: { fill: '#b91c1c', opacity: 1 },
  [ElementType.STAIRCASE]: { fill: '#fbbf24', opacity: 1 },
  [ElementType.ELEVATOR]: { fill: '#06b6d4', opacity: 1 },
  [ElementType.CHARGING_STATION]: { fill: '#22c55e', opacity: 1 },
  [ElementType.GUIDANCE_SIGN]: { fill: '#f59e0b', opacity: 1 },
  [ElementType.SAFE_EXIT]: { fill: '#10b981', opacity: 1 },
  [ElementType.SPEED_BUMP]: { fill: '#eab308', opacity: 1 },
  [ElementType.FIRE_EXTINGUISHER]: { fill: '#ef4444', opacity: 1 },
  [ElementType.LANE_LINE]: { fill: 'none', opacity: 1 },
  [ElementType.CONVEX_MIRROR]: { fill: '#38bdf8', opacity: 1 }
};

export interface MapRendererHandle {
  downloadJpg: () => void;
}

const MapRenderer = forwardRef<MapRendererHandle>((props, ref) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { layout, violations } = useStore();

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
      if (!svgString.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
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
          ctx.fillStyle = "#0f172a"; 
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
    
    const zOrder = [
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
    
    const sortedElements = [...elements].sort((a, b) => {
      const idxA = zOrder.indexOf(a.type as ElementType);
      const idxB = zOrder.indexOf(b.type as ElementType);
      return (idxA === -1 ? 999 : idxA) - (idxB === -1 ? 999 : idxB);
    });

    mainGroup.selectAll("g.element")
      .data(sortedElements)
      .enter()
      .append("g")
      .attr("class", "element")
      .attr("transform", d => `translate(${d.x}, ${d.y})`)
      .each(function(this: any, d) {
        const g = d3.select(this);
        const style = ELEMENT_STYLES[d.type as string] || { fill: '#ff00ff', opacity: 1 };
        const w = d.width;
        const h = d.height;
        const cx = w / 2;
        const cy = h / 2;

        if (d.type === ElementType.LANE_LINE) {
             const isVertical = h > w;
             g.append("line")
               .attr("x1", isVertical ? cx : 0).attr("y1", isVertical ? 0 : cy)
               .attr("x2", isVertical ? cx : w).attr("y2", isVertical ? h : cy)
               .attr("stroke", "#facc15").attr("stroke-width", 1.5).attr("stroke-dasharray", "8,8")
               .style("shape-rendering", "geometricPrecision"); 
        } 
        else if (d.type === ElementType.SIDEWALK) {
            g.append("rect").attr("width", w).attr("height", h).attr("fill", style.fill).attr("opacity", 0.3);
            const isHorizontal = w > h;
            const stripeSize = 4;
            const gap = 4;
            
            if (isHorizontal) {
                const count = Math.floor(w / (stripeSize + gap));
                for(let i=0; i<count; i++) {
                    g.append("rect")
                        .attr("x", i * (stripeSize + gap))
                        .attr("y", 0)
                        .attr("width", stripeSize)
                        .attr("height", h)
                        .attr("fill", "#e2e8f0");
                }
            } else {
                const count = Math.floor(h / (stripeSize + gap));
                for(let i=0; i<count; i++) {
                    g.append("rect")
                        .attr("x", 0)
                        .attr("y", i * (stripeSize + gap))
                        .attr("width", w)
                        .attr("height", stripeSize)
                        .attr("fill", "#e2e8f0");
                }
            }
        }
        else if (d.type === ElementType.SPEED_BUMP) {
             // 🛡️ 健壮性修复：即时上下文纠错 (Just-in-Time Correction)
             // 目的：即使上游给出的数据方向错误（如在横向道路上给了横向减速带），
             //      渲染层也能强制将其修正为垂直于道路的状态。

             // 1. 获取上下文：找到该减速带所在的"父道路"
             // 利用闭包访问 layout.elements，通过简单的中心点包含检测
             const cx = d.x + w / 2;
             const cy = d.y + h / 2;
             const parentRoad = layout?.elements.find(r => 
                 r.type === ElementType.ROAD && 
                 cx >= r.x && cx <= r.x + r.width &&
                 cy >= r.y && cy <= r.y + r.height
             );

             // 2. 准备渲染参数
             let renderW = w;
             let renderH = h;
             let offsetX = 0;
             let offsetY = 0;

             // 3. 逻辑校验与纠错
             if (parentRoad) {
                 const isRoadHorizontal = parentRoad.width > parentRoad.height;
                 const isBumpHorizontal = w > h;

                 // 核心规则：减速带必须"切断"道路（即方向应互相垂直）
                 // 如果方向一致（例如都是横向），说明数据错了，必须强制旋转 90 度
                 if (isRoadHorizontal === isBumpHorizontal) {
                     // 交换宽高
                     renderW = h;
                     renderH = w;
                     
                     // 计算偏移量，确保旋转后中心点位置不变
                     // 原中心(w/2, h/2)，新中心(renderW/2 + offX, renderH/2 + offY)
                     offsetX = (w - renderW) / 2;
                     offsetY = (h - renderH) / 2;
                 }
             }

             // 4. 绘制修正后的矩形
             g.append("rect")
               .attr("x", offsetX)
               .attr("y", offsetY)
               .attr("width", renderW)
               .attr("height", renderH)
               .attr("fill", style.fill) // 严格跟随全局语义颜色
               .attr("rx", 2);           // 微小圆角，提升精致感
        
        }
        else if (d.type === ElementType.GUIDANCE_SIGN) {
            g.append("rect").attr("width", w).attr("height", h).attr("fill", style.fill).attr("rx", 2);
            const s = Math.min(w, h) * 0.7;
            const rot = d.rotation || 0;
            g.append("path")
             .attr("d", `M ${cx - s/4} ${cy - s/2} L ${cx + s/2} ${cy} L ${cx - s/4} ${cy + s/2} M ${cx + s/2} ${cy} L ${cx - s/2} ${cy}`)
             .attr("stroke", "white").attr("fill", "none").attr("stroke-width", 2)
             .attr("stroke-linecap", "round").attr("stroke-linejoin", "round")
             .attr("transform", `rotate(${rot}, ${cx}, ${cy})`)
             .style("shape-rendering", "geometricPrecision");
        } 
        else {
            const rect = g.append("rect")
              .attr("width", w).attr("height", h)
              .attr("fill", style.fill)
              .attr("opacity", style.opacity)
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
  }, [layout, violations]);

  return (
    <div className="w-full h-full bg-slate-950 rounded-lg overflow-hidden border border-slate-800 shadow-inner relative">
       <svg ref={svgRef} className="block cursor-grab active:cursor-grabbing w-full h-full" />
    </div>
  );
});

export default MapRenderer;