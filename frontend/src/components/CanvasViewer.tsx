import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Sliders,
  Maximize2,
  Eye,
  EyeOff,
  Move
} from 'lucide-react';

interface LesionToggle {
  key: string;
  label: string;
  color: string;
  enabled: boolean;
}

interface CanvasViewerProps {
  originalImageUrl: string;
  predictedMaskUrl?: string | null;
  groundTruthUrl?: string | null;
  taskType?: string;
}

export const CanvasViewer: React.FC<CanvasViewerProps> = ({
  originalImageUrl,
  predictedMaskUrl,
  groundTruthUrl,
  taskType = 'IDRID_LESIONS',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Images in memory
  const origImgRef = useRef<HTMLImageElement | null>(null);
  const predImgRef = useRef<HTMLImageElement | null>(null);
  const gtImgRef = useRef<HTMLImageElement | null>(null);

  // Pan & Zoom Transform State
  const [scale, setScale] = useState<number>(1);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Display Mode: 'blend' | 'split' | 'original' | 'prediction' | 'groundtruth'
  const [viewMode, setViewMode] = useState<string>('blend');
  const [opacity, setOpacity] = useState<number>(0.65);
  const [splitPosition, setSplitPosition] = useState<number>(50); // % from left

  // Inspection Pixel State
  const [hoverCoords, setHoverCoords] = useState<{ x: number; y: number } | null>(null);

  // Lesion Class Toggles
  const [classes, setClasses] = useState<LesionToggle[]>([
    { key: 'MA', label: 'Microaneurysms (MA)', color: '#ef4444', enabled: true },
    { key: 'HE', label: 'Haemorrhages (HE)', color: '#f97316', enabled: true },
    { key: 'EX', label: 'Hard Exudates (EX)', color: '#eab308', enabled: true },
    { key: 'SE', label: 'Soft Exudates (SE)', color: '#22c55e', enabled: true },
    { key: 'OD', label: 'Optic Disc (OD)', color: '#3b82f6', enabled: true },
    { key: 'VESSEL', label: 'Blood Vessels', color: '#a855f7', enabled: true },
  ]);

  // Load Images into memory
  useEffect(() => {
    if (originalImageUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = originalImageUrl;
      img.onload = () => {
        origImgRef.current = img;
        resetView();
      };
    }

    if (predictedMaskUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = predictedMaskUrl;
      img.onload = () => {
        predImgRef.current = img;
        draw();
      };
    } else {
      predImgRef.current = null;
    }

    if (groundTruthUrl) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = groundTruthUrl;
      img.onload = () => {
        gtImgRef.current = img;
        draw();
      };
    } else {
      gtImgRef.current = null;
    }
  }, [originalImageUrl, predictedMaskUrl, groundTruthUrl]);

  const resetView = useCallback(() => {
    if (!canvasRef.current || !origImgRef.current) return;
    const canvas = canvasRef.current;
    const img = origImgRef.current;

    // Fit image to canvas width/height
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    const fitScale = Math.min(scaleX, scaleY, 1.0) * 0.95;

    const initialX = (canvas.width - img.naturalWidth * fitScale) / 2;
    const initialY = (canvas.height - img.naturalHeight * fitScale) / 2;

    setScale(fitScale);
    setOffset({ x: initialX, y: initialY });
  }, []);

  // Main Canvas Rendering Loop
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear background
    ctx.fillStyle = '#0f172a'; // slate-900 dark background for optimal medical viewing
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const orig = origImgRef.current;
    if (!orig || !orig.complete) return;

    ctx.save();
    // Apply pan and zoom
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    const imgW = orig.naturalWidth;
    const imgH = orig.naturalHeight;

    if (viewMode === 'original') {
      ctx.drawImage(orig, 0, 0);
    } else if (viewMode === 'prediction' && predImgRef.current) {
      ctx.drawImage(predImgRef.current, 0, 0, imgW, imgH);
    } else if (viewMode === 'groundtruth' && gtImgRef.current) {
      ctx.drawImage(gtImgRef.current, 0, 0, imgW, imgH);
    } else if (viewMode === 'split' && predImgRef.current) {
      // Split view: Left original, Right prediction
      const splitX = (imgW * splitPosition) / 100;
      
      // Draw original on left
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, imgH);
      ctx.clip();
      ctx.drawImage(orig, 0, 0);
      ctx.restore();

      // Draw prediction overlay on right
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, imgW - splitX, imgH);
      ctx.clip();
      ctx.drawImage(predImgRef.current, 0, 0, imgW, imgH);
      ctx.restore();

      // Draw split divider line
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2 / scale;
      ctx.beginPath();
      ctx.moveTo(splitX, 0);
      ctx.lineTo(splitX, imgH);
      ctx.stroke();
    } else {
      // Blend Mode (Original + translucent prediction overlay)
      ctx.drawImage(orig, 0, 0);

      if (predImgRef.current && opacity > 0) {
        ctx.save();
        ctx.globalAlpha = opacity;
        ctx.drawImage(predImgRef.current, 0, 0, imgW, imgH);
        ctx.restore();
      }
    }

    ctx.restore();
  }, [offset, scale, viewMode, opacity, splitPosition]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Handle window resizing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = containerRef.current.clientHeight;
        draw();
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  // Mouse Interaction: Pan & Zoom
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }

    // Coordinate calculation
    if (origImgRef.current && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const imgX = Math.round((mouseX - offset.x) / scale);
      const imgY = Math.round((mouseY - offset.y) / scale);

      if (
        imgX >= 0 &&
        imgY >= 0 &&
        imgX < origImgRef.current.naturalWidth &&
        imgY < origImgRef.current.naturalHeight
      ) {
        setHoverCoords({ x: imgX, y: imgY });
      } else {
        setHoverCoords(null);
      }
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const newScale = Math.max(0.1, Math.min(scale * zoomFactor, 10));

    // Zoom centered at cursor
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const newX = mouseX - (mouseX - offset.x) * (newScale / scale);
      const newY = mouseY - (mouseY - offset.y) * (newScale / scale);
      setScale(newScale);
      setOffset({ x: newX, y: newY });
    }
  };

  const toggleClass = (key: string) => {
    setClasses((prev) =>
      prev.map((c) => (c.key === key ? { ...c, enabled: !c.enabled } : c))
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-xl overflow-hidden shadow-xl border border-slate-800">
      {/* Top Toolbar */}
      <div className="bg-slate-800/90 border-b border-slate-700/80 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-300">
        {/* View Mode Tabs */}
        <div className="flex items-center space-x-1 bg-slate-900/80 p-1 rounded-lg border border-slate-700">
          <button
            onClick={() => setViewMode('blend')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              viewMode === 'blend' ? 'bg-sky-600 text-white' : 'hover:text-white'
            }`}
          >
            Overlay Blend
          </button>
          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              viewMode === 'split' ? 'bg-sky-600 text-white' : 'hover:text-white'
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              viewMode === 'original' ? 'bg-sky-600 text-white' : 'hover:text-white'
            }`}
          >
            Original
          </button>
          <button
            onClick={() => setViewMode('prediction')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              viewMode === 'prediction' ? 'bg-sky-600 text-white' : 'hover:text-white'
            }`}
          >
            AI Mask
          </button>
          {groundTruthUrl && (
            <button
              onClick={() => setViewMode('groundtruth')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                viewMode === 'groundtruth' ? 'bg-sky-600 text-white' : 'hover:text-white'
              }`}
            >
              Ground Truth
            </button>
          )}
        </div>

        {/* Opacity Slider for Blend Mode */}
        {viewMode === 'blend' && (
          <div className="flex items-center space-x-2">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400">Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-24 accent-sky-500 cursor-pointer"
            />
            <span className="w-8 text-[11px] font-mono text-slate-300">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        )}

        {/* Split Position Slider for Split Mode */}
        {viewMode === 'split' && (
          <div className="flex items-center space-x-2">
            <span className="text-[11px] text-slate-400">Split:</span>
            <input
              type="range"
              min="0"
              max="100"
              value={splitPosition}
              onChange={(e) => setSplitPosition(parseInt(e.target.value))}
              className="w-28 accent-sky-500 cursor-pointer"
            />
            <span className="w-8 text-[11px] font-mono text-slate-300">
              {splitPosition}%
            </span>
          </div>
        )}

        {/* Zoom & Pan Controls */}
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setScale((s) => Math.min(s * 1.25, 10))}
            title="Phóng to (Zoom In)"
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 hover:text-white transition"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setScale((s) => Math.max(s * 0.8, 0.1))}
            title="Thu nhỏ (Zoom Out)"
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 hover:text-white transition"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={resetView}
            title="Khôi phục góc nhìn (Reset View)"
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 hover:text-white transition"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <div className="text-[11px] font-mono text-sky-400 pl-2">
            {Math.round(scale * 100)}%
          </div>
        </div>
      </div>

      {/* Main Interactive Canvas Area */}
      <div ref={containerRef} className="relative flex-1 min-h-[480px] cursor-grab active:cursor-grabbing overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          className="absolute inset-0 block w-full h-full"
        />

        {/* Coordinates Pill */}
        {hoverCoords && (
          <div className="absolute bottom-3 left-3 bg-slate-900/90 border border-slate-700/80 px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 pointer-events-none shadow-md">
            X: {hoverCoords.x} | Y: {hoverCoords.y} px
          </div>
        )}

        {/* Pan helper hint */}
        <div className="absolute top-3 right-3 bg-slate-900/70 border border-slate-700/50 px-2 py-1 rounded text-[10px] text-slate-400 pointer-events-none flex items-center space-x-1">
          <Move className="w-3 h-3 text-slate-400" />
          <span>Kéo chuột để di chuyển (Pan)</span>
        </div>
      </div>

      {/* Bottom Lesion Legend & Layer Toggles */}
      <div className="bg-slate-800/90 border-t border-slate-700/80 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2 text-xs text-slate-400 font-medium">
          <Layers className="w-3.5 h-3.5 text-sky-400" />
          <span>Lớp phân vùng:</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {classes.map((c) => (
            <button
              key={c.key}
              onClick={() => toggleClass(c.key)}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition ${
                c.enabled
                  ? 'bg-slate-700/80 border-slate-600 text-slate-200'
                  : 'bg-slate-900/60 border-slate-800 text-slate-500 line-through'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: c.color }}
              />
              <span>{c.label}</span>
              {c.enabled ? (
                <Eye className="w-3 h-3 text-slate-400 ml-1" />
              ) : (
                <EyeOff className="w-3 h-3 text-slate-600 ml-1" />
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
