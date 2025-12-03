import React, { useState, useRef, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import BoundingBox from './BoundingBox';
import { toPixels, toNormalized } from '../utils/pdfHelpers';

// Set up the worker for react-pdf - use worker from public directory
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

const PDFViewer = ({ 
  pdfUrl, 
  boundingBoxes = [], 
  selectedBoxId = null,
  onBboxUpdate,
  onBboxCreate,
  onBboxDelete,
  onBboxSelect,
  scrollToPage = null,
  scrollToBbox = null, // { bbox_normalized: [x, y, w, h], page_number: number }
  isEditable = false,
  createMode = false
}) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [pageWidth, setPageWidth] = useState(null);
  const [pageHeight, setPageHeight] = useState(null);
  const [scale, setScale] = useState(1.0);
  const MAX_SCALE = 6.0;
  const MIN_SCALE = 0.5;
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [currentDraw, setCurrentDraw] = useState(null);
  
  const containerRef = useRef(null);
  const pageRef = useRef(null);
  const [baseWidth, setBaseWidth] = useState(null);

  useEffect(() => {
    if (scrollToPage !== null && scrollToPage !== pageNumber) {
      setPageNumber(scrollToPage);
    }
  }, [scrollToPage]);

  // Scroll to center the bbox in the viewer
  useEffect(() => {
    if (!scrollToBbox || !containerRef.current || !pageRef.current) return;
    
    // First, navigate to the correct page if needed
    if (scrollToBbox.page_number && scrollToBbox.page_number !== pageNumber) {
      setPageNumber(scrollToBbox.page_number);
      // Wait for page to render before scrolling - will re-trigger when pageNumber changes
      return;
    }
    
    // Use a small delay to ensure the page has rendered after page change
    const scrollTimeout = setTimeout(() => {
      const container = containerRef.current;
      const pageElement = pageRef.current;
      if (!container || !pageElement) return;
      
      // Get the actual canvas element inside the page
      const canvas = pageElement.querySelector('canvas');
      if (!canvas) return;
      
      // Get canvas dimensions (the actual rendered PDF page size)
      const canvasRect = canvas.getBoundingClientRect();
      const canvasWidth = canvasRect.width;
      const canvasHeight = canvasRect.height;
      
      // bbox_normalized format is [ymin, xmin, ymax, xmax] in 0-1000 scale
      const [ymin, xmin, ymax, xmax] = scrollToBbox.bbox_normalized;
      
      // Calculate center of bbox in pixel coordinates
      const bboxCenterXInCanvas = ((xmin + xmax) / 2 / 1000) * canvasWidth;
      const bboxCenterYInCanvas = ((ymin + ymax) / 2 / 1000) * canvasHeight;
      
      // Get canvas position relative to container's scroll area
      const containerRect = container.getBoundingClientRect();
      const canvasOffsetX = canvasRect.left - containerRect.left + container.scrollLeft;
      const canvasOffsetY = canvasRect.top - containerRect.top + container.scrollTop;
      
      // Calculate absolute position of bbox center in the scroll container
      const bboxAbsoluteX = canvasOffsetX + bboxCenterXInCanvas;
      const bboxAbsoluteY = canvasOffsetY + bboxCenterYInCanvas;
      
      // Get container visible dimensions
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      
      // Calculate scroll position to center the bbox
      const scrollX = bboxAbsoluteX - containerWidth / 2;
      const scrollY = bboxAbsoluteY - containerHeight / 2;
      
      // Smooth scroll to the position
      container.scrollTo({
        left: Math.max(0, scrollX),
        top: Math.max(0, scrollY),
        behavior: 'smooth'
      });
    }, 200); // Delay to ensure rendering is complete
    
    return () => clearTimeout(scrollTimeout);
  }, [scrollToBbox, pageNumber]);

  useEffect(() => {
    const updateBaseWidth = () => {
      if (containerRef.current) {
        // Set base width for PDF at 100% scale - smaller so zoom has room
        const availableWidth = containerRef.current.offsetWidth - 48; // account for padding
        setBaseWidth(Math.min(availableWidth * 0.9, 700)); // 90% of available, max 700px
      }
    };
    updateBaseWidth();
    window.addEventListener('resize', updateBaseWidth);
    return () => window.removeEventListener('resize', updateBaseWidth);
  }, []);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  const onPageLoadSuccess = (page) => {
    // Get the actual rendered dimensions from the page element
    setTimeout(() => {
      if (pageRef.current) {
        const canvas = pageRef.current.querySelector('canvas');
        if (canvas) {
          // Use the display dimensions, not internal canvas resolution
          const rect = canvas.getBoundingClientRect();
          setPageWidth(rect.width);
          setPageHeight(rect.height);
        }
      }
    }, 100);
  };

  const handleMouseDown = (e) => {
    if (!createMode) return;
    
    const rect = pageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setIsDrawing(true);
    setDrawStart({ x, y });
    setCurrentDraw({ x, y, w: 0, h: 0 });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !drawStart) return;
    
    const rect = pageRef.current.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    const x = Math.min(drawStart.x, currentX);
    const y = Math.min(drawStart.y, currentY);
    const w = Math.abs(currentX - drawStart.x);
    const h = Math.abs(currentY - drawStart.y);
    
    setCurrentDraw({ x, y, w, h });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !currentDraw || currentDraw.w < 10 || currentDraw.h < 10) {
      setIsDrawing(false);
      setDrawStart(null);
      setCurrentDraw(null);
      return;
    }
    
    // Convert to normalized coordinates
    const normalized = toNormalized(
      currentDraw.x,
      currentDraw.y,
      currentDraw.w,
      currentDraw.h,
      pageWidth,
      pageHeight
    );
    
    if (onBboxCreate) {
      onBboxCreate({
        bbox_normalized: normalized,
        page_number: pageNumber
      });
    }
    
    setIsDrawing(false);
    setDrawStart(null);
    setCurrentDraw(null);
  };

  const handleBboxUpdate = (boxId, newPixelBbox) => {
    if (!pageWidth || !pageHeight) return;
    
    const normalized = toNormalized(
      newPixelBbox.x,
      newPixelBbox.y,
      newPixelBbox.w,
      newPixelBbox.h,
      pageWidth,
      pageHeight
    );
    
    console.log('PDFViewer handleBboxUpdate:', { 
      boxId, 
      pixelBbox: newPixelBbox, 
      normalized,
      pageSize: { width: pageWidth, height: pageHeight }
    });
    
    if (onBboxUpdate) {
      onBboxUpdate(boxId, { bbox_normalized: normalized });
    }
  };

  const changePage = (offset) => {
    setPageNumber(prevPageNumber => {
      const newPage = prevPageNumber + offset;
      return Math.max(1, Math.min(newPage, numPages));
    });
  };

  return (
    <div className="flex flex-col h-full bg-gray-100" style={{ overflow: 'hidden', maxWidth: '100%', maxHeight: '100%' }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-white border-b flex-shrink-0">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => changePage(-1)}
            disabled={pageNumber <= 1}
            className="px-3 py-1 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed rounded"
          >
            Previous
          </button>
          <span className="text-sm">
            Page {pageNumber} of {numPages || '--'}
          </span>
          <button
            onClick={() => changePage(1)}
            disabled={pageNumber >= numPages}
            className="px-3 py-1 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed rounded"
          >
            Next
          </button>
        </div>
        
        <div className="flex items-center space-x-3">
          <span className="text-xs text-gray-500">🔍−</span>
          <input
            type="range"
            min={MIN_SCALE * 100}
            max={MAX_SCALE * 100}
            value={scale * 100}
            onChange={(e) => setScale(Number(e.target.value) / 100)}
            className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            title={`Zoom: ${Math.round(scale * 100)}%`}
          />
          <span className="text-xs text-gray-500">🔍+</span>
          <span className="text-sm font-medium w-12 text-center">{Math.round(scale * 100)}%</span>
          <button
            onClick={() => setScale(1.0)}
            className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
            disabled={scale === 1.0}
            title="Reset zoom to 100%"
          >
            Reset
          </button>
        </div>
      </div>

      {/* PDF Canvas - Fixed container with scroll */}
      <div 
        ref={containerRef} 
        className="flex-1 bg-gray-50 p-4" 
        style={{ 
          minHeight: 0, 
          overflow: 'auto',
          maxWidth: '100%',
          maxHeight: '100%'
        }}
      >
        <div style={{ display: 'inline-block' }}>
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={<div className="text-center p-8">Loading PDF...</div>}
            error={<div className="text-center p-8 text-red-500">Error loading PDF</div>}
          >
            <div
              ref={pageRef}
              className="relative"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              style={{ 
                cursor: createMode ? 'crosshair' : 'default',
                display: 'inline-block'
              }}
            >
              <Page
                pageNumber={pageNumber}
                width={baseWidth}
                scale={scale}
                onLoadSuccess={onPageLoadSuccess}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
              
              {/* Render bounding boxes */}
              {pageWidth && pageHeight && boundingBoxes
                .filter(box => box.page_number === pageNumber)
                .map((box) => {
                  const pixelBbox = toPixels(box.bbox_normalized, pageWidth, pageHeight);
                  console.log('Rendering BoundingBox:', box.id, 'with bbox_normalized:', box.bbox_normalized, 'pixels:', pixelBbox);
                  const padding = Math.max(2, Math.min(pixelBbox.w, pixelBbox.h) * 0.02);
                  const paddedBbox = {
                    x: Math.max(0, pixelBbox.x - padding),
                    y: Math.max(0, pixelBbox.y - padding),
                    w: Math.min(pageWidth - pixelBbox.x + padding, pixelBbox.w + padding * 2),
                    h: Math.min(pageHeight - pixelBbox.y + padding, pixelBbox.h + padding * 2),
                  };
                  return (
                    <BoundingBox
                      key={box.id}
                      bbox={paddedBbox}
                      color={box.color || 'green'}
                      isEditable={isEditable}
                      isSelected={box.id === selectedBoxId}
                      onUpdate={(newBbox) => handleBboxUpdate(box.id, newBbox)}
                      onDelete={onBboxDelete ? () => onBboxDelete(box.id) : undefined}
                      onSelect={() => onBboxSelect && onBboxSelect(box.id)}
                      score={box.confidence}
                      label={box.label}
                      overlappingTags={box.overlappingTags}
                    />
                  );
                })}
              
              {/* Current drawing box */}
              {isDrawing && currentDraw && (
                <div
                  style={{
                    position: 'absolute',
                    left: `${currentDraw.x}px`,
                    top: `${currentDraw.y}px`,
                    width: `${currentDraw.w}px`,
                    height: `${currentDraw.h}px`,
                    border: '2px dashed blue',
                    backgroundColor: 'rgba(0, 0, 255, 0.1)',
                    pointerEvents: 'none'
                  }}
                />
              )}
            </div>
          </Document>
        </div>
      </div>
    </div>
  );
};

export default PDFViewer;

