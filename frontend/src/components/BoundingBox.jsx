import React, { useState, useRef, useEffect } from 'react';

const BoundingBox = ({
  bbox,
  color = 'green',
  isEditable = false,
  isSelected = false,
  onUpdate,
  onDelete,
  onSelect,
  score,
  label, // Label text to display on the box (e.g., LLM-assigned label)
  overlappingTags = [], // Array of {tag_name, confidence} for overlapping detections
}) => {
  const [currentBbox, setCurrentBbox] = useState(bbox);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const boxRef = useRef(null);

  // Update local state when prop changes
  useEffect(() => {
    if (!isDragging && !isResizing) {
      setCurrentBbox(bbox);
    }
  }, [bbox, isDragging, isResizing]);

  const { x, y, w, h } = currentBbox;
  const borderColor = isSelected ? 'red' : color;

  const handleMouseDown = (e, handle = null) => {
    if (!isEditable) return;
    
    e.stopPropagation();
    
    if (onSelect) {
      onSelect();
    }

    const rect = e.currentTarget.getBoundingClientRect();
    setDragStart({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      bboxX: x,
      bboxY: y,
      bboxW: w,
      bboxH: h
    });

    if (handle) {
      setIsResizing(true);
      setResizeHandle(handle);
    } else {
      setIsDragging(true);
    }
  };

  const handleMouseMove = (e) => {
    if (!isEditable || (!isDragging && !isResizing)) return;

    const parent = boxRef.current?.parentElement;
    if (!parent) return;

    const parentRect = parent.getBoundingClientRect();
    const mouseX = e.clientX - parentRect.left;
    const mouseY = e.clientY - parentRect.top;

    let newX = x;
    let newY = y;
    let newW = w;
    let newH = h;

    if (isDragging) {
      newX = mouseX - dragStart.x;
      newY = mouseY - dragStart.y;
      
      // Keep within bounds
      newX = Math.max(0, Math.min(newX, parentRect.width - w));
      newY = Math.max(0, Math.min(newY, parentRect.height - h));
    } else if (isResizing) {
      const deltaX = mouseX - (dragStart.bboxX + dragStart.x);
      const deltaY = mouseY - (dragStart.bboxY + dragStart.y);

      switch (resizeHandle) {
        case 'nw':
          newX = dragStart.bboxX + deltaX;
          newY = dragStart.bboxY + deltaY;
          newW = dragStart.bboxW - deltaX;
          newH = dragStart.bboxH - deltaY;
          break;
        case 'ne':
          newY = dragStart.bboxY + deltaY;
          newW = dragStart.bboxW + deltaX;
          newH = dragStart.bboxH - deltaY;
          break;
        case 'sw':
          newX = dragStart.bboxX + deltaX;
          newW = dragStart.bboxW - deltaX;
          newH = dragStart.bboxH + deltaY;
          break;
        case 'se':
          newW = dragStart.bboxW + deltaX;
          newH = dragStart.bboxH + deltaY;
          break;
        case 'n':
          newY = dragStart.bboxY + deltaY;
          newH = dragStart.bboxH - deltaY;
          break;
        case 's':
          newH = dragStart.bboxH + deltaY;
          break;
        case 'w':
          newX = dragStart.bboxX + deltaX;
          newW = dragStart.bboxW - deltaX;
          break;
        case 'e':
          newW = dragStart.bboxW + deltaX;
          break;
        default:
          break;
      }

      // Minimum size
      if (newW < 20) newW = 20;
      if (newH < 20) newH = 20;
    }

    // Update local state immediately for smooth dragging
    setCurrentBbox({ x: newX, y: newY, w: newW, h: newH });
  };

  const handleMouseUp = () => {
    // Call onUpdate only when drag/resize is complete
    if ((isDragging || isResizing) && onUpdate) {
      onUpdate(currentBbox);
    }
    
    setIsDragging(false);
    setIsResizing(false);
    setResizeHandle(null);
  };

  React.useEffect(() => {
    if (isDragging || isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, isResizing, dragStart]);

  const handleSize = 8;

  // Check if there are overlapping tags to show
  const hasOverlappingTags = overlappingTags && overlappingTags.length > 0;
  
  // Determine if tooltip should show above or below based on bbox position
  // Show below if bbox is near the top (y < 150px estimated tooltip height)
  const showTooltipBelow = y < 150;

  return (
    <div
      ref={boxRef}
      style={{
        position: 'absolute',
        left: `${x}px`,
        top: `${y}px`,
        width: `${w}px`,
        height: `${h}px`,
        border: `2px solid ${borderColor}`,
        boxShadow: `0 0 0 1px rgba(0,0,0,0.3)`,
        cursor: isEditable ? (isDragging ? 'move' : 'pointer') : 'default',
        boxSizing: 'border-box',
        pointerEvents: 'auto'
      }}
      onMouseDown={(e) => handleMouseDown(e)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="bbox-overlay"
    >
      {/* Hover tooltip showing all overlapping tags */}
      {isHovered && hasOverlappingTags && (
        <div
          style={{
            position: 'absolute',
            // Position above or below based on available space
            ...(showTooltipBelow ? {
              top: '100%',
              marginTop: '8px',
            } : {
              bottom: '100%',
              marginBottom: '8px',
            }),
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '8px 12px',
            backgroundColor: 'rgba(30, 30, 30, 0.95)',
            color: '#fff',
            borderRadius: '6px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            zIndex: 1000,
            minWidth: '120px',
            maxWidth: '250px',
          }}
        >
          <div style={{ fontSize: '10px', color: '#aaa', marginBottom: '4px', fontWeight: 'bold' }}>
            Detected Tags ({overlappingTags.length})
          </div>
          {overlappingTags.map((tag, idx) => (
            <div 
              key={idx} 
              style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '3px 0',
                borderBottom: idx < overlappingTags.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none',
              }}
            >
              <span style={{ fontSize: '12px', fontWeight: '500' }}>
                {tag.tag_name || 'Unknown'}
              </span>
              <span style={{ 
                fontSize: '11px', 
                backgroundColor: tag.confidence > 0.7 ? '#22c55e' : tag.confidence > 0.4 ? '#f59e0b' : '#ef4444',
                padding: '1px 6px',
                borderRadius: '10px',
                marginLeft: '8px',
              }}>
                {Math.round(tag.confidence * 100)}%
              </span>
            </div>
          ))}
          {/* Tooltip arrow - points up or down based on position */}
          <div style={{
            position: 'absolute',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 0,
            height: 0,
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            ...(showTooltipBelow ? {
              top: '-6px',
              borderBottom: '6px solid rgba(30, 30, 30, 0.95)',
            } : {
              bottom: '-6px',
              borderTop: '6px solid rgba(30, 30, 30, 0.95)',
            }),
          }} />
        </div>
      )}
      
      {/* Label display (for AI-matched icons) */}
      {label && (
        <div
          style={{
            position: 'absolute',
            top: -18,
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '11px',
            fontWeight: 'bold',
            padding: '1px 6px',
            backgroundColor: borderColor,
            color: '#fff',
            borderRadius: '3px',
            whiteSpace: 'nowrap',
            boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
          }}
        >
          {label}
        </div>
      )}
      {/* Confidence score display */}
      {score !== undefined && !label && (
        <div
          style={{
            position: 'absolute',
            top: -14,
            left: 0,
            fontSize: '10px',
            padding: '0 2px',
            backgroundColor: 'rgba(0,0,0,0.6)',
            color: '#fff',
            borderRadius: '2px',
          }}
        >
          {Math.round(score * 100)}%
        </div>
      )}
      {isEditable && isSelected && (
        <>
          {/* Corner handles */}
          {['nw', 'ne', 'sw', 'se'].map((handle) => (
            <div
              key={handle}
              onMouseDown={(e) => handleMouseDown(e, handle)}
              style={{
                position: 'absolute',
                width: `${handleSize}px`,
                height: `${handleSize}px`,
                backgroundColor: borderColor,
                border: '1px solid white',
                cursor: `${handle}-resize`,
                ...(handle.includes('n') ? { top: -handleSize / 2 } : { bottom: -handleSize / 2 }),
                ...(handle.includes('w') ? { left: -handleSize / 2 } : { right: -handleSize / 2 }),
              }}
            />
          ))}
          
          {/* Edge handles */}
          {['n', 's', 'e', 'w'].map((handle) => (
            <div
              key={handle}
              onMouseDown={(e) => handleMouseDown(e, handle)}
              style={{
                position: 'absolute',
                backgroundColor: borderColor,
                border: '1px solid white',
                cursor: `${handle === 'n' || handle === 's' ? 'ns' : 'ew'}-resize`,
                ...(handle === 'n' && { top: -handleSize / 2, left: '50%', transform: 'translateX(-50%)', width: `${handleSize}px`, height: `${handleSize}px` }),
                ...(handle === 's' && { bottom: -handleSize / 2, left: '50%', transform: 'translateX(-50%)', width: `${handleSize}px`, height: `${handleSize}px` }),
                ...(handle === 'e' && { right: -handleSize / 2, top: '50%', transform: 'translateY(-50%)', width: `${handleSize}px`, height: `${handleSize}px` }),
                ...(handle === 'w' && { left: -handleSize / 2, top: '50%', transform: 'translateY(-50%)', width: `${handleSize}px`, height: `${handleSize}px` }),
              }}
            />
          ))}

          {/* Delete button */}
          {onDelete && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
              className="absolute -top-8 -right-8 bg-danger text-white rounded-full w-6 h-6 flex items-center justify-center hover:bg-red-600"
              title="Delete"
            >
              ×
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default BoundingBox;

