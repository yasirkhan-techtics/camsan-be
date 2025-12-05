import React, { useState, useEffect, useRef } from 'react';
import PDFViewer from './PDFViewer';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const LegendItemsSection = () => {
  const {
    selectedProject,
    pdfPages,
    legendTables,
    selectProject,
  } = useProject();
  
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionMessage, setExtractionMessage] = useState('');
  const [scrollToPage, setScrollToPage] = useState(null);
  const [isSelectingIcon, setIsSelectingIcon] = useState(false);
  const [isSelectingLabel, setIsSelectingLabel] = useState(false);
  const [iconTemplate, setIconTemplate] = useState(null);
  const [labelTemplates, setLabelTemplates] = useState([]); // Multiple label templates per item
  const [showTemplateSelector, setShowTemplateSelector] = useState(null); // 'icon' or 'label'
  const [tempBbox, setTempBbox] = useState(null); // Temporary bbox while drawing
  const [newTagName, setNewTagName] = useState(''); // Tag name for new label template
  const [editingLabelTemplateId, setEditingLabelTemplateId] = useState(null); // ID of template being edited
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawStart, setDrawStart] = useState(null);
  const [imageZoom, setImageZoom] = useState(1.0);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const imageContainerRef = useRef(null);

  // Aggregate all legend items from all legend tables
  const legendItems = React.useMemo(() => {
    return legendTables.flatMap(table => table.legend_items || []);
  }, [legendTables]);

  // Extract legend items from all legend tables
  const handleExtractLegendItems = async () => {
    if (!selectedProject || legendTables.length === 0) return;
    
    setIsExtracting(true);
    setExtractionMessage('Extracting legend items...');
    
    try {
      let totalExtracted = 0;
      
      for (let i = 0; i < legendTables.length; i++) {
        const table = legendTables[i];
        setExtractionMessage(`Extracting from table ${i + 1} of ${legendTables.length}...`);
        
        try {
          const response = await api.extractLegendItems(selectedProject.id, table.id);
          totalExtracted += response.data?.length || 0;
        } catch (error) {
          console.error(`Error extracting from table ${table.id}:`, error);
      }
      }
      
      // Refresh project to get updated legend tables with items
      await selectProject(selectedProject.id);
      
      setExtractionMessage(`✅ Extracted ${totalExtracted} legend items!`);
      setTimeout(() => setExtractionMessage(''), 3000);
    } catch (error) {
      console.error('Error extracting legend items:', error);
      setExtractionMessage(`❌ Error: ${error.message}`);
    } finally {
      setIsExtracting(false);
    }
  };

  if (!selectedProject) {
    return <div className="p-8 text-center">No project selected</div>;
  }

  const pdfUrl = api.getProjectPdf(selectedProject.id);

  const handleItemClick = async (item) => {
    setSelectedItemId(item.id);
    setIsSelectingIcon(false);
    setIsSelectingLabel(false);
    
    // Try to load icon template for this item
    try {
      const response = await api.getIconTemplate(item.id);
      setIconTemplate(response.data);
    } catch (error) {
      setIconTemplate(null); // No icon template yet
    }
    
    // Try to load all label templates for this item (supports multiple tags)
    try {
      const response = await api.getLabelTemplates(item.id);
      setLabelTemplates(response.data || []);
    } catch (error) {
      setLabelTemplates([]); // No label templates yet
    }
    
    // Optionally scroll to the legend table page
    const legendTable = legendTables.find(t => t.id === item.legend_table_id);
    if (legendTable) {
      const page = pdfPages?.find(p => p.id === legendTable.page_id);
      if (page) {
        setScrollToPage(page.page_number);
      }
    }
  };

  const handleSelectIcon = () => {
    setShowTemplateSelector('icon');
    setTempBbox(null);
    setImageZoom(1.0);
  };

  const handleSelectLabel = (templateId = null) => {
    setShowTemplateSelector('label');
    setTempBbox(null);
    setImageZoom(1.0);
    setEditingLabelTemplateId(templateId);
    setNewTagName('');
  };

  const handleSaveIconTemplate = async () => {
    if (!tempBbox || !selectedItemId) return;
    
    // Validate bbox
    if (tempBbox.width <= 0 || tempBbox.height <= 0) {
      alert('Invalid bounding box! Width and height must be greater than 0.');
      return;
    }
    
    try {
      console.log('📦 Saving icon bbox directly from legend table image:', tempBbox);
      console.log('📏 Image dimensions:', {
        natural: imageRef.current ? { w: imageRef.current.naturalWidth, h: imageRef.current.naturalHeight } : 'N/A',
        display: imageRef.current ? imageRef.current.getBoundingClientRect() : 'N/A'
      });
      
      const response = await api.drawIconBbox(selectedItemId, tempBbox);
      setIconTemplate(response.data);
      setShowTemplateSelector(null);
      setTempBbox(null);
      alert('Icon template saved successfully!');
    } catch (error) {
      console.error('Error saving icon bbox:', error);
      alert('Error saving icon: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleSaveLabelTemplate = async () => {
    if (!tempBbox || !selectedItemId) return;
    
    try {
      console.log('📝 Saving label bbox directly from legend table image:', tempBbox);
      const response = await api.drawLabelBbox(
        selectedItemId, 
        tempBbox, 
        newTagName || null, 
        editingLabelTemplateId
      );
      
      // Update the label templates list
      if (editingLabelTemplateId) {
        // Update existing template in the list
        setLabelTemplates(prev => prev.map(t => 
          t.id === editingLabelTemplateId ? response.data : t
        ));
      } else {
        // Add new template to the list
        setLabelTemplates(prev => [...prev, response.data]);
      }
      
      setShowTemplateSelector(null);
      setTempBbox(null);
      setNewTagName('');
      setEditingLabelTemplateId(null);
      alert(`Label template ${editingLabelTemplateId ? 'updated' : 'saved'} successfully!`);
    } catch (error) {
      console.error('Error saving label bbox:', error);
      alert('Error saving label: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleDeleteLabelTemplate = async (templateId) => {
    if (!window.confirm('Delete this tag template? This will also delete all detections for this tag.')) {
      return;
    }
    
    try {
      await api.deleteLabelTemplate(templateId);
      setLabelTemplates(prev => prev.filter(t => t.id !== templateId));
      alert('Tag template deleted successfully!');
    } catch (error) {
      console.error('Error deleting label template:', error);
      alert('Error deleting tag: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleImageMouseDown = (e) => {
    if (!imageRef.current) return;
    
    const img = imageRef.current;
    const rect = img.getBoundingClientRect();
    
    // Get click position relative to displayed image
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    
    // Convert to actual image coordinates
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    
    const x = Math.round(displayX * scaleX);
    const y = Math.round(displayY * scaleY);
    
    console.log('🖱️ Mouse down:', { displayX, displayY, actualX: x, actualY: y, scaleX, scaleY });
    
    setIsDrawing(true);
    setDrawStart({ x, y });
    setTempBbox({ x, y, width: 0, height: 0 });
  };

  const handleImageMouseMove = (e) => {
    if (!isDrawing || !drawStart || !imageRef.current) return;
    
    const img = imageRef.current;
    const rect = img.getBoundingClientRect();
    
    // Get current position relative to displayed image
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;
    
    // Convert to actual image coordinates
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    
    const currentX = Math.round(displayX * scaleX);
    const currentY = Math.round(displayY * scaleY);
    
    const x = Math.min(drawStart.x, currentX);
    const y = Math.min(drawStart.y, currentY);
    const width = Math.abs(currentX - drawStart.x);
    const height = Math.abs(currentY - drawStart.y);
    
    setTempBbox({ x, y, width, height });
    
    // Draw on canvas (in display coordinates)
    if (canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Convert back to display coordinates for drawing
      const displayBbox = {
        x: x / scaleX,
        y: y / scaleY,
        width: width / scaleX,
        height: height / scaleY
      };
      
      // Draw bbox with thin line
      ctx.strokeStyle = '#3B82F6';
      ctx.lineWidth = 1;
      ctx.strokeRect(displayBbox.x, displayBbox.y, displayBbox.width, displayBbox.height);
      
      // Fill with very light semi-transparent blue
      ctx.fillStyle = 'rgba(59, 130, 246, 0.05)';
      ctx.fillRect(displayBbox.x, displayBbox.y, displayBbox.width, displayBbox.height);
      
      // Draw corner handles (small circles)
      const handleSize = 6;
      ctx.fillStyle = '#3B82F6';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      
      const corners = [
        [displayBbox.x, displayBbox.y], // top-left
        [displayBbox.x + displayBbox.width, displayBbox.y], // top-right
        [displayBbox.x, displayBbox.y + displayBbox.height], // bottom-left
        [displayBbox.x + displayBbox.width, displayBbox.y + displayBbox.height] // bottom-right
      ];
      
      corners.forEach(([cx, cy]) => {
        ctx.beginPath();
        ctx.arc(cx, cy, handleSize/2, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });
    }
  };

  const handleImageMouseUp = () => {
    setIsDrawing(false);
    setDrawStart(null);
  };

  const redrawCanvas = () => {
    if (!canvasRef.current || !imageRef.current || !tempBbox) return;
    
    const img = imageRef.current;
    const rect = img.getBoundingClientRect();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Convert actual image coordinates to display coordinates
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    
    const displayBbox = {
      x: tempBbox.x / scaleX,
      y: tempBbox.y / scaleY,
      width: tempBbox.width / scaleX,
      height: tempBbox.height / scaleY
    };
    
    // Draw bbox with thin line
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 1;
    ctx.strokeRect(displayBbox.x, displayBbox.y, displayBbox.width, displayBbox.height);
    
    // Fill with very light semi-transparent blue
    ctx.fillStyle = 'rgba(59, 130, 246, 0.05)';
    ctx.fillRect(displayBbox.x, displayBbox.y, displayBbox.width, displayBbox.height);
    
    // Draw corner handles (small circles)
    const handleSize = 6;
    ctx.fillStyle = '#3B82F6';
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1;
    
    const corners = [
      [displayBbox.x, displayBbox.y], // top-left
      [displayBbox.x + displayBbox.width, displayBbox.y], // top-right
      [displayBbox.x, displayBbox.y + displayBbox.height], // bottom-left
      [displayBbox.x + displayBbox.width, displayBbox.y + displayBbox.height] // bottom-right
    ];
    
    corners.forEach(([cx, cy]) => {
      ctx.beginPath();
      ctx.arc(cx, cy, handleSize/2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    });
  };

  useEffect(() => {
    if (isDrawing) {
      window.addEventListener('mousemove', handleImageMouseMove);
      window.addEventListener('mouseup', handleImageMouseUp);
    } else {
      window.removeEventListener('mousemove', handleImageMouseMove);
      window.removeEventListener('mouseup', handleImageMouseUp);
    }
    
    return () => {
      window.removeEventListener('mousemove', handleImageMouseMove);
      window.removeEventListener('mouseup', handleImageMouseUp);
    };
  }, [isDrawing, drawStart]);

  // Redraw canvas when tempBbox changes (from manual input)
  useEffect(() => {
    if (tempBbox && !isDrawing) {
      redrawCanvas();
    }
  }, [tempBbox]);

  // Update canvas size when zoom changes
  useEffect(() => {
    if (imageRef.current && canvasRef.current) {
      const rect = imageRef.current.getBoundingClientRect();
      canvasRef.current.width = rect.width;
      canvasRef.current.height = rect.height;
      if (tempBbox && !isDrawing) {
        redrawCanvas();
      }
    }
  }, [imageZoom]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* PDF Viewer - Main Area */}
      <div className="flex-1 min-w-0 flex flex-col">
        <PDFViewer
          pdfUrl={pdfUrl}
          boundingBoxes={[]}
          scrollToPage={scrollToPage}
          isEditable={false}
          createMode={false}
        />
      </div>

      {/* Right Sidebar - Legend Items List */}
      <div className="w-80 flex-shrink-0 bg-white border-l overflow-y-auto">
        <div className="p-4 border-b bg-gray-50">
          <h2 className="text-lg font-semibold mb-2">Legend Items</h2>
          <p className="text-sm text-gray-600">
            Click an item to configure its icon and label templates
          </p>
          
          {/* Extract Legend Items Button */}
          {legendTables.length > 0 && (
            <button
              onClick={handleExtractLegendItems}
              disabled={isExtracting}
              className="mt-3 w-full py-2 px-4 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2 text-sm font-medium"
            >
              {isExtracting ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Extracting...
                </>
              ) : (
                <>
                  📋 Extract Legend Items
                </>
              )}
            </button>
          )}
          
          {/* Extraction Status Message */}
          {extractionMessage && (
            <div className={`mt-2 p-2 rounded text-sm ${
              extractionMessage.includes('✅') ? 'bg-green-100 text-green-700' :
              extractionMessage.includes('❌') ? 'bg-red-100 text-red-700' :
              'bg-blue-100 text-blue-700'
            }`}>
              {extractionMessage}
          </div>
          )}
        </div>

        <div className="p-4 space-y-2">
          {legendItems.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <p className="mb-4">No legend items extracted yet</p>
              {legendTables.length === 0 ? (
              <p className="text-sm text-gray-400">
                  Go to "Legend Tables" section to create tables first
              </p>
              ) : (
                <p className="text-sm text-gray-400">
                  Click "Extract Legend Items" above to extract items from {legendTables.length} table(s)
                </p>
              )}
            </div>
          ) : (
            legendItems.map((item, index) => (
              <div
                key={item.id}
                onClick={() => handleItemClick(item)}
                className={`
                  p-3 border rounded-lg cursor-pointer
                  transition-all hover:shadow-md
                  ${selectedItemId === item.id ? 'ring-2 ring-blue-500 bg-blue-50' : 'border-gray-300'}
                `}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{item.description}</div>
                    {item.label_text && (
                      <div className="text-xs text-gray-600 mt-1">
                        Label: {item.label_text}
                      </div>
                    )}
                  </div>
                  <div className="ml-2">
                    <span className="inline-block px-2 py-1 text-xs bg-gray-200 rounded">
                      {index + 1}
                    </span>
                  </div>
                </div>
                
              </div>
            ))
          )}
        </div>

        {/* Action Buttons */}
        {selectedItemId && (
          <div className="border-t bg-gray-50 p-4 space-y-2">
            <h3 className="font-semibold text-sm mb-3">Template Selection</h3>
            
            {/* Icon Template Status */}
            <div>
              <div className="text-xs font-medium text-gray-700 mb-1">Icon Template</div>
              {iconTemplate ? (
                <div className="text-xs text-green-600 p-2 bg-green-50 rounded flex items-center">
                  <span className="mr-2">✓</span>
                  <span>Icon saved</span>
                </div>
              ) : (
                <div className="text-xs text-gray-500 p-2 bg-gray-100 rounded">
                  Not selected yet
                </div>
              )}
            </div>
            
            {/* Label Templates Status - Supports Multiple Tags */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="text-xs font-medium text-gray-700">Label/Tag Templates</div>
                <span className="text-xs text-gray-500">({labelTemplates.length} tags)</span>
              </div>
              {labelTemplates.length > 0 ? (
                <div className="space-y-1">
                  {labelTemplates.map((template, index) => (
                    <div 
                      key={template.id} 
                      className="text-xs text-green-600 p-2 bg-green-50 rounded flex items-center justify-between group"
                    >
                      <div className="flex items-center">
                        <span className="mr-2">✓</span>
                        <span>{template.tag_name || `Tag ${index + 1}`}</span>
                      </div>
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSelectLabel(template.id);
                          }}
                          className="text-blue-600 hover:text-blue-800 p-1"
                          title="Edit tag"
                        >
                          ✏️
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteLabelTemplate(template.id);
                          }}
                          className="text-red-600 hover:text-red-800 p-1"
                          title="Delete tag"
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-gray-500 p-2 bg-gray-100 rounded">
                  No tags selected yet
                </div>
              )}
            </div>
            
            <div className="pt-2 space-y-2">
              {/* Select Icon Button */}
              <button
                onClick={handleSelectIcon}
                className="w-full py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 text-sm"
              >
                {iconTemplate ? '🔄 Update Icon' : '➕ Select Icon'}
              </button>
              
              {/* Add Label/Tag Button */}
              <button
                onClick={() => handleSelectLabel()}
                className="w-full py-2 px-4 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50 text-sm"
              >
                ➕ Add Label/Tag
              </button>
              
              {/* Instruction */}
              {selectedItemId && legendItems.find(item => item.id === selectedItemId) && (
                <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded mt-2">
                  💡 <strong>Tip:</strong> Click "Select Icon" or "Add Label/Tag" to draw a tight bounding box around the symbol in the legend table image.
                </div>
              )}
              
              <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded mt-3">
                ℹ️ <strong>Next Step:</strong> After configuring all templates, go to the "Processing" tab to run detection and matching.
              </div>
            </div>
          </div>
        )}

      </div>
      
      {/* Template Selector Modal */}
      {showTemplateSelector && selectedItemId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            <div className="p-4 border-b bg-gray-50">
              <h2 className="text-lg font-semibold">
                {showTemplateSelector === 'icon' ? '🎯 Select Icon Template' : '📝 Select Label/Tag Template'}
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Draw a tight bounding box around ONLY the {showTemplateSelector} (no whitespace)
              </p>
              {showTemplateSelector === 'label' && (
                <div className="mt-3">
                  <label className="text-sm font-medium text-gray-700">Tag Name (e.g., CF1, CF2)</label>
                  <input
                    type="text"
                    value={newTagName}
                    onChange={(e) => setNewTagName(e.target.value)}
                    placeholder="Enter tag name..."
                    className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-purple-500 focus:border-purple-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    💡 This helps identify different tags for the same icon (e.g., CF1, CF2, CF3)
                  </p>
                </div>
              )}
            </div>
            
            <div className="flex-1 overflow-auto p-6 bg-gray-100">
              <div className="bg-yellow-100 border-l-4 border-yellow-500 p-4 mb-4">
                <p className="font-semibold text-yellow-800">⚠️ IMPORTANT:</p>
                <ul className="text-sm text-yellow-700 mt-2 space-y-1 list-disc list-inside">
                  <li>Draw the box as <strong>tight as possible</strong> around the symbol</li>
                  <li>Exclude all whitespace and background</li>
                  <li>A loose/white template will match everything and cause detection to hang</li>
                  <li>Use the browser's developer tools to inspect the legend table image</li>
                </ul>
              </div>
              
              <div className="bg-white p-4 rounded shadow">
                <div className="text-center mb-2">
                  <p className="text-gray-600 font-medium">
                    👇 Click and drag to draw a bounding box:
                  </p>
                </div>
                {(() => {
                  const selectedItem = legendItems.find(item => item.id === selectedItemId);
                  const legendTable = legendTables.find(t => t.id === selectedItem?.legend_table_id);
                  if (legendTable?.cropped_image_url) {
                    return (
                      <div className="w-full">
                        {/* Zoom controls */}
                        <div className="flex items-center justify-center gap-2 mb-2">
                          <button
                            onClick={() => setImageZoom(Math.max(0.5, imageZoom - 0.25))}
                            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
                          >
                            🔍−
                          </button>
                          <span className="text-sm font-mono bg-gray-100 px-3 py-1 rounded">
                            {Math.round(imageZoom * 100)}%
                          </span>
                          <button
                            onClick={() => setImageZoom(Math.min(3, imageZoom + 0.25))}
                            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
                          >
                            🔍+
                          </button>
                          <button
                            onClick={() => setImageZoom(1.0)}
                            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                          >
                            Reset
                          </button>
                        </div>
                        
                        <div 
                          ref={imageContainerRef}
                          className="relative w-full overflow-auto border-2 border-gray-400 bg-gray-50"
                          style={{ maxHeight: '60vh' }}
                        >
                          <div style={{ 
                            transform: `scale(${imageZoom})`, 
                            transformOrigin: 'top left',
                            width: `${100 / imageZoom}%`
                          }}>
                            <img 
                              ref={imageRef}
                              src={api.getLegendTableImage(legendTable.id)} 
                              alt="Legend Table"
                              className="w-full cursor-crosshair select-none"
                              draggable={false}
                              onMouseDown={handleImageMouseDown}
                              onLoad={(e) => {
                                // Set canvas size to match displayed image exactly
                                if (canvasRef.current) {
                                  const rect = e.target.getBoundingClientRect();
                                  canvasRef.current.width = rect.width;
                                  canvasRef.current.height = rect.height;
                                  console.log('📏 Image loaded:', {
                                    naturalWidth: e.target.naturalWidth,
                                    naturalHeight: e.target.naturalHeight,
                                    displayWidth: rect.width,
                                    displayHeight: rect.height,
                                    scaleX: e.target.naturalWidth / rect.width,
                                    scaleY: e.target.naturalHeight / rect.height
                                  });
                                }
                              }}
                            />
                            <canvas
                              ref={canvasRef}
                              className="absolute top-0 left-0 pointer-events-none"
                              style={{ 
                                width: '100%', 
                                height: '100%',
                                imageRendering: 'crisp-edges'
                              }}
                            />
                          </div>
                        </div>
                        
                        {imageRef.current && (
                          <div className="text-xs text-center text-gray-500 mt-2 bg-gray-100 p-2 rounded">
                            📏 Original: {imageRef.current.naturalWidth} × {imageRef.current.naturalHeight} px
                            {' | '}
                            Displayed: {Math.round(imageRef.current.getBoundingClientRect().width)} × {Math.round(imageRef.current.getBoundingClientRect().height)} px
                            {' | '}
                            Scale: {(imageRef.current.naturalWidth / imageRef.current.getBoundingClientRect().width).toFixed(2)}x
                          </div>
                        )}
                      </div>
                    );
                  }
                  return <p className="text-gray-500">No legend table image available</p>;
                })()}
                
                {tempBbox && tempBbox.width > 0 && tempBbox.height > 0 && (
                  <div className="mt-3 space-y-3">
                    <div className="p-3 bg-blue-50 rounded border border-blue-200">
                      <div className="text-sm font-semibold text-blue-800 mb-2">
                        📐 Bounding Box Coordinates (in actual image pixels):
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        <div>
                          <label className="text-xs text-gray-600 block mb-1">X</label>
                          <input
                            type="number"
                            value={tempBbox.x}
                            onChange={(e) => {
                              const newX = parseInt(e.target.value) || 0;
                              setTempBbox(prev => ({ ...prev, x: newX }));
                            }}
                            className="w-full px-2 py-1 border rounded text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 block mb-1">Y</label>
                          <input
                            type="number"
                            value={tempBbox.y}
                            onChange={(e) => {
                              const newY = parseInt(e.target.value) || 0;
                              setTempBbox(prev => ({ ...prev, y: newY }));
                            }}
                            className="w-full px-2 py-1 border rounded text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 block mb-1">Width</label>
                          <input
                            type="number"
                            value={tempBbox.width}
                            onChange={(e) => {
                              const newWidth = parseInt(e.target.value) || 0;
                              setTempBbox(prev => ({ ...prev, width: newWidth }));
                            }}
                            className="w-full px-2 py-1 border rounded text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-xs text-gray-600 block mb-1">Height</label>
                          <input
                            type="number"
                            value={tempBbox.height}
                            onChange={(e) => {
                              const newHeight = parseInt(e.target.value) || 0;
                              setTempBbox(prev => ({ ...prev, height: newHeight }));
                            }}
                            className="w-full px-2 py-1 border rounded text-sm"
                          />
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        💡 You can manually adjust these values for precision
                      </div>
                    </div>
                    
                    {/* Preview of cropped area */}
                    {imageRef.current && (
                      <div className="p-3 bg-green-50 rounded border border-green-200">
                        <div className="text-sm font-semibold text-green-800 mb-2">
                          👁️ Preview (what will be saved):
                        </div>
                        <canvas
                          ref={(canvas) => {
                            if (canvas && imageRef.current) {
                              const img = imageRef.current;
                              const ctx = canvas.getContext('2d');
                              
                              // Set canvas to bbox size
                              canvas.width = tempBbox.width;
                              canvas.height = tempBbox.height;
                              
                              // Draw the cropped portion
                              ctx.drawImage(
                                img,
                                tempBbox.x, tempBbox.y, tempBbox.width, tempBbox.height, // source
                                0, 0, tempBbox.width, tempBbox.height // destination
                              );
                            }
                          }}
                          className="border-2 border-green-300 bg-white"
                          style={{ maxWidth: '200px', maxHeight: '200px', imageRendering: 'pixelated' }}
                        />
                        <div className="text-xs text-gray-600 mt-2">
                          Size: {tempBbox.width} × {tempBbox.height} px
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            <div className="p-4 border-t bg-gray-50 flex justify-between">
              <button
                onClick={() => {
                  setTempBbox(null);
                  if (canvasRef.current) {
                    const ctx = canvasRef.current.getContext('2d');
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                  }
                }}
                disabled={!tempBbox}
                className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50"
              >
                🔄 Clear
              </button>
              
              <div className="flex space-x-2">
                <button
                  onClick={() => {
                    setShowTemplateSelector(null);
                    setTempBbox(null);
                    if (canvasRef.current) {
                      const ctx = canvasRef.current.getContext('2d');
                      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                    }
                  }}
                  className="px-4 py-2 bg-gray-300 text-gray-800 rounded hover:bg-gray-400"
                >
                  Cancel
                </button>
                <button
                  onClick={showTemplateSelector === 'icon' ? handleSaveIconTemplate : handleSaveLabelTemplate}
                  disabled={!tempBbox || !tempBbox.width || !tempBbox.height}
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                >
                  💾 Save {showTemplateSelector === 'icon' ? 'Icon' : 'Label'} Template
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LegendItemsSection;

