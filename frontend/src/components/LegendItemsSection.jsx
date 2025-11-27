import React, { useState, useEffect, useRef } from 'react';
import PDFViewer from './PDFViewer';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const LegendItemsSection = () => {
  const {
    selectedProject,
    pdfPages,
    legendTables,
    detections,
    fetchDetections,
    updateDetection,
    deleteDetection
  } = useProject();
  
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [selectedDetectionId, setSelectedDetectionId] = useState(null);
  const [scrollToPage, setScrollToPage] = useState(null);
  const [isSelectingIcon, setIsSelectingIcon] = useState(false);
  const [isSelectingLabel, setIsSelectingLabel] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isSearchingAll, setIsSearchingAll] = useState(false);
  const [searchStatus, setSearchStatus] = useState(''); // Current step status message
  const [searchIcons, setSearchIcons] = useState(true);
  const [searchLabels, setSearchLabels] = useState(true);
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

  useEffect(() => {
    // Load detections when an item is selected
    if (selectedItemId) {
      fetchDetections(selectedItemId);
    }
  }, [selectedItemId, fetchDetections]);

  if (!selectedProject) {
    return <div className="p-8 text-center">No project selected</div>;
  }

  const pdfUrl = api.getProjectPdf(selectedProject.id);

  const pageMetadata = React.useMemo(() => {
    const map = new Map();
    (pdfPages || []).forEach((page) => {
      if (page?.id) {
        map.set(page.id, page);
      }
    });
    return map;
  }, [pdfPages]);

  const boundingBoxes = React.useMemo(() => {
    if (!detections?.length) return [];

    return detections
      .map((detection) => {
        if (
          !Array.isArray(detection.bbox_normalized) ||
          detection.bbox_normalized.length !== 4
        ) {
          return null;
        }

        const page = pageMetadata.get(detection.page_id);
        const isSelected = detection.id === selectedDetectionId;
        const baseColor =
          detection.type === 'label'
            ? '#2563EB'
            : '#22C55E';

        return {
          id: detection.id,
          bbox_normalized: detection.bbox_normalized,
          page_number: detection.page_number || page?.page_number || 1,
          color: isSelected ? '#EF4444' : baseColor,
          type: detection.type,
          confidence: detection.confidence,
        };
      })
      .filter(Boolean);
  }, [detections, selectedDetectionId, pageMetadata]);

  const handleItemClick = async (item) => {
    setSelectedItemId(item.id);
    setSelectedDetectionId(null);
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

  const handleBboxUpdate = async (detectionId, newBbox) => {
    const detection = detections.find(d => d.id === detectionId);
    if (detection) {
      try {
        await updateDetection(detection.type, detectionId, newBbox);
      } catch (error) {
        console.error('Error updating detection:', error);
      }
    }
  };

  const handleBboxDelete = async (detectionId) => {
    const detection = detections.find(d => d.id === detectionId);
    if (detection && window.confirm('Delete this detection?')) {
      try {
        await deleteDetection(detection.type, detectionId);
        if (selectedDetectionId === detectionId) {
          setSelectedDetectionId(null);
        }
      } catch (error) {
        console.error('Error deleting detection:', error);
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

  const handleSearchItem = async () => {
    if (!selectedItemId) {
      console.log('❌ No item selected');
      return;
    }
    
    if (!searchIcons && !searchLabels) {
      alert('Please select at least one option (Icons or Labels) to search.');
      return;
    }
    
    console.log('🔍 ========== STARTING SEARCH ==========');
    console.log('Legend Item ID:', selectedItemId);
    console.log('Search options:', { searchIcons, searchLabels });
    setIsSearching(true);
    setSearchStatus('Initializing search...');
    
    let iconCount = 0;
    let labelCount = 0;
    let verificationResults = { icons: null, labels: null };
    
    try {
      // Refresh templates from server to ensure we have latest state
      let currentIconTemplate = iconTemplate;
      let currentLabelTemplates = labelTemplates;
      
      try {
        const iconResponse = await api.getIconTemplate(selectedItemId);
        currentIconTemplate = iconResponse.data;
        setIconTemplate(currentIconTemplate);
      } catch (e) {
        currentIconTemplate = null;
        setIconTemplate(null);
      }
      
      try {
        const labelResponse = await api.getLabelTemplates(selectedItemId);
        currentLabelTemplates = labelResponse.data || [];
        setLabelTemplates(currentLabelTemplates);
      } catch (e) {
        currentLabelTemplates = [];
        setLabelTemplates([]);
      }
      
      // Search for icons if enabled
      if (searchIcons && currentIconTemplate) {
        // STEP 1: Preprocess icon template if needed
        if (!currentIconTemplate.preprocessed_icon_url) {
          setSearchStatus('Preprocessing icon template...');
          console.log('🔧 Preprocessing icon template...');
          const preprocessResponse = await api.preprocessIcon(selectedItemId);
          console.log('✅ Icon preprocessed:', preprocessResponse.data);
          currentIconTemplate = preprocessResponse.data;
          setIconTemplate(currentIconTemplate);
        } else {
          console.log('✅ Icon already preprocessed');
        }

        // STEP 2: Search for icons
        setSearchStatus('Detecting icons across all pages...');
        console.log('🔎 Detecting icons across all pages...');
        const iconDetectResponse = await api.detectIcons(selectedProject.id);
        iconCount = iconDetectResponse.data?.length || 0;
        console.log(`✅ Icon detection complete! Found ${iconCount} total icons`);
        
        // STEP 3: LLM Verification for icons
        if (iconCount > 0) {
          setSearchStatus(`Verifying ${iconCount} icons with AI (calculating thresholds)...`);
          console.log('🤖 Starting AI verification for icons...');
          try {
            const verifyResponse = await api.verifyIconDetections(selectedProject.id);
            verificationResults.icons = verifyResponse.data;
            console.log('✅ Icon verification complete:', verifyResponse.data);
          } catch (verifyError) {
            console.warn('⚠️ Icon verification failed:', verifyError.message);
          }
        }
      } else if (searchIcons && !currentIconTemplate) {
        console.log('⚠️ Icon search enabled but no icon template defined');
      }
      
      // Search for labels if enabled
      if (searchLabels && currentLabelTemplates.length > 0) {
        setSearchStatus(`Detecting labels across all pages (${currentLabelTemplates.length} tag templates)...`);
        console.log(`🔎 Detecting labels across all pages (${currentLabelTemplates.length} tag templates)...`);
        const labelDetectResponse = await api.detectLabels(selectedProject.id);
        labelCount = labelDetectResponse.data?.length || 0;
        console.log(`✅ Label detection complete! Found ${labelCount} total labels`);
        
        // STEP 4: LLM Verification for labels
        if (labelCount > 0) {
          setSearchStatus(`Verifying ${labelCount} labels with AI (calculating thresholds)...`);
          console.log('🤖 Starting AI verification for labels...');
          try {
            const verifyResponse = await api.verifyLabelDetections(selectedProject.id);
            verificationResults.labels = verifyResponse.data;
            console.log('✅ Label verification complete:', verifyResponse.data);
          } catch (verifyError) {
            console.warn('⚠️ Label verification failed:', verifyError.message);
          }
        }
      } else if (searchLabels && currentLabelTemplates.length === 0) {
        console.log('ℹ️ Label search enabled but no label templates defined');
      }

      // Refresh all detections
      setSearchStatus('Refreshing detections...');
      await fetchDetections(selectedItemId);

      // Build summary message
      const results = [];
      if (searchIcons) {
        let iconMsg = `Icons: ${iconCount} detected`;
        if (verificationResults.icons) {
          const v = verificationResults.icons;
          iconMsg += ` → ${v.auto_approved} auto-approved, ${v.llm_approved} AI-approved, ${v.llm_rejected} rejected`;
        }
        results.push(iconMsg);
      }
      if (searchLabels) {
        let labelMsg = `Labels: ${labelCount} detected`;
        if (verificationResults.labels) {
          const v = verificationResults.labels;
          labelMsg += ` → ${v.auto_approved} auto-approved, ${v.llm_approved} AI-approved, ${v.llm_rejected} rejected`;
        }
        results.push(labelMsg);
      }
      
      setSearchStatus('');
      alert(`Search & Verification Complete!\n\n${results.join('\n')}\n\nCheck the PDF for highlighted detections.`);

      console.log('✅ ========== SEARCH COMPLETED ==========');
    } catch (error) {
      console.error('❌ ========== SEARCH FAILED ==========');
      console.error('Error:', error);
      console.error('Error details:', error.response?.data);
      setSearchStatus('');
      alert('Error searching: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsSearching(false);
      setSearchStatus('');
    }
  };

  const handleSearchAll = async () => {
    if (!selectedProject?.id) {
      console.log('❌ No project selected');
      return;
    }
    
    // Check if any legend item has an icon template
    const itemsWithIconTemplate = [];
    for (const item of legendItems) {
      try {
        const response = await api.getIconTemplate(item.id);
        if (response.data) {
          itemsWithIconTemplate.push(item);
        }
      } catch (error) {
        // No icon template for this item
      }
    }
    
    if (itemsWithIconTemplate.length === 0) {
      alert('No legend items have icon templates. Please select icons for at least one legend item first.');
      return;
    }
    
    const confirmMsg = `This will search for icons and labels across all pages for ${itemsWithIconTemplate.length} legend item(s) with templates.\n\nThis includes AI verification of results. Continue?`;
    if (!window.confirm(confirmMsg)) {
      return;
    }
    
    console.log('🔍 ========== STARTING SEARCH ALL ==========');
    console.log(`Project ID: ${selectedProject.id}`);
    console.log(`Items with templates: ${itemsWithIconTemplate.length}`);
    setIsSearchingAll(true);
    setSearchStatus('Starting full search...');
    
    let iconCount = 0;
    let labelCount = 0;
    let verificationResults = { icons: null, labels: null };
    
    try {
      // STEP 1: Preprocess all icon templates
      setSearchStatus('Step 1/6: Preprocessing icon templates...');
      console.log('🔧 STEP 1: Preprocessing all icon templates...');
      for (const item of itemsWithIconTemplate) {
        try {
          const templateResponse = await api.getIconTemplate(item.id);
          if (templateResponse.data && !templateResponse.data.preprocessed_icon_url) {
            console.log(`   Preprocessing icon for: ${item.description}`);
            await api.preprocessIcon(item.id);
          }
        } catch (error) {
          console.warn(`   Warning: Could not preprocess icon for ${item.description}:`, error.message);
        }
      }
      console.log('✅ All icon templates preprocessed');

      // STEP 2: Detect all icons
      setSearchStatus('Step 2/6: Detecting icons across all pages...');
      console.log('🔎 STEP 2: Detecting all icons across all pages...');
      const iconDetectResponse = await api.detectIcons(selectedProject.id);
      iconCount = iconDetectResponse.data?.length || 0;
      console.log(`✅ Icon detection complete! Found ${iconCount} total icons`);

      // STEP 3: AI Verification for icons
      if (iconCount > 0) {
        setSearchStatus(`Step 3/6: AI verifying ${iconCount} icons...`);
        console.log('🤖 STEP 3: AI verification for icons...');
        try {
          const verifyResponse = await api.verifyIconDetections(selectedProject.id);
          verificationResults.icons = verifyResponse.data;
          console.log('✅ Icon verification complete:', verifyResponse.data);
        } catch (verifyError) {
          console.warn('⚠️ Icon verification failed:', verifyError.message);
        }
      }

      // STEP 4: Detect all labels
      setSearchStatus('Step 4/6: Detecting labels across all pages...');
      console.log('🔎 STEP 4: Detecting all labels across all pages...');
      try {
        const labelDetectResponse = await api.detectLabels(selectedProject.id);
        labelCount = labelDetectResponse.data?.length || 0;
        console.log(`✅ Label detection complete! Found ${labelCount} total labels`);
      } catch (error) {
        console.warn('⚠️ Label detection skipped or failed:', error.message);
      }

      // STEP 5: AI Verification for labels
      if (labelCount > 0) {
        setSearchStatus(`Step 5/6: AI verifying ${labelCount} labels...`);
        console.log('🤖 STEP 5: AI verification for labels...');
        try {
          const verifyResponse = await api.verifyLabelDetections(selectedProject.id);
          verificationResults.labels = verifyResponse.data;
          console.log('✅ Label verification complete:', verifyResponse.data);
        } catch (verifyError) {
          console.warn('⚠️ Label verification failed:', verifyError.message);
        }
      }

      // STEP 6: Refresh detections
      setSearchStatus('Step 6/6: Refreshing results...');
      if (selectedItemId) {
        await fetchDetections(selectedItemId);
      }

      // Build summary
      const results = [];
      let iconMsg = `Icons: ${iconCount} detected`;
      if (verificationResults.icons) {
        const v = verificationResults.icons;
        iconMsg += `\n  → ${v.auto_approved} auto-approved (high confidence)`;
        iconMsg += `\n  → ${v.llm_approved} AI-approved (verified by LLM)`;
        iconMsg += `\n  → ${v.llm_rejected} rejected`;
      }
      results.push(iconMsg);
      
      let labelMsg = `Labels: ${labelCount} detected`;
      if (verificationResults.labels) {
        const v = verificationResults.labels;
        labelMsg += `\n  → ${v.auto_approved} auto-approved (high confidence)`;
        labelMsg += `\n  → ${v.llm_approved} AI-approved (verified by LLM)`;
        labelMsg += `\n  → ${v.llm_rejected} rejected`;
      }
      results.push(labelMsg);

      console.log('✅ ========== SEARCH ALL COMPLETED ==========');
      setSearchStatus('');
      alert(`Search & AI Verification Complete!\n\n${results.join('\n\n')}\n\nSelect a legend item to view its detections.`);
    } catch (error) {
      console.error('❌ ========== SEARCH ALL FAILED ==========');
      console.error('Error:', error);
      console.error('Error details:', error.response?.data);
      setSearchStatus('');
      alert('Error searching: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsSearchingAll(false);
      setSearchStatus('');
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* PDF Viewer - Main Area */}
      <div className="flex-1 min-w-0 flex flex-col">
        
        {(isSearching || isSearchingAll) && (
          <div className="p-3 bg-gradient-to-r from-indigo-100 to-purple-100 border-b border-indigo-200">
            <div className="flex items-center justify-center space-x-3">
              <div className="relative">
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-indigo-300 border-t-indigo-600"></div>
              </div>
              <div className="text-sm font-medium text-indigo-800">
                {searchStatus || (isSearchingAll ? '🔍 Searching all legend items...' : '🔍 Searching for icons and labels...')}
              </div>
            </div>
            {searchStatus && searchStatus.includes('AI') && (
              <div className="mt-2 text-xs text-center text-indigo-600">
                🤖 AI is calculating confidence thresholds and verifying low-confidence detections...
              </div>
            )}
          </div>
        )}
        
        <PDFViewer
          pdfUrl={pdfUrl}
          boundingBoxes={boundingBoxes}
          selectedBoxId={selectedDetectionId}
          onBboxUpdate={handleBboxUpdate}
          onBboxDelete={handleBboxDelete}
          onBboxSelect={setSelectedDetectionId}
          scrollToPage={scrollToPage}
          isEditable={true}
          createMode={false}
        />
      </div>

      {/* Right Sidebar - Legend Items List */}
      <div className="w-80 flex-shrink-0 bg-white border-l overflow-y-auto">
        <div className="p-4 border-b bg-gray-50">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-lg font-semibold">Legend Items</h2>
            <button
              onClick={handleSearchAll}
              disabled={isSearchingAll || legendItems.length === 0}
              className="px-3 py-1 bg-orange-500 text-white text-xs rounded hover:bg-orange-600 disabled:opacity-50 font-medium"
              title="Search for all icons and labels at once"
            >
              {isSearchingAll ? '⏳ Searching...' : '🔍 Search All'}
            </button>
          </div>
          <p className="text-sm text-gray-600">
            Click an item to view its detections
          </p>
        </div>

        <div className="p-4 space-y-2">
          {legendItems.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <p className="mb-4">No legend items extracted yet</p>
              <p className="text-sm text-gray-400">
                Go to "Legend Tables" section to extract items
              </p>
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
                
                {selectedItemId === item.id && detections.length > 0 && (
                  <div className="mt-2 pt-2 border-t text-xs text-gray-600">
                    {detections.length} detection(s) found
                  </div>
                )}
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
                  💡 <strong>Tip:</strong> Click "Select Icon" or "Select Label" to draw a tight bounding box around the symbol in the legend table image.
                </div>
              )}
              
              {/* Search Options Checkboxes */}
              <div className="pt-3 pb-2 border-t mt-3">
                <div className="text-xs font-medium text-gray-700 mb-2">Search Options</div>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={searchIcons}
                      onChange={(e) => setSearchIcons(e.target.checked)}
                      className="w-4 h-4 text-blue-500 rounded border-gray-300 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">🔷 Icons</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={searchLabels}
                      onChange={(e) => setSearchLabels(e.target.checked)}
                      className="w-4 h-4 text-purple-500 rounded border-gray-300 focus:ring-purple-500"
                    />
                    <span className="text-sm text-gray-700">📝 Labels/Tags</span>
                  </label>
                </div>
              </div>
              
              {/* Search Button */}
              <button
                onClick={handleSearchItem}
                disabled={(!iconTemplate && labelTemplates.length === 0) || isSearching || (!searchIcons && !searchLabels)}
                className="w-full py-2 px-4 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50 text-sm font-semibold"
              >
                {isSearching ? '🔍 Searching...' : '🔍 Search & Detect'}
              </button>
              
              <div className="text-xs text-gray-500 italic pt-1">
                * At least one template and option required.
              </div>
              <div className="text-xs text-amber-600 bg-amber-50 p-2 rounded mt-2">
                ⚠️ <strong>Note:</strong> Detection uses project-wide settings. If you get too many false matches, increase the threshold in Detection Settings.
              </div>
            </div>
          </div>
        )}

        {/* Detection Details */}
        {selectedItemId && detections.length > 0 && (
          <div className="border-t bg-gray-50 p-4">
            <h3 className="font-semibold text-sm mb-2">
              Detections ({detections.length})
            </h3>
            <div className="space-y-2">
              {detections.map((detection, index) => (
                <div
                  key={detection.id}
                  onClick={() => setSelectedDetectionId(detection.id)}
                  className={`
                    p-2 border rounded cursor-pointer text-xs
                    ${selectedDetectionId === detection.id ? 'bg-red-100 border-red-500' : 'bg-white border-gray-300'}
                  `}
                >
                  <div className="flex justify-between">
                    <span className="font-medium">
                      {detection.type === 'icon' ? '🔷 Icon' : '📝 Label'} #{index + 1}
                    </span>
                    <span className="text-gray-500">
                      Page {selectedProject.pages?.find(p => p.id === detection.page_id)?.page_number || '?'}
                    </span>
                  </div>
                  {detection.confidence && (
                    <div className="text-gray-600 mt-1">
                      Confidence: {(detection.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
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

