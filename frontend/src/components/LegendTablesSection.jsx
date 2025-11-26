import React, { useState, useEffect, useMemo } from 'react';
import PDFViewer from './PDFViewer';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const LegendTablesSection = () => {
  const { selectedProject, legendTables: contextLegendTables, updateLegendBBox, selectProject } = useProject();
  const [selectedTableId, setSelectedTableId] = useState(null);
  const [scrollToPage, setScrollToPage] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [localLegendTables, setLocalLegendTables] = useState([]);
  const [isDrawMode, setIsDrawMode] = useState(false);
  const [newDrawnBox, setNewDrawnBox] = useState(null);
  const [showExtractDialog, setShowExtractDialog] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [pendingBbox, setPendingBbox] = useState(null); // Temporary bbox being edited

  // Initialize local state from context on mount and when detecting new tables
  useEffect(() => {
    setLocalLegendTables(contextLegendTables);
  }, [contextLegendTables.length]); // Only when count changes, not on every update

  const legendTables = localLegendTables;

  if (!selectedProject) {
    return <div className="p-8 text-center">No project selected</div>;
  }

  const pdfUrl = api.getProjectPdf(selectedProject.id);

  // Convert legend tables to bounding box format for PDFViewer
  const boundingBoxes = useMemo(() => {
    console.log('Recalculating boundingBoxes from legendTables:', legendTables);
    const boxes = legendTables.map(table => {
      const page = selectedProject.pages?.find(p => p.id === table.page_id);
      return {
        id: table.id,
        bbox_normalized: table.bbox_normalized,
        page_number: page?.page_number || 1,
        color: selectedTableId === table.id ? 'red' : 'green'
      };
    });
    
    // Add pending bbox if exists
    if (pendingBbox) {
      boxes.push({
        id: 'pending',
        bbox_normalized: pendingBbox.bbox_normalized,
        page_number: pendingBbox.page_number,
        color: 'blue'
      });
    }
    
    return boxes;
  }, [legendTables, selectedTableId, selectedProject.pages, pendingBbox]);

  const handleTableClick = (table) => {
    setSelectedTableId(table.id);
    // Scroll to the page where this legend table is located
    // We need to find the page number from the pages array
    const page = selectedProject.pages?.find(p => p.id === table.page_id);
    if (page) {
      setScrollToPage(page.page_number);
    }
  };

  const handleBboxUpdate = (tableId, newBbox) => {
    console.log('handleBboxUpdate called:', { tableId, newBbox });
    console.log('newBbox.bbox_normalized:', newBbox.bbox_normalized);
    // Update local state immediately for responsive UI
    setLocalLegendTables(prevTables => {
      const updated = prevTables.map(table => {
        if (table.id === tableId) {
          console.log('Updating table:', table.id, 'from', table.bbox_normalized, 'to', newBbox.bbox_normalized);
          return { ...table, bbox_normalized: newBbox.bbox_normalized };
        }
        return table;
      });
      console.log('Updated local tables:', updated);
      console.log('First table bbox after update:', updated[0]?.bbox_normalized);
      return updated;
    });
  };

  const handleBboxDelete = async (tableId) => {
    if (!window.confirm('Delete this legend table? This will also delete all its legend items.')) {
      return;
    }
    
    try {
      await api.deleteLegendTable(selectedProject.id, tableId);
      // Remove from local state
      setLocalLegendTables(prevTables => prevTables.filter(t => t.id !== tableId));
      if (selectedTableId === tableId) {
        setSelectedTableId(null);
      }
      alert('Legend table deleted successfully!');
    } catch (error) {
      console.error('Error deleting legend table:', error);
      alert('Error deleting legend table: ' + (error.response?.data?.detail || error.message));
    }
  };

  const handleBboxCreate = (newBbox) => {
    console.log('New bbox created:', newBbox);
    setPendingBbox(newBbox);
    setIsDrawMode(false);
    // Don't show extract dialog yet, let user adjust bbox first
  };

  const handlePendingBboxUpdate = (boxId, newBbox) => {
    setPendingBbox(prev => ({
      ...prev,
      bbox_normalized: newBbox.bbox_normalized
    }));
  };

  const handleConfirmBbox = () => {
    setNewDrawnBox(pendingBbox);
    setPendingBbox(null);
    setShowExtractDialog(true);
  };

  const handleCancelPendingBbox = () => {
    setPendingBbox(null);
  };

  const handleExtractItems = async () => {
    setIsExtracting(true);
    setShowExtractDialog(false);
    
    try {
      // First create the legend table in backend
      const createResponse = await api.createLegendTable(selectedProject.id, {
        bbox_normalized: newDrawnBox.bbox_normalized,
        page_number: newDrawnBox.page_number
      });
      
      const newTable = createResponse.data;
      
      // Then extract legend items
      await api.extractLegendItems(selectedProject.id, newTable.id);
      
      // Refresh project to get the new table and items
      await selectProject(selectedProject.id);
      
      alert('Legend table created and items extracted successfully!');
    } catch (error) {
      console.error('Error creating legend table:', error);
      alert('Error: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsExtracting(false);
      setNewDrawnBox(null);
    }
  };

  const handleSkipExtraction = async () => {
    setShowExtractDialog(false);
    
    try {
      // Just create the legend table without extracting items
      const createResponse = await api.createLegendTable(selectedProject.id, {
        bbox_normalized: newDrawnBox.bbox_normalized,
        page_number: newDrawnBox.page_number
      });
      
      // Add to local state
      setLocalLegendTables(prev => [...prev, createResponse.data]);
      
      alert('Legend table created. You can extract items later.');
    } catch (error) {
      console.error('Error creating legend table:', error);
      alert('Error: ' + (error.response?.data?.detail || error.message));
    } finally {
      setNewDrawnBox(null);
    }
  };

  const handleSave = async () => {
    if (!selectedTableId) return;
    
    try {
      setIsSaving(true);
      const selectedTable = legendTables.find(t => t.id === selectedTableId);
      if (selectedTable) {
        // Call API directly to avoid full project refresh
        const response = await api.updateLegendBBox(
          selectedProject.id, 
          selectedTableId, 
          { bbox_normalized: selectedTable.bbox_normalized }
        );
        // Update local state with the response
        setLocalLegendTables(prevTables =>
          prevTables.map(t => t.id === selectedTableId ? response.data : t)
        );
        alert('Legend table bounding box updated successfully!');
      }
    } catch (error) {
      console.error('Error saving bbox:', error);
      alert('Error saving: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="flex h-full overflow-hidden">
      {/* PDF Viewer - Main Area */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Draw Mode Controls */}
        {!isDrawMode && (
          <div className="p-2 bg-gray-100 border-b flex items-center justify-between">
            <div className="text-sm text-gray-600">
              {legendTables.length} legend table(s)
            </div>
            <button
              onClick={() => setIsDrawMode(true)}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Draw New Legend Table
            </button>
          </div>
        )}
        
        {isDrawMode && (
          <div className="p-2 bg-blue-100 border-b flex items-center justify-between">
            <div className="text-sm font-medium text-blue-800">
              🖱️ Draw Mode: Click and drag on the PDF to draw a bounding box
            </div>
            <button
              onClick={() => setIsDrawMode(false)}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        )}
        
        {pendingBbox && !isDrawMode && (
          <div className="p-2 bg-yellow-100 border-b flex items-center justify-between">
            <div className="text-sm font-medium text-yellow-800">
              ✏️ Adjust the blue bounding box if needed, then confirm
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleCancelPendingBbox}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmBbox}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Confirm
              </button>
            </div>
          </div>
        )}
        
        <PDFViewer
          pdfUrl={pdfUrl}
          boundingBoxes={boundingBoxes}
          selectedBoxId={pendingBbox ? 'pending' : selectedTableId}
          onBboxUpdate={pendingBbox ? handlePendingBboxUpdate : handleBboxUpdate}
          onBboxSelect={setSelectedTableId}
          onBboxCreate={handleBboxCreate}
          scrollToPage={scrollToPage}
          isEditable={true}
          createMode={isDrawMode}
          onBboxDelete={pendingBbox ? null : handleBboxDelete}
        />
      </div>

      {/* Right Sidebar - Legend Table Screenshots */}
      <div className="w-80 flex-shrink-0 bg-white border-l overflow-y-auto flex flex-col">
        <div className="p-4 border-b bg-gray-50 flex-shrink-0">
          <h2 className="text-lg font-semibold">Legend Tables</h2>
          <p className="text-sm text-gray-600 mt-1">
            Click a table to view and edit its location
          </p>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-4">
            {legendTables.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <p className="mb-4">No legend tables detected yet</p>
                <button
                  onClick={async () => {
                    try {
                      setIsDetecting(true);
                      console.log('Detecting legends for project:', selectedProject.id);
                      const response = await api.detectLegends(selectedProject.id);
                      console.log('Detection response:', response.data);
                      // Refresh project data
                      await selectProject(selectedProject.id);
                      alert(`Successfully detected ${response.data.length} legend table(s)!`);
                    } catch (error) {
                      console.error('Error detecting legends:', error);
                      alert('Error detecting legends: ' + (error.response?.data?.detail || error.message));
                    } finally {
                      setIsDetecting(false);
                    }
                  }}
                  disabled={isDetecting}
                  className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50"
                >
                  {isDetecting ? 'Detecting...' : 'Detect Legend Tables'}
                </button>
              </div>
            ) : (
            legendTables.map((table, index) => (
              <div
                key={table.id}
                onClick={() => handleTableClick(table)}
                className={`
                  border rounded-lg overflow-hidden cursor-pointer
                  transition-all hover:shadow-lg
                  ${selectedTableId === table.id ? 'ring-2 ring-red-500' : 'border-gray-300'}
                `}
              >
                <div className="p-2 bg-gray-100 border-b">
                  <div className="font-medium">Table {index + 1}</div>
                  <div className="text-xs text-gray-500">
                    Page {selectedProject.pages?.find(p => p.id === table.page_id)?.page_number || '?'}
                  </div>
                </div>
                <div className="p-2">
                  {table.cropped_image_url ? (
                    <img
                      src={api.getLegendTableImage(table.id)}
                      alt={`Legend Table ${index + 1}`}
                      className="w-full h-auto"
                    />
                  ) : (
                    <div className="h-32 bg-gray-200 flex items-center justify-center text-gray-500">
                      No preview
                    </div>
                  )}
                </div>
              </div>
            ))
            )}
          </div>
        </div>

        {/* Action Buttons */}
        {selectedTableId && (
          <div className="flex-shrink-0 p-4 bg-white border-t">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="w-full py-2 px-4 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        )}
      </div>

      {/* Extract Items Dialog */}
      {showExtractDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Legend Table Created</h3>
            <p className="text-gray-600 mb-6">
              Would you like to extract legend items from this table now?
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleSkipExtraction}
                disabled={isExtracting}
                className="flex-1 px-4 py-2 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
              >
                Skip for Now
              </button>
              <button
                onClick={handleExtractItems}
                disabled={isExtracting}
                className="flex-1 px-4 py-2 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50"
              >
                {isExtracting ? 'Extracting...' : 'Extract Legend Items'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LegendTablesSection;

