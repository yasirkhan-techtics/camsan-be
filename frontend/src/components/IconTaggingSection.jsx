import React, { useEffect, useMemo, useState } from 'react';
import PDFViewer from './PDFViewer';
import AIAgentStatus from './AIAgentStatus';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'matched', label: 'Matched' },
  { key: 'unmatched', label: 'Unmatched' },
  { key: 'llm_matched', label: 'AI Matched' },
];

const IconTaggingSection = () => {
  const {
    selectedProject,
    iconDetections,
    labelDetections,
    matches,
    legendTables,
    fetchProjectDetections,
    fetchMatches,
    runDetectIcons,
    runDetectLabels,
    runMatchIconsLabels,
    // AI verification
    aiStatus,
    resetAIStatus,
    dismissAIStatus,
    runDetectIconsWithVerification,
    runDetectLabelsWithVerification,
    runMatchWithLLM,
    runFullPipelineWithLLM,
  } = useProject();

  const [actionLoading, setActionLoading] = useState({
    icons: false,
    labels: false,
    match: false,
    fullPipeline: false,
  });
  const [actionError, setActionError] = useState(null);
  const [selectedMatchId, setSelectedMatchId] = useState(null);
  const [filter, setFilter] = useState('all');
  const [scrollToPage, setScrollToPage] = useState(null);
  const [showIcons, setShowIcons] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [useAIVerification, setUseAIVerification] = useState(true);
  const [sidebarTab, setSidebarTab] = useState('legend'); // 'legend' or 'matches'
  const [selectedLegendItemId, setSelectedLegendItemId] = useState(null);
  const [showAllMatchesOnPdf, setShowAllMatchesOnPdf] = useState(false); // Render all matches on PDF at once
  const [visibleMatchIds, setVisibleMatchIds] = useState(new Set()); // Which individual matches to show on PDF
  const [scrollToBbox, setScrollToBbox] = useState(null); // For scrolling PDF to center on a bbox

  // Aggregate all legend items from all legend tables
  const legendItems = useMemo(() => {
    return legendTables.flatMap(table => table.legend_items || []);
  }, [legendTables]);

  // Get legend item for a detection (icon or label)
  const getDetectionLegendItemId = (detection) => {
    if (!detection) return null;
    // For icon detections, get legend_item_id via icon_template
    if (detection.icon_template_id) {
      // Find the icon template and get its legend_item_id
      for (const table of legendTables) {
        for (const item of (table.legend_items || [])) {
          if (item.icon_template?.id === detection.icon_template_id) {
            return item.id;
          }
        }
      }
    }
    // For label detections, get legend_item_id via label_template
    if (detection.label_template_id) {
      for (const table of legendTables) {
        for (const item of (table.legend_items || [])) {
          if (item.label_templates?.some(t => t.id === detection.label_template_id)) {
            return item.id;
          }
        }
      }
    }
    return null;
  };

  useEffect(() => {
    if (!selectedProject) return;
    fetchProjectDetections();
    fetchMatches();
  }, [selectedProject, fetchProjectDetections, fetchMatches]);

  // Initialize visible matches when matches change
  useEffect(() => {
    if (matches.length > 0) {
      setVisibleMatchIds(new Set(matches.map(m => m.id)));
    }
  }, [matches]);

  // Toggle visibility of a single match
  const toggleMatchVisibility = (matchId) => {
    setVisibleMatchIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(matchId)) {
        newSet.delete(matchId);
      } else {
        newSet.add(matchId);
      }
      return newSet;
    });
  };

  // Show all matches
  const showAllVisibleMatches = () => {
    setVisibleMatchIds(new Set(filteredMatches.map(m => m.id)));
  };

  // Hide all matches
  const hideAllVisibleMatches = () => {
    setVisibleMatchIds(new Set());
  };

  useEffect(() => {
    const match = matches.find((m) => m.id === selectedMatchId);
    if (!match) return;
    const icon = iconDetections.find((det) => det.id === match.icon_detection_id);
    if (icon?.page_number) {
      setScrollToPage(icon.page_number);
    }
  }, [matches, iconDetections, selectedMatchId]);

  const iconMap = useMemo(() => {
    const map = new Map();
    iconDetections.forEach((det) => map.set(det.id, det));
    return map;
  }, [iconDetections]);

  const labelMap = useMemo(() => {
    const map = new Map();
    labelDetections.forEach((det) => map.set(det.id, det));
    return map;
  }, [labelDetections]);

  const derivedMatches = useMemo(() => {
    return matches.map((match) => {
      const icon = iconMap.get(match.icon_detection_id);
      const label = match.label_detection_id
        ? labelMap.get(match.label_detection_id)
        : null;
      // For AI-matched icons, consider them matched if they have llm_assigned_label
      const isMatched = Boolean(label) || Boolean(match.llm_assigned_label);
      return {
        ...match,
        icon,
        label,
        isMatched,
        page_number: icon?.page_number || label?.page_number || 1,
      };
    });
  }, [matches, iconMap, labelMap]);

  const filteredMatches = useMemo(() => {
    let result = derivedMatches;
    
    // Apply status filter
    if (filter === 'matched') {
      return result.filter((m) => m.isMatched);
    }
    if (filter === 'unmatched') {
      return result.filter((m) => !m.isMatched);
    }
    if (filter === 'llm_matched') {
      return result.filter((m) => m.match_method === 'llm_matched');
    }
    return result;
  }, [derivedMatches, filter]);

  const selectedMatch = derivedMatches.find((m) => m.id === selectedMatchId);

  const pdfUrl = selectedProject ? api.getProjectPdf(selectedProject.id) : null;

  // Generate a color for each legend item (for distinguishing matches)
  const legendItemColors = useMemo(() => {
    const colors = [
      '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6', 
      '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#ec4899'
    ];
    const colorMap = new Map();
    legendItems.forEach((item, index) => {
      colorMap.set(item.id, colors[index % colors.length]);
    });
    return colorMap;
  }, [legendItems]);

  const boundingBoxes = useMemo(() => {
    if (!selectedProject) return [];
    const boxes = [];

    // Show all matches on PDF (filtered by visible match IDs)
    if (showAllMatchesOnPdf) {
      filteredMatches.forEach((match, index) => {
        // Skip if this match is not visible
        if (!visibleMatchIds.has(match.id)) {
          return;
        }
        
        // Get color based on legend item or use index-based color
        const iconLegendItemId = getDetectionLegendItemId(match.icon);
        const matchColor = iconLegendItemId 
          ? legendItemColors.get(iconLegendItemId) || '#22c55e'
          : `hsl(${(index * 37) % 360}, 70%, 50%)`;
        
        if (showIcons && match.icon) {
          // For AI-matched icons, show the llm_assigned_label on the box
          const iconLabel = match.match_method === 'llm_matched' && match.llm_assigned_label
            ? match.llm_assigned_label
            : null;
          boxes.push({
            id: `icon-${match.icon.id}`,
            bbox_normalized: match.icon.bbox_normalized,
            page_number: match.icon.page_number,
            color: selectedMatchId === match.id ? '#ff0000' : matchColor,
            confidence: match.icon.confidence,
            label: iconLabel, // Display LLM-assigned label on the box
          });
        }
        if (showLabels && match.label) {
          boxes.push({
            id: `label-${match.label.id}`,
            bbox_normalized: match.label.bbox_normalized,
            page_number: match.label.page_number,
            color: selectedMatchId === match.id ? '#ff0000' : matchColor,
            confidence: match.label.confidence,
          });
        }
      });
      return boxes;
    }

    // Show only selected match
    if (selectedMatch) {
      if (showIcons && selectedMatch.icon) {
        // For AI-matched icons, show the llm_assigned_label on the box
        const iconLabel = selectedMatch.match_method === 'llm_matched' && selectedMatch.llm_assigned_label
          ? selectedMatch.llm_assigned_label
          : null;
        boxes.push({
          id: `icon-${selectedMatch.icon.id}`,
          bbox_normalized: selectedMatch.icon.bbox_normalized,
          page_number: selectedMatch.icon.page_number,
          color: '#f97316',
          confidence: selectedMatch.icon.confidence,
          label: iconLabel, // Display LLM-assigned label on the box
        });
      }
      if (showLabels && selectedMatch.label) {
        boxes.push({
          id: `label-${selectedMatch.label.id}`,
          bbox_normalized: selectedMatch.label.bbox_normalized,
          page_number: selectedMatch.label.page_number,
          color: '#2563eb',
          confidence: selectedMatch.label.confidence,
        });
      }
      return boxes;
    }

    // Default: show all detections (not grouped by match)
    if (showIcons) {
      boxes.push(
        ...iconDetections.map((icon) => ({
          id: `icon-${icon.id}`,
          bbox_normalized: icon.bbox_normalized,
          page_number: icon.page_number,
          color: '#22c55e',
          confidence: icon.confidence,
        }))
      );
    }
    if (showLabels) {
      boxes.push(
        ...labelDetections.map((label) => ({
          id: `label-${label.id}`,
          bbox_normalized: label.bbox_normalized,
          page_number: label.page_number,
          color: '#2563eb',
          confidence: label.confidence,
        }))
      );
    }
    return boxes;
  }, [selectedProject, selectedMatch, selectedMatchId, showIcons, showLabels, showAllMatchesOnPdf, filteredMatches, iconDetections, labelDetections, legendItemColors, visibleMatchIds]);

  const handleAction = async (type) => {
    if (!selectedProject) return;
    setActionError(null);
    setActionLoading((prev) => ({ ...prev, [type]: true }));
    try {
      if (type === 'icons') {
        if (useAIVerification) {
          await runDetectIconsWithVerification();
        } else {
          await runDetectIcons();
        }
      } else if (type === 'labels') {
        if (useAIVerification) {
          await runDetectLabelsWithVerification();
        } else {
          await runDetectLabels();
        }
      } else if (type === 'match') {
        if (useAIVerification) {
          await runMatchWithLLM();
        } else {
          await runMatchIconsLabels();
        }
      } else if (type === 'fullPipeline') {
        await runFullPipelineWithLLM();
      }
    } catch (err) {
      setActionError(err.message || 'Action failed');
    } finally {
      setActionLoading((prev) => ({ ...prev, [type]: false }));
    }
  };

  if (!selectedProject) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Select a project to manage icon tagging.
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <div className="bg-white border-b px-4 py-3 space-y-3">
          {/* Action Buttons Row */}
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={() => handleAction('icons')}
              className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50 flex items-center gap-2"
              disabled={actionLoading.icons || aiStatus.isRunning}
            >
              {actionLoading.icons ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Detecting...
                </>
              ) : (
                'Detect Icons'
              )}
            </button>
            <button
              onClick={() => handleAction('labels')}
              className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50 flex items-center gap-2"
              disabled={actionLoading.labels || aiStatus.isRunning}
            >
              {actionLoading.labels ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Detecting...
                </>
              ) : (
                'Detect Tags'
              )}
            </button>
            <button
              onClick={() => handleAction('match')}
              className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50 flex items-center gap-2"
              disabled={actionLoading.match || aiStatus.isRunning}
            >
              {actionLoading.match ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Matching...
                </>
              ) : (
                'Match Icons & Tags'
              )}
            </button>
            
            <div className="h-6 w-px bg-gray-300 mx-1" />
            
            <button
              onClick={() => handleAction('fullPipeline')}
              className="px-4 py-2 rounded bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 flex items-center gap-2 shadow-sm"
              disabled={actionLoading.fullPipeline || aiStatus.isRunning}
            >
              {actionLoading.fullPipeline ? (
                <>
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Running...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Run Full AI Pipeline
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                fetchProjectDetections();
                fetchMatches();
              }}
              className="px-4 py-2 rounded border border-gray-300 hover:bg-gray-100"
              disabled={aiStatus.isRunning}
            >
              Refresh Data
            </button>

            {/* AI Verification Toggle */}
            <label className="flex items-center gap-2 ml-auto text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={useAIVerification}
                onChange={() => setUseAIVerification(!useAIVerification)}
                className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
              />
              <span className="flex items-center gap-1">
                <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                AI Verification
              </span>
            </label>
            
            {actionError && (
              <div className="text-sm text-red-600">{actionError}</div>
            )}
          </div>

          {/* AI Agent Status */}
          <AIAgentStatus
            isRunning={aiStatus.isRunning}
            currentStep={aiStatus.currentStep}
            totalSteps={aiStatus.totalSteps}
            stepDescription={aiStatus.stepDescription}
            results={aiStatus.results}
            error={aiStatus.error}
            onRetry={() => {
              resetAIStatus();
              // Retry the last action if possible
            }}
            onDismiss={dismissAIStatus}
          />
        </div>

        <div className="flex-1 flex overflow-hidden min-w-0">
          {/* PDF Viewer - Main Area */}
          <div className="flex-1 min-w-0 flex flex-col">
            {/* Detection visibility controls */}
            <div className="flex items-center gap-4 p-3 bg-white border-b text-sm text-gray-700 flex-wrap">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showIcons}
                  onChange={() => setShowIcons((prev) => !prev)}
                  className="w-4 h-4 text-green-600 rounded"
                />
                <span className="text-green-700">Show Icons ({iconDetections.length})</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={() => setShowLabels((prev) => !prev)}
                  className="w-4 h-4 text-blue-600 rounded"
                />
                <span className="text-blue-700">Show Tags ({labelDetections.length})</span>
              </label>
              
              <div className="h-4 w-px bg-gray-300" />
              
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showAllMatchesOnPdf}
                  onChange={() => setShowAllMatchesOnPdf((prev) => !prev)}
                  className="w-4 h-4 text-indigo-600 rounded"
                />
                <span className="text-indigo-700 font-medium">Show All Matches ({filteredMatches.length})</span>
              </label>
              
              <span className="text-xs text-gray-500 ml-auto">
                {showAllMatchesOnPdf
                  ? `Showing ${filteredMatches.length} matches (color-coded)`
                  : selectedMatch
                    ? 'Highlighting selected match'
                    : 'Showing all detections'}
              </span>
            </div>

            {/* PDF Viewer */}
            <PDFViewer
              pdfUrl={pdfUrl}
              boundingBoxes={boundingBoxes}
              selectedBoxId={
                selectedMatch
                  ? `icon-${selectedMatch.icon_detection_id}`
                  : null
              }
              scrollToPage={scrollToPage}
              scrollToBbox={scrollToBbox}
              isEditable={false}
              createMode={false}
            />
          </div>

          <div className="w-96 border-l bg-white flex flex-col">
            {/* Sidebar Tabs */}
            <div className="flex border-b">
              <button
                onClick={() => setSidebarTab('legend')}
                className={`flex-1 px-4 py-3 text-sm font-medium ${
                  sidebarTab === 'legend'
                    ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
              >
                Legend Items ({legendItems.length})
              </button>
              <button
                onClick={() => setSidebarTab('matches')}
                className={`flex-1 px-4 py-3 text-sm font-medium ${
                  sidebarTab === 'matches'
                    ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
              >
                Matches ({derivedMatches.length})
              </button>
            </div>

            {/* Legend Items Tab */}
            {sidebarTab === 'legend' && (
              <div className="flex-1 overflow-y-auto">
                {legendItems.length === 0 ? (
                  <div className="p-6 text-center text-gray-500">
                    <p className="mb-2">No legend items found.</p>
                    <p className="text-xs">Go to "Legend Tables" section to extract items first.</p>
                  </div>
                ) : (
                  <div className="p-2 space-y-2">
                    {legendItems.map((item, index) => {
                      const isSelected = selectedLegendItemId === item.id;
                      const hasIcon = item.icon_template || item.icon_bbox_status === 'saved';
                      const hasTags = item.label_templates && item.label_templates.length > 0;
                      const itemColor = legendItemColors.get(item.id) || '#6b7280';
                      
                      return (
                        <button
                          key={item.id}
                          onClick={() => setSelectedLegendItemId(isSelected ? null : item.id)}
                          className={`w-full text-left p-3 rounded-lg border transition-all ${
                            isSelected 
                              ? 'border-indigo-500 bg-indigo-50 ring-2 ring-indigo-200' 
                              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            {/* Color indicator */}
                            <div 
                              className="w-3 h-3 rounded-full mt-1 flex-shrink-0"
                              style={{ backgroundColor: itemColor }}
                              title={`Color for this legend item`}
                            />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between">
                                <div className="flex-1 min-w-0">
                                  <div className="font-medium text-sm text-gray-900 truncate">
                                    {item.description || `Item ${index + 1}`}
                                  </div>
                                  {item.label_text && (
                                    <div className="text-xs text-gray-500 mt-0.5">
                                      Label: {item.label_text}
                                    </div>
                                  )}
                                </div>
                                <span className="ml-2 px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                                  #{index + 1}
                                </span>
                              </div>
                              
                              {/* Status indicators */}
                              <div className="flex items-center gap-2 mt-2">
                                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs ${
                                  hasIcon ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                                }`}>
                                  {hasIcon ? '✓ Icon' : '○ No Icon'}
                                </span>
                                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs ${
                                  hasTags ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'
                                }`}>
                                  {hasTags ? `✓ ${item.label_templates.length} Tag(s)` : '○ No Tags'}
                                </span>
                              </div>
                              
                              {/* Tags list if available */}
                              {hasTags && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {item.label_templates.map((tpl, i) => (
                                    <span key={tpl.id || i} className="px-1.5 py-0.5 bg-blue-50 text-blue-600 text-xs rounded">
                                      {tpl.tag_name || `Tag ${i + 1}`}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
                
                {/* Info box */}
                <div className="p-3 m-2 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700">
                  <strong>💡 Tip:</strong> Select a legend item, then use the buttons above to detect icons/tags. 
                  Go to "Legend Items" section to draw icon and tag bounding boxes.
                </div>
              </div>
            )}

            {/* Matches Tab */}
            {sidebarTab === 'matches' && (
              <>
                <div className="p-4 border-b space-y-3">
                  {/* Status Filter */}
                  <div className="flex gap-2 flex-wrap">
                    {FILTERS.map((f) => (
                      <button
                        key={f.key}
                        onClick={() => setFilter(f.key)}
                        className={`px-3 py-1 rounded text-sm ${
                          filter === f.key
                            ? 'bg-primary text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>
                  
                  {/* Show/Hide All (only when Show All Matches on PDF is enabled) */}
                  {showAllMatchesOnPdf && filteredMatches.length > 0 && (
                    <div className="flex items-center justify-between pt-2 border-t">
                      <span className="text-xs text-gray-600">
                        Toggle visibility ({visibleMatchIds.size} visible)
                      </span>
                      <div className="flex gap-2">
                        <button
                          onClick={showAllVisibleMatches}
                          className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                        >
                          Show All
                        </button>
                        <button
                          onClick={hideAllVisibleMatches}
                          className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
                        >
                          Hide All
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex-1 overflow-y-auto">
                  {filteredMatches.length === 0 ? (
                    <div className="p-6 text-center text-gray-500">
                      <p className="mb-2">No matches to display.</p>
                      <p className="text-xs">Run "Match Icons & Tags" after detection.</p>
                    </div>
                  ) : (
                    filteredMatches.map((match) => {
                      const isSelected = selectedMatchId === match.id;
                      const isVisible = visibleMatchIds.has(match.id);
                      const isLLMMatched = match.match_method === 'llm_matched';
                      const iconLegendItemId = getDetectionLegendItemId(match.icon);
                      // Use legend item color if found, otherwise use index-based colorful fallback
                      const fallbackColors = [
                        '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6', 
                        '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#ec4899'
                      ];
                      const matchIndex = filteredMatches.indexOf(match);
                      const matchColor = iconLegendItemId 
                        ? legendItemColors.get(iconLegendItemId) || fallbackColors[matchIndex % fallbackColors.length]
                        : fallbackColors[matchIndex % fallbackColors.length];
                      
                      // Scroll to the match location
                      const scrollToMatch = () => {
                        const targetBbox = match.icon?.bbox_normalized || match.label?.bbox_normalized;
                        if (targetBbox) {
                          // Include timestamp to force re-trigger even for same bbox
                          setScrollToBbox({
                            bbox_normalized: targetBbox,
                            page_number: match.page_number,
                            _ts: Date.now()
                          });
                        }
                      };
                      
                      // When showAllMatchesOnPdf is enabled, clicking toggles visibility AND scrolls
                      // Otherwise, clicking selects the match and scrolls to it
                      const handleCardClick = () => {
                        if (showAllMatchesOnPdf) {
                          // Toggle visibility and scroll to the match
                          toggleMatchVisibility(match.id);
                          scrollToMatch();
                        } else {
                          if (isSelected) {
                            setSelectedMatchId(null);
                            setScrollToBbox(null);
                          } else {
                            setSelectedMatchId(match.id);
                            scrollToMatch();
                          }
                        }
                      };
                      
                      return (
                        <div
                          key={match.id}
                          onClick={handleCardClick}
                          className={`flex items-start gap-2 px-3 py-3 border-b cursor-pointer transition-colors ${
                            isSelected ? 'bg-blue-50 hover:bg-blue-100' : 'hover:bg-gray-50'
                          } ${!isVisible && showAllMatchesOnPdf ? 'opacity-40 bg-gray-50' : ''}`}
                          style={{
                            borderLeftWidth: '4px',
                            borderLeftColor: matchColor,
                          }}
                        >
                          {/* Visibility indicator (only when Show All Matches is enabled) */}
                          {showAllMatchesOnPdf && (
                            <div
                              className={`mt-1 w-5 h-5 rounded flex items-center justify-center flex-shrink-0 border-2 transition-colors`}
                              style={{ 
                                borderColor: matchColor,
                                backgroundColor: isVisible ? matchColor : 'white'
                              }}
                            >
                              {isVisible && (
                                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                </svg>
                              )}
                            </div>
                          )}
                          
                          {/* Match content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between text-sm">
                              <div className="flex items-center gap-2">
                                {/* Color indicator (only when not showing all matches - otherwise left border shows color) */}
                                {!showAllMatchesOnPdf && (
                                  <div 
                                    className="w-3 h-3 rounded-full flex-shrink-0"
                                    style={{ backgroundColor: matchColor }}
                                  />
                                )}
                                <span className="font-medium text-gray-800">
                                  Page {match.page_number}
                                </span>
                              </div>
                              <div className="flex items-center gap-1">
                                {isLLMMatched && (
                                  <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 flex items-center gap-1">
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                    </svg>
                                    AI
                                  </span>
                                )}
                                <span
                                  className={`text-xs px-2 py-0.5 rounded-full ${
                                    match.isMatched
                                      ? 'bg-green-100 text-green-700'
                                      : 'bg-yellow-100 text-yellow-700'
                                  }`}
                                >
                                  {match.isMatched ? 'Matched' : 'Unmatched'}
                                </span>
                              </div>
                            </div>
                            <div className="mt-1 text-xs text-gray-600 space-y-0.5">
                              <div className="flex items-center gap-1">
                                <span style={{ color: matchColor }}>●</span>
                                Icon: {match.icon_detection_id?.slice(0, 8)}...
                                {match.icon && (
                                  <span className={`ml-1 px-1.5 py-0.5 rounded text-[10px] ${
                                    match.icon.verification_status === 'verified' 
                                      ? 'bg-green-100 text-green-700'
                                      : match.icon.verification_status === 'rejected'
                                        ? 'bg-red-100 text-red-700'
                                        : 'bg-gray-100 text-gray-600'
                                  }`}>
                                    {match.icon.verification_status || 'pending'}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-1">
                                <span style={{ color: matchColor }}>●</span>
                                Tag: {match.label_detection_id?.slice(0, 8) ?? '—'}
                                {match.label && (
                                  <span className={`ml-1 px-1.5 py-0.5 rounded text-[10px] ${
                                    match.label.verification_status === 'verified' 
                                      ? 'bg-green-100 text-green-700'
                                      : match.label.verification_status === 'rejected'
                                        ? 'bg-red-100 text-red-700'
                                        : 'bg-gray-100 text-gray-600'
                                  }`}>
                                    {match.label.verification_status || 'pending'}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center justify-between">
                                <span>
                                  Distance: {match.distance ? match.distance.toFixed(1) + 'px' : '—'}
                                </span>
                                <span className="text-gray-400">
                                  {match.match_confidence ? `${(match.match_confidence * 100).toFixed(0)}% conf` : ''}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IconTaggingSection;


