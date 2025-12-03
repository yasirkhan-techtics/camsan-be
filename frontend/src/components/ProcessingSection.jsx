import React, { useEffect, useMemo, useState, useCallback } from 'react';
import PDFViewer from './PDFViewer';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

// Processing stages with their configurations
const STAGES = [
  { 
    id: 'detection', 
    name: 'Raw Detection', 
    description: 'Detect icons and labels across all pages',
    icon: '🔍'
  },
  { 
    id: 'overlap', 
    name: 'Overlap Removal', 
    description: 'Remove overlapping tag detections',
    icon: '🔄'
  },
  { 
    id: 'verification', 
    name: 'LLM Verification', 
    description: 'AI verifies detected icons and labels',
    icon: '🤖'
  },
  { 
    id: 'matching', 
    name: 'Basic Matching', 
    description: 'Match icons with nearby tags by distance',
    icon: '🔗'
  },
  { 
    id: 'tag-matching', 
    name: 'Tag Matching', 
    description: 'Find tags for unlabeled icons using AI',
    icon: '🏷️'
  },
  { 
    id: 'icon-matching', 
    name: 'Icon Matching', 
    description: 'Find icons for unlabeled tags using AI',
    icon: '🎯'
  },
];

const VERIFICATION_FILTERS = [
  { key: 'all', label: 'All', color: 'gray' },
  { key: 'verified', label: 'Verified', color: 'green' },
  { key: 'rejected', label: 'Rejected', color: 'red' },
  { key: 'pending', label: 'Pending', color: 'yellow' },
];

const OVERLAP_FILTERS = [
  { key: 'all', label: 'Show All', color: 'gray' },
  { key: 'removed', label: 'Show Removed', color: 'red' },
  { key: 'kept', label: 'After Overlap Removed', color: 'green' },
];

const MATCH_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'matched', label: 'Matched' },
  { key: 'unmatched', label: 'Unmatched' },
  { key: 'llm_tag_for_icon', label: 'AI Tag→Icon' },
  { key: 'llm_icon_for_tag', label: 'AI Icon→Tag' },
];

const ProcessingSection = () => {
  const {
    selectedProject,
    pdfPages,
    legendTables,
    iconDetections,
    labelDetections,
    matches,
    fetchProjectDetections,
    fetchMatches,
  } = useProject();

  // Stage management
  const [currentStage, setCurrentStage] = useState(0);
  const [completedStages, setCompletedStages] = useState(new Set());
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMessage, setProcessingMessage] = useState('');
  const [stageResults, setStageResults] = useState({});
  const [error, setError] = useState(null);
  
  // View state
  const [showIcons, setShowIcons] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [verificationFilter, setVerificationFilter] = useState('all');
  const [overlapFilter, setOverlapFilter] = useState('all');
  const [matchFilter, setMatchFilter] = useState('all');
  const [scrollToPage, setScrollToPage] = useState(null);
  const [selectedDetectionId, setSelectedDetectionId] = useState(null);
  
  // Detection options
  const [searchSelectedOnly, setSearchSelectedOnly] = useState(false);
  const [selectedLegendItemIds, setSelectedLegendItemIds] = useState(new Set());

  // Aggregate legend items
  const legendItems = useMemo(() => {
    const tables = legendTables || [];
    return tables.flatMap(table => table.legend_items || []);
  }, [legendTables]);

  // Refresh data on project change
  useEffect(() => {
    if (selectedProject) {
      fetchProjectDetections();
      fetchMatches();
    }
  }, [selectedProject]);

  // Filter detections based on current stage and filter
  const filteredIconDetections = useMemo(() => {
    const icons = iconDetections || [];
    // Stage 0 (Raw Detection): Show ALL detections, no filtering
    if (currentStage === 0) {
      return icons;
    }
    // Stage 1 (Overlap Removal): Use overlap filter
    if (currentStage === 1) {
      if (overlapFilter === 'all') return icons;
      if (overlapFilter === 'removed') return icons.filter(d => d.verification_status === 'rejected');
      if (overlapFilter === 'kept') return icons.filter(d => d.verification_status !== 'rejected');
    }
    // Stage 2 (Verification) and others: Use verification filter
    if (verificationFilter === 'all') return icons;
    return icons.filter(d => d.verification_status === verificationFilter);
  }, [iconDetections, verificationFilter, overlapFilter, currentStage]);

  const filteredLabelDetections = useMemo(() => {
    const labels = labelDetections || [];
    // Stage 0 (Raw Detection): Show ALL detections, no filtering
    if (currentStage === 0) {
      return labels;
    }
    // Stage 1 (Overlap Removal): Use overlap filter
    if (currentStage === 1) {
      if (overlapFilter === 'all') return labels;
      if (overlapFilter === 'removed') return labels.filter(d => d.verification_status === 'rejected');
      if (overlapFilter === 'kept') return labels.filter(d => d.verification_status !== 'rejected');
    }
    // Stage 2 (Verification) and others: Use verification filter
    if (verificationFilter === 'all') return labels;
    return labels.filter(d => d.verification_status === verificationFilter);
  }, [labelDetections, verificationFilter, overlapFilter, currentStage]);

  // Filter matches
  const filteredMatches = useMemo(() => {
    const matchList = matches || [];
    if (matchFilter === 'all') return matchList;
    if (matchFilter === 'matched') return matchList.filter(m => m.match_status === 'matched');
    if (matchFilter === 'unmatched') return matchList.filter(m => m.match_status !== 'matched');
    if (matchFilter === 'llm_tag_for_icon') return matchList.filter(m => m.match_method === 'llm_tag_for_icon');
    if (matchFilter === 'llm_icon_for_tag') return matchList.filter(m => m.match_method === 'llm_icon_for_tag');
    return matchList;
  }, [matches, matchFilter]);

  // Get stage status counts with detailed breakdown
  const stageCounts = useMemo(() => {
    // Safety checks for undefined arrays
    const icons = iconDetections || [];
    const labels = labelDetections || [];
    const matchList = matches || [];
    
    // Icons breakdown
    const iconTotal = icons.length;
    const iconVerified = icons.filter(d => d.verification_status === 'verified').length;
    const iconRejected = icons.filter(d => d.verification_status === 'rejected').length;
    const iconPending = icons.filter(d => d.verification_status === 'pending').length;
    
    // Labels breakdown - separate overlap removal from LLM verification rejections
    const labelTotal = labels.length;
    const labelVerified = labels.filter(d => d.verification_status === 'verified').length;
    
    // Use rejection_source to distinguish between overlap removal and LLM verification rejections
    const labelsRemovedByOverlap = labels.filter(d => 
      d.verification_status === 'rejected' && d.rejection_source === 'overlap_removal'
    ).length;
    const labelsRejectedByLLM = labels.filter(d => 
      d.verification_status === 'rejected' && d.rejection_source === 'llm_verification'
    ).length;
    const labelRejected = labels.filter(d => d.verification_status === 'rejected').length;
    const labelPending = labels.filter(d => d.verification_status === 'pending').length;
    
    // Labels after overlap removal (not rejected by overlap removal - includes pending and verified)
    const labelsAfterOverlap = labels.filter(d => 
      d.rejection_source !== 'overlap_removal'
    ).length;
    
    // Final counts (verified only - ready for matching)
    const iconsFinal = iconVerified;
    const labelsFinal = labelVerified;
    
    // Matching breakdown
    const matchedCount = matchList.filter(m => m.match_status === 'matched').length;
    const unmatchedIconsCount = matchList.filter(m => m.match_status === 'unmatched_icon').length;
    const unassignedTagsCount = matchList.filter(m => m.match_status === 'unassigned_tag').length;
    
    return {
      // Raw counts
      icons: iconTotal,
      labels: labelTotal,
      
      // Icon verification
      iconVerified,
      iconRejected,
      iconPending,
      iconsFinal,
      
      // Label verification (LLM only, not overlap removal)
      labelVerified,
      labelRejected,
      labelsRejectedByLLM,
      labelPending,
      labelsFinal,
      
      // Overlap removal
      labelsAfterOverlap,
      labelsRemovedByOverlap,
      
      // Matching
      matched: matchedCount,
      unmatchedIcons: unmatchedIconsCount,
      unassignedTags: unassignedTagsCount,
      total: matchList.length,
    };
  }, [iconDetections, labelDetections, matches]);

  // PDF URL
  const pdfUrl = selectedProject ? api.getProjectPdf(selectedProject.id) : null;

  // Bounding boxes for PDF viewer
  const boundingBoxes = useMemo(() => {
    const boxes = [];
    const stage = STAGES[currentStage]?.id;

    // Helper to get stage-specific color
    const getDetectionColor = (det, isSelected) => {
      if (isSelected) return '#3b82f6'; // Blue for selected
      
      // Stage-specific coloring
      if (stage === 'detection') {
        // Raw Detection: Neutral amber color for all (shows raw state)
        return '#f59e0b';
      } else if (stage === 'overlap') {
        // Overlap Removal: Red for removed, green for kept
        return det.verification_status === 'rejected' ? '#ef4444' : '#22c55e';
      } else {
        // LLM Verification and others: Full verification status colors
        return det.verification_status === 'verified' ? '#22c55e' 
          : det.verification_status === 'rejected' ? '#ef4444' 
          : '#f59e0b';
      }
    };

    // Helper to calculate IoU (Intersection over Union) between two normalized bboxes
    const calculateIoU = (bbox1, bbox2) => {
      // bbox_normalized format: [ymin, xmin, ymax, xmax] in 0-1000 scale
      const [y1min, x1min, y1max, x1max] = bbox1;
      const [y2min, x2min, y2max, x2max] = bbox2;
      
      // Calculate intersection
      const xIntersect = Math.max(0, Math.min(x1max, x2max) - Math.max(x1min, x2min));
      const yIntersect = Math.max(0, Math.min(y1max, y2max) - Math.max(y1min, y2min));
      const intersection = xIntersect * yIntersect;
      
      // Calculate union
      const area1 = (x1max - x1min) * (y1max - y1min);
      const area2 = (x2max - x2min) * (y2max - y2min);
      const union = area1 + area2 - intersection;
      
      return union > 0 ? intersection / union : 0;
    };

    // Helper to find overlapping detections for a given detection
    const findOverlappingTags = (det, allDetections) => {
      const overlapping = [];
      allDetections.forEach(other => {
        if (other.page_number !== det.page_number) return;
        // Skip rejected detections (removed by overlap removal)
        if (other.verification_status === 'rejected') return;
        
        // Check if bboxes overlap (IoU > 0.1 or centers are close)
        const iou = calculateIoU(det.bbox_normalized, other.bbox_normalized);
        if (iou > 0.1) {
          overlapping.push({
            tag_name: other.tag_name || 'Unknown',
            confidence: other.confidence,
            id: other.id,
            status: other.verification_status,
          });
        }
      });
      
      // Sort by confidence descending
      return overlapping.sort((a, b) => b.confidence - a.confidence);
    };

    // For detection and verification stages, show individual detections
    if (['detection', 'verification', 'overlap'].includes(stage)) {
      if (showIcons) {
        filteredIconDetections.forEach(det => {
          const isSelected = det.id === selectedDetectionId;
          boxes.push({
            id: `icon-${det.id}`,
            bbox_normalized: det.bbox_normalized,
            page_number: det.page_number,
            color: getDetectionColor(det, isSelected),
            confidence: det.confidence,
            label: `🔷 ${Math.round(det.confidence * 100)}%`,  // Show icon confidence
          });
        });
      }
      if (showLabels) {
        filteredLabelDetections.forEach(det => {
          const isSelected = det.id === selectedDetectionId;
          // Find all overlapping tags for this detection (only non-rejected tags)
          // Don't calculate overlapping tags for rejected detections
          const overlappingTags = det.verification_status !== 'rejected' 
            ? findOverlappingTags(det, labelDetections) // Use all labelDetections to find overlaps
            : [];
          
          boxes.push({
            id: `label-${det.id}`,
            bbox_normalized: det.bbox_normalized,
            page_number: det.page_number,
            color: getDetectionColor(det, isSelected),
            confidence: det.confidence,
            label: det.tag_name 
              ? `${det.tag_name} (${Math.round(det.confidence * 100)}%)` 
              : `${Math.round(det.confidence * 100)}%`,  // Show tag text + confidence
            overlappingTags: overlappingTags.length > 1 ? overlappingTags : [], // Only show if multiple tags overlap
          });
        });
      }
    }
    
    // For matching stages, show matches
    if (['matching', 'tag-matching', 'icon-matching'].includes(stage)) {
      const iconMap = new Map((iconDetections || []).map(d => [d.id, d]));
      const labelMap = new Map((labelDetections || []).map(d => [d.id, d]));
      
      filteredMatches.forEach((match, index) => {
        const icon = iconMap.get(match.icon_detection_id);
        const label = match.label_detection_id ? labelMap.get(match.label_detection_id) : null;
        const color = match.match_status === 'matched' ? '#22c55e' : '#f59e0b';
        
        if (showIcons && icon) {
          boxes.push({
            id: `match-icon-${match.id}`,
            bbox_normalized: icon.bbox_normalized,
            page_number: icon.page_number,
            color,
            label: match.llm_assigned_label || null,
          });
        }
        if (showLabels && label) {
          boxes.push({
            id: `match-label-${match.id}`,
            bbox_normalized: label.bbox_normalized,
            page_number: label.page_number,
            color,
          });
        }
      });
    }

    return boxes;
  }, [currentStage, showIcons, showLabels, filteredIconDetections, filteredLabelDetections, filteredMatches, selectedDetectionId, iconDetections, labelDetections]);

  // Toggle legend item selection
  const toggleLegendItemSelection = (itemId) => {
    setSelectedLegendItemIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  // Select all legend items
  const selectAllLegendItems = () => {
    setSelectedLegendItemIds(new Set(legendItems.map(item => item.id)));
  };

  // Deselect all legend items
  const deselectAllLegendItems = () => {
    setSelectedLegendItemIds(new Set());
  };

  // Stage execution functions
  const runDetection = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      // Preprocess icons if needed
      setProcessingMessage('Preprocessing icon templates...');
      for (const item of legendItems) {
        if (searchSelectedOnly && !selectedLegendItemIds.has(item.id)) continue;
        try {
          const templateResponse = await api.getIconTemplate(item.id);
          if (templateResponse.data && !templateResponse.data.preprocessed_icon_url) {
            await api.preprocessIcon(item.id);
          }
        } catch (e) {
          // No template for this item
        }
      }

      // Detect icons
      setProcessingMessage('Detecting icons across all pages...');
      const iconResponse = await api.detectIcons(selectedProject.id);
      const iconCount = iconResponse.data?.length || 0;

      // Detect labels
      setProcessingMessage('Detecting labels/tags across all pages...');
      const labelResponse = await api.detectLabels(selectedProject.id);
      const labelCount = labelResponse.data?.length || 0;

      // Refresh data
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        detection: { icons: iconCount, labels: labelCount }
      }));
      setCompletedStages(prev => new Set([...prev, 0]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runVerification = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      let iconResult = null;
      let labelResult = null;

      // Verify icons
      if ((iconDetections || []).length > 0) {
        setProcessingMessage('AI verifying icon detections...');
        const iconResponse = await api.verifyIconDetections(selectedProject.id);
        iconResult = iconResponse.data;
      }

      // Verify labels
      if ((labelDetections || []).length > 0) {
        setProcessingMessage('AI verifying label detections...');
        const labelResponse = await api.verifyLabelDetections(selectedProject.id);
        labelResult = labelResponse.data;
      }

      // Refresh data
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        verification: { icons: iconResult, labels: labelResult }
      }));
      setCompletedStages(prev => new Set([...prev, 1]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runOverlapRemoval = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      setProcessingMessage('Resolving overlapping tag detections...');
      const response = await api.resolveTagOverlaps(selectedProject.id);
      const result = response.data;

      // Refresh data
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        overlap: result
      }));
      setCompletedStages(prev => new Set([...prev, 2]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runBasicMatching = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      setProcessingMessage('Matching icons with tags by distance...');
      const response = await api.matchIconsAndLabels(selectedProject.id);

      // Refresh data
      await fetchMatches();
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        matching: { matches: response.data?.length || 0 }
      }));
      setCompletedStages(prev => new Set([...prev, 3]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runTagMatchingForIcons = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      setProcessingMessage('AI finding tags for unlabeled icons...');
      const response = await api.matchTagsForIcons(selectedProject.id);
      const result = response.data;

      // Refresh data
      await fetchMatches();
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        tagMatching: result
      }));
      setCompletedStages(prev => new Set([...prev, 4]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runIconMatchingForTags = async () => {
    setIsProcessing(true);
    setError(null);
    try {
      setProcessingMessage('AI finding icons for unlabeled tags...');
      const response = await api.matchIconsForTags(selectedProject.id);
      const result = response.data;

      // Refresh data
      await fetchMatches();
      await fetchProjectDetections();

      setStageResults(prev => ({
        ...prev,
        iconMatching: result
      }));
      setCompletedStages(prev => new Set([...prev, 5]));
      setProcessingMessage('');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Run all stages
  const runAllStages = async () => {
    await runDetection();
    await runOverlapRemoval();
    await runVerification();
    await runBasicMatching();
    await runTagMatchingForIcons();
    await runIconMatchingForTags();
  };

  // Get run function for current stage
  const getStageRunner = () => {
    switch (currentStage) {
      case 0: return runDetection;
      case 1: return runOverlapRemoval;
      case 2: return runVerification;
      case 3: return runBasicMatching;
      case 4: return runTagMatchingForIcons;
      case 5: return runIconMatchingForTags;
      default: return null;
    }
  };

  if (!selectedProject) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Select a project to start processing.
      </div>
    );
  }

  const currentStageConfig = STAGES[currentStage];

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Stage Navigation */}
        <div className="bg-white border-b px-4 py-3">
          <div className="flex items-center gap-1 overflow-x-auto pb-2">
            {STAGES.map((stage, index) => {
              const isActive = currentStage === index;
              const isCompleted = completedStages.has(index);
              return (
                <button
                  key={stage.id}
                  onClick={() => setCurrentStage(index)}
                  className={`
                    flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all
                    ${isActive 
                      ? 'bg-indigo-600 text-white shadow-md' 
                      : isCompleted 
                        ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}
                  `}
                >
                  <span>{stage.icon}</span>
                  <span>{stage.name}</span>
                  {isCompleted && !isActive && (
                    <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              );
            })}
          </div>
          
          {/* Stage description and action */}
          <div className="flex items-center justify-between mt-3">
            <div>
              <h2 className="text-lg font-semibold text-gray-800">
                {currentStageConfig.icon} {currentStageConfig.name}
              </h2>
              <p className="text-sm text-gray-600">{currentStageConfig.description}</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={getStageRunner()}
                disabled={isProcessing}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </>
                ) : (
                  <>Run {currentStageConfig.name}</>
                )}
              </button>
              <button
                onClick={runAllStages}
                disabled={isProcessing}
                className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 flex items-center gap-2"
              >
                ⚡ Run All Stages
              </button>
              <button
                onClick={() => {
                  fetchProjectDetections();
                  fetchMatches();
                }}
                disabled={isProcessing}
                className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                🔄
              </button>
            </div>
          </div>
        </div>

        {/* Processing status */}
        {(isProcessing || error) && (
          <div className={`px-4 py-3 ${error ? 'bg-red-50 border-b border-red-200' : 'bg-indigo-50 border-b border-indigo-200'}`}>
            {isProcessing && (
              <div className="flex items-center gap-3">
                <svg className="animate-spin h-5 w-5 text-indigo-600" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span className="text-sm font-medium text-indigo-800">{processingMessage}</span>
              </div>
            )}
            {error && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-red-700">❌ {error}</span>
                <button onClick={() => setError(null)} className="text-red-600 hover:text-red-800 text-sm">Dismiss</button>
              </div>
            )}
          </div>
        )}

        {/* Stage-specific controls */}
        <div className="bg-white border-b px-4 py-3">
          {/* Detection stage options */}
          {currentStage === 0 && (
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={searchSelectedOnly}
                  onChange={(e) => setSearchSelectedOnly(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 rounded"
                />
                <span className="text-sm text-gray-700">Search selected items only</span>
              </label>
              {searchSelectedOnly && (
                <span className="text-sm text-indigo-600">
                  ({selectedLegendItemIds.size} of {legendItems.length} selected)
                </span>
              )}
              <div className="ml-auto flex items-center gap-4">
                <div className="flex items-center gap-2 bg-amber-50 px-3 py-1 rounded-lg border border-amber-200">
                  <span className="text-amber-600 font-medium">🔷 {stageCounts.icons}</span>
                  <span className="text-xs text-amber-500">icons detected</span>
                </div>
                <div className="flex items-center gap-2 bg-amber-50 px-3 py-1 rounded-lg border border-amber-200">
                  <span className="text-amber-600 font-medium">📝 {stageCounts.labels}</span>
                  <span className="text-xs text-amber-500">tags detected</span>
                </div>
              </div>
            </div>
          )}

          {/* Overlap removal stage filters */}
          {currentStage === 1 && (() => {
            const removedCount = stageCounts.labelsRemovedByOverlap;
            const keptCount = stageCounts.labelsAfterOverlap;
            const totalCount = stageCounts.labels;
            const displayCount = filteredLabelDetections.length;
            const savingsPercent = totalCount > 0 ? Math.round((removedCount / totalCount) * 100) : 0;
            
            return (
              <div className="flex items-center gap-4 flex-wrap">
                <span className="text-sm font-medium text-gray-700">Show:</span>
                {OVERLAP_FILTERS.map(f => {
                  const count = f.key === 'all' ? totalCount : f.key === 'removed' ? removedCount : keptCount;
                  return (
                    <button
                      key={f.key}
                      onClick={() => setOverlapFilter(f.key)}
                      className={`px-3 py-1 text-sm rounded ${
                        overlapFilter === f.key
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      {f.label} ({count})
                    </button>
                  );
                })}
                <div className="ml-auto flex items-center gap-3">
                  <div className="flex items-center gap-2 bg-red-50 px-3 py-1 rounded-lg border border-red-200">
                    <span className="text-red-600 font-medium">🗑️ {removedCount}</span>
                    <span className="text-xs text-red-500">removed ({savingsPercent}%)</span>
                  </div>
                  <span className="text-gray-400">→</span>
                  <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                    <span className="text-green-600 font-medium">✓ {keptCount}</span>
                    <span className="text-xs text-green-500">tags remaining</span>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Verification stage filters */}
          {currentStage === 2 && (
            <div className="flex items-center gap-4 flex-wrap">
              <span className="text-sm font-medium text-gray-700">Show:</span>
              {VERIFICATION_FILTERS.map(f => {
                // Calculate count for each filter
                let iconCount = 0, labelCount = 0;
                if (f.key === 'all') {
                  iconCount = stageCounts.icons;
                  labelCount = stageCounts.labelsAfterOverlap; // Only non-overlap-rejected
                } else if (f.key === 'verified') {
                  iconCount = stageCounts.iconVerified;
                  labelCount = stageCounts.labelVerified;
                } else if (f.key === 'rejected') {
                  iconCount = stageCounts.iconRejected;
                  labelCount = stageCounts.labelsRejectedByLLM; // Only LLM rejections, not overlap removal
                } else if (f.key === 'pending') {
                  iconCount = stageCounts.iconPending;
                  labelCount = stageCounts.labelPending;
                }
                return (
                  <button
                    key={f.key}
                    onClick={() => setVerificationFilter(f.key)}
                    className={`px-3 py-1 text-sm rounded ${
                      verificationFilter === f.key
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {f.label} ({iconCount + labelCount})
                  </button>
                );
              })}
              <div className="ml-auto flex items-center gap-3">
                <div className="text-xs bg-gray-100 px-2 py-1 rounded">
                  <span className="text-green-600 font-medium">🔷 {stageCounts.iconVerified}✓</span>
                  <span className="text-red-600 font-medium ml-1">{stageCounts.iconRejected}✗</span>
                  <span className="text-gray-400 ml-1">icons</span>
                </div>
                <div className="text-xs bg-gray-100 px-2 py-1 rounded">
                  <span className="text-green-600 font-medium">📝 {stageCounts.labelVerified}✓</span>
                  <span className="text-red-600 font-medium ml-1">{stageCounts.labelsRejectedByLLM}✗</span>
                  <span className="text-gray-400 ml-1">tags</span>
                </div>
                <span className="text-gray-400">→</span>
                <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                  <span className="text-green-600 font-medium">✓ {stageCounts.iconVerified + stageCounts.labelVerified}</span>
                  <span className="text-xs text-green-500">verified total</span>
                </div>
              </div>
            </div>
          )}

          {/* Matching stage filters */}
          {currentStage >= 3 && (
            <div className="flex items-center gap-4 flex-wrap">
              <span className="text-sm font-medium text-gray-700">Filter:</span>
              {MATCH_FILTERS.map(f => {
                let count = 0;
                const matchList = matches || [];
                if (f.key === 'all') count = stageCounts.total;
                else if (f.key === 'matched') count = stageCounts.matched;
                else if (f.key === 'unmatched') count = stageCounts.unmatchedIcons + stageCounts.unassignedTags;
                else if (f.key === 'llm_tag_for_icon') count = matchList.filter(m => m.match_method === 'llm_matched' && m.label_detection_id).length;
                else if (f.key === 'llm_icon_for_tag') count = matchList.filter(m => m.match_method === 'llm_matched' && !m.label_detection_id).length;
                return (
                  <button
                    key={f.key}
                    onClick={() => setMatchFilter(f.key)}
                    className={`px-3 py-1 text-sm rounded ${
                      matchFilter === f.key
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {f.label} ({count})
                  </button>
                );
              })}
              <div className="ml-auto flex items-center gap-3">
                <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                  <span className="text-green-600 font-medium">🔗 {stageCounts.matched}</span>
                  <span className="text-xs text-green-500">matched pairs</span>
                </div>
                {(stageCounts.unmatchedIcons > 0 || stageCounts.unassignedTags > 0) && (
                  <div className="flex items-center gap-2 bg-amber-50 px-3 py-1 rounded-lg border border-amber-200">
                    <span className="text-amber-600 font-medium">⚠️ {stageCounts.unmatchedIcons + stageCounts.unassignedTags}</span>
                    <span className="text-xs text-amber-500">unmatched</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* View toggles */}
          <div className="flex items-center gap-4 mt-2 pt-2 border-t">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showIcons}
                onChange={(e) => setShowIcons(e.target.checked)}
                className="w-4 h-4 text-green-600 rounded"
              />
              <span className="text-sm text-green-700">Show Icons</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-blue-700">Show Labels</span>
            </label>
          </div>
        </div>

        {/* PDF Viewer */}
        <div className="flex-1 min-h-0">
          <PDFViewer
            pdfUrl={pdfUrl}
            boundingBoxes={boundingBoxes}
            scrollToPage={scrollToPage}
            isEditable={false}
            createMode={false}
          />
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-80 border-l bg-white flex flex-col overflow-hidden">
        {/* Detection stage: Legend items selection */}
        {currentStage === 0 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">Legend Items</h3>
              <p className="text-xs text-gray-600 mt-1">
                {searchSelectedOnly ? 'Select items to search' : 'All items will be searched'}
              </p>
              {searchSelectedOnly && (
                <div className="flex gap-2 mt-2">
                  <button onClick={selectAllLegendItems} className="text-xs text-indigo-600 hover:underline">Select All</button>
                  <button onClick={deselectAllLegendItems} className="text-xs text-gray-500 hover:underline">Clear</button>
                </div>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {legendItems.map((item, index) => {
                const isSelected = selectedLegendItemIds.has(item.id);
                const hasIcon = item.icon_template;
                const hasTags = item.label_templates?.length > 0;
                // Use cropped icon (preprocessed=false for reliability)
                const iconUrl = hasIcon ? api.getIconTemplateImage(item.id, false) : null;
                const tagNames = hasTags ? item.label_templates.map(t => t.tag_name || item.label_text).filter(Boolean) : [];
                
                return (
                  <div
                    key={item.id}
                    onClick={() => searchSelectedOnly && toggleLegendItemSelection(item.id)}
                    className={`
                      p-3 rounded-lg border text-sm transition-all
                      ${searchSelectedOnly ? 'cursor-pointer' : ''}
                      ${isSelected ? 'border-indigo-500 bg-indigo-50 shadow-sm' : 'border-gray-200 hover:border-gray-300'}
                    `}
                  >
                    <div className="flex items-start gap-3">
                      {searchSelectedOnly && (
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleLegendItemSelection(item.id)}
                          className="mt-1 w-4 h-4 text-indigo-600 rounded"
                          onClick={(e) => e.stopPropagation()}
                        />
                      )}
                      
                      {/* Icon thumbnail */}
                      <div className="flex-shrink-0 w-10 h-10">
                        {hasIcon ? (
                          <img 
                            src={iconUrl} 
                            alt="Icon" 
                            className="w-10 h-10 object-contain border border-gray-200 rounded bg-white"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"%3E%3Crect fill="%23f3f4f6" width="40" height="40"/%3E%3Ctext x="50%25" y="50%25" font-size="8" fill="%239ca3af" text-anchor="middle" dy=".3em"%3ENo img%3C/text%3E%3C/svg%3E';
                            }}
                          />
                        ) : (
                          <div className="w-10 h-10 border border-dashed border-gray-300 rounded flex items-center justify-center text-gray-400 text-[9px] text-center leading-tight">
                            No<br/>Icon
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        {/* Description */}
                        <div className="font-medium text-gray-800 truncate">
                          {item.description || `Item ${index + 1}`}
                        </div>
                        
                        {/* Tags */}
                        {tagNames.length > 0 ? (
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {tagNames.map((tag, i) => (
                              <span 
                                key={i}
                                className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800"
                              >
                                🏷️ {tag}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <div className="text-xs text-gray-400 mt-1.5">No tags defined</div>
                        )}
                        
                        {/* Status badges */}
                        <div className="flex gap-2 mt-2">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${hasIcon ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                            {hasIcon ? '✓ Icon Ready' : '○ No Icon'}
                          </span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${hasTags ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>
                            {hasTags ? `✓ ${tagNames.length} Tag${tagNames.length > 1 ? 's' : ''}` : '○ No Tags'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {/* Overlap Removal stage: Show removed vs kept */}
        {currentStage === 1 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">Overlap Removal Results</h3>
              <div className="text-xs mt-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-500">Before:</span>
                  <span className="font-medium">{stageCounts.labels} tags</span>
                </div>
                <div className="flex justify-between text-red-600">
                  <span>🗑️ Removed:</span>
                  <span className="font-medium">-{stageCounts.labelsRemovedByOverlap}</span>
                </div>
                <div className="flex justify-between text-green-600 border-t pt-1">
                  <span>✓ Remaining:</span>
                  <span className="font-medium">{stageCounts.labelsAfterOverlap} tags</span>
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {showLabels && filteredLabelDetections.map((det, index) => (
                <div
                  key={det.id}
                  onClick={() => {
                    setSelectedDetectionId(det.id);
                    setScrollToPage(det.page_number);
                  }}
                  className={`
                    p-2 rounded border text-xs cursor-pointer transition-all
                    ${selectedDetectionId === det.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">
                      📝 {det.tag_name ? `"${det.tag_name}"` : `Label #${index + 1}`}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      det.verification_status === 'rejected' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                    }`}>
                      {det.verification_status === 'rejected' ? 'Removed' : 'Kept'}
                    </span>
                  </div>
                  <div className="text-gray-500 mt-1">
                    Page {det.page_number} | Conf: {(det.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* LLM Verification stage: Show verification status */}
        {currentStage === 2 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">LLM Verification Results</h3>
              <div className="text-xs mt-2 space-y-2">
                {/* Icons */}
                <div className="bg-white rounded p-2 border">
                  <div className="font-medium text-gray-700 mb-1">🔷 Icons ({stageCounts.icons})</div>
                  <div className="flex gap-2 text-[10px]">
                    <span className="bg-green-100 text-green-700 px-1.5 py-0.5 rounded">✓ {stageCounts.iconVerified}</span>
                    <span className="bg-red-100 text-red-700 px-1.5 py-0.5 rounded">✗ {stageCounts.iconRejected}</span>
                    <span className="bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">⏳ {stageCounts.iconPending}</span>
                  </div>
                </div>
                {/* Tags */}
                <div className="bg-white rounded p-2 border">
                  <div className="font-medium text-gray-700 mb-1">📝 Tags ({stageCounts.labelsAfterOverlap})</div>
                  <div className="flex gap-2 text-[10px]">
                    <span className="bg-green-100 text-green-700 px-1.5 py-0.5 rounded">✓ {stageCounts.labelVerified}</span>
                    <span className="bg-red-100 text-red-700 px-1.5 py-0.5 rounded">✗ {stageCounts.labelsRejectedByLLM}</span>
                    <span className="bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">⏳ {stageCounts.labelPending}</span>
                  </div>
                </div>
                {/* Final */}
                <div className="bg-green-50 rounded p-2 border border-green-200">
                  <div className="flex justify-between text-green-700">
                    <span className="font-medium">Ready for Matching:</span>
                    <span className="font-bold">{stageCounts.iconVerified + stageCounts.labelVerified}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {showIcons && filteredIconDetections.map((det, index) => (
                <div
                  key={det.id}
                  onClick={() => {
                    setSelectedDetectionId(det.id);
                    setScrollToPage(det.page_number);
                  }}
                  className={`
                    p-2 rounded border text-xs cursor-pointer transition-all
                    ${selectedDetectionId === det.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">🔷 Icon #{index + 1}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      det.verification_status === 'verified' ? 'bg-green-100 text-green-700' :
                      det.verification_status === 'rejected' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {det.verification_status || 'pending'}
                    </span>
                  </div>
                  <div className="text-gray-500 mt-1">
                    Page {det.page_number} | Conf: {(det.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
              {showLabels && filteredLabelDetections.map((det, index) => (
                <div
                  key={det.id}
                  onClick={() => {
                    setSelectedDetectionId(det.id);
                    setScrollToPage(det.page_number);
                  }}
                  className={`
                    p-2 rounded border text-xs cursor-pointer transition-all
                    ${selectedDetectionId === det.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'}
                  `}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">
                      📝 {det.tag_name ? `"${det.tag_name}"` : `Label #${index + 1}`}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                      det.verification_status === 'verified' ? 'bg-green-100 text-green-700' :
                      det.verification_status === 'rejected' ? 'bg-red-100 text-red-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {det.verification_status || 'pending'}
                    </span>
                  </div>
                  <div className="text-gray-500 mt-1">
                    Page {det.page_number} | Conf: {(det.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Matching stages: Match list */}
        {currentStage >= 3 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">
                {currentStage === 3 ? 'Basic Matching' : currentStage === 4 ? 'Tag→Icon Matching' : 'Icon→Tag Matching'}
              </h3>
              <div className="text-xs mt-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-500">Input:</span>
                  <span className="font-medium">{stageCounts.iconVerified} icons, {stageCounts.labelVerified} tags</span>
                </div>
                <div className="flex justify-between text-green-600">
                  <span>🔗 Matched:</span>
                  <span className="font-medium">{stageCounts.matched} pairs</span>
                </div>
                {stageCounts.unmatchedIcons > 0 && (
                  <div className="flex justify-between text-amber-600">
                    <span>⚠️ Unmatched Icons:</span>
                    <span className="font-medium">{stageCounts.unmatchedIcons}</span>
                  </div>
                )}
                {stageCounts.unassignedTags > 0 && (
                  <div className="flex justify-between text-amber-600">
                    <span>⚠️ Unassigned Tags:</span>
                    <span className="font-medium">{stageCounts.unassignedTags}</span>
                  </div>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {filteredMatches.length === 0 ? (
                <div className="text-center text-gray-500 py-8 text-sm">
                  No matches yet. Run Basic Matching first.
                </div>
              ) : (
                filteredMatches.map((match, index) => {
                  const icon = (iconDetections || []).find(d => d.id === match.icon_detection_id);
                  return (
                    <div
                      key={match.id}
                      onClick={() => icon && setScrollToPage(icon.page_number)}
                      className="p-2 rounded border border-gray-200 text-xs cursor-pointer hover:bg-gray-50"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Match #{index + 1}</span>
                        <div className="flex gap-1">
                          {match.match_method?.includes('llm') && (
                            <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-[10px]">AI</span>
                          )}
                          <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                            match.match_status === 'matched' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                          }`}>
                            {match.match_status}
                          </span>
                        </div>
                      </div>
                      <div className="text-gray-500 mt-1">
                        Page {icon?.page_number || '?'} | 
                        {match.llm_assigned_label && <span className="text-purple-600 ml-1">"{match.llm_assigned_label}"</span>}
                        {match.distance > 0 && <span className="ml-1">Dist: {match.distance.toFixed(0)}px</span>}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </>
        )}

        {/* Stage results summary */}
        {Object.keys(stageResults).length > 0 && (
          <div className="p-4 border-t bg-green-50">
            <h4 className="font-semibold text-green-800 text-sm mb-2">✅ Results</h4>
            <div className="text-xs text-green-700 space-y-1">
              {stageResults.detection && (
                <div>Detection: {stageResults.detection.icons} icons, {stageResults.detection.labels} labels</div>
              )}
              {stageResults.verification?.icons && (
                <div>Icon Verification: {stageResults.verification.icons.auto_approved + stageResults.verification.icons.llm_approved} approved, {stageResults.verification.icons.llm_rejected} rejected</div>
              )}
              {stageResults.verification?.labels && (
                <div>Label Verification: {stageResults.verification.labels.auto_approved + stageResults.verification.labels.llm_approved} approved, {stageResults.verification.labels.llm_rejected} rejected</div>
              )}
              {stageResults.overlap && (
                <div>Overlap: {stageResults.overlap.tags_removed} removed, {stageResults.overlap.tags_kept} kept</div>
              )}
              {stageResults.tagMatching && (
                <div>Tag→Icon: {stageResults.tagMatching.icons_matched} matched</div>
              )}
              {stageResults.iconMatching && (
                <div>Icon→Tag: {stageResults.iconMatching.tags_matched} matched</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingSection;

