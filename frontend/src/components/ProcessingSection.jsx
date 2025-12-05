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
  { 
    id: 'legend-counts', 
    name: 'Legend Counts', 
    description: 'Summary table of all matched icons and tags',
    icon: '📊'
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
  const [selectedLegendRowKey, setSelectedLegendRowKey] = useState(null); // For Legend Counts stage - stores key string only
  const [selectedSidebarLegendItem, setSelectedSidebarLegendItem] = useState(null); // For filtering overlays by legend item
  
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

  // Clear sidebar legend item selection and legend row when stage changes
  useEffect(() => {
    setSelectedSidebarLegendItem(null);
    setSelectedLegendRowKey(null);
  }, [currentStage]);

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

  // Filter matches - stage-specific + user filter
  // Also adds displayStatus to show original state for each stage
  const filteredMatches = useMemo(() => {
    const matchList = matches || [];
    
    // First, filter by stage and add displayStatus to reflect original state
    let stageFiltered = matchList;
    if (currentStage === 3) {
      // Basic Matching: Show all matches, but display their "original" status
      // - llm_tag_for_icon matches were originally unmatched_icon
      // - llm_icon_for_tag matches were originally unassigned_tag
      stageFiltered = matchList.map(m => {
        if (m.match_method === 'llm_tag_for_icon') {
          return { ...m, displayStatus: 'unmatched_icon' };
        }
        if (m.match_method === 'llm_icon_for_tag') {
          return { ...m, displayStatus: 'unassigned_tag' };
        }
        return { ...m, displayStatus: m.match_status };
      });
    } else if (currentStage === 4) {
      // Tag Matching: Show distance + tag-for-icon matches
      // - llm_icon_for_tag matches were originally unassigned_tag
      stageFiltered = matchList
        .filter(m => m.match_method === 'distance' || m.match_method === 'llm_tag_for_icon')
        .map(m => ({ ...m, displayStatus: m.match_status }));
      // Also include llm_icon_for_tag as unassigned for display
      const iconForTagMatches = matchList
        .filter(m => m.match_method === 'llm_icon_for_tag')
        .map(m => ({ ...m, displayStatus: 'unassigned_tag' }));
      stageFiltered = [...stageFiltered, ...iconForTagMatches];
    } else {
      // Stage 5 (Icon Matching): Show all matches with actual status
      stageFiltered = matchList.map(m => ({ ...m, displayStatus: m.match_status }));
    }
    
    // Then apply user filter (use displayStatus for filtering)
    if (matchFilter === 'all') return stageFiltered;
    if (matchFilter === 'matched') return stageFiltered.filter(m => m.displayStatus === 'matched');
    if (matchFilter === 'unmatched') return stageFiltered.filter(m => m.displayStatus !== 'matched');
    if (matchFilter === 'llm_tag_for_icon') return stageFiltered.filter(m => m.match_method === 'llm_tag_for_icon');
    if (matchFilter === 'llm_icon_for_tag') return stageFiltered.filter(m => m.match_method === 'llm_icon_for_tag');
    return stageFiltered;
  }, [matches, matchFilter, currentStage]);

  // Get stage status counts with detailed breakdown
  const stageCounts = useMemo(() => {
    // Safety checks for undefined arrays
    let icons = iconDetections || [];
    let labels = labelDetections || [];
    let matchList = matches || [];
    
    // Filter by selected legend item if one is selected
    if (selectedSidebarLegendItem) {
      const selectedIconTemplateId = selectedSidebarLegendItem.icon_template?.id;
      const selectedLabelTemplateIds = (selectedSidebarLegendItem.label_templates || []).map(t => t.id);
      
      // Filter detections
      icons = icons.filter(d => d.icon_template_id === selectedIconTemplateId);
      labels = labels.filter(d => selectedLabelTemplateIds.includes(d.label_template_id));
      
      // Filter matches - include if either icon or label belongs to selected legend item
      const filteredIconIds = new Set(icons.map(d => d.id));
      const filteredLabelIds = new Set(labels.map(d => d.id));
      matchList = matchList.filter(m => 
        (m.icon_detection_id && filteredIconIds.has(m.icon_detection_id)) ||
        (m.label_detection_id && filteredLabelIds.has(m.label_detection_id))
      );
    }
    
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
    
    // Stage-specific matching breakdown
    // Calculate stats to reflect what each stage produced
    let matchedCount = 0;
    let unmatchedIconsCount = 0;
    let unassignedTagsCount = 0;
    
    const distanceMatches = matchList.filter(m => m.match_method === 'distance');
    const tagForIconMatches = matchList.filter(m => m.match_method === 'llm_tag_for_icon');
    const iconForTagMatches = matchList.filter(m => m.match_method === 'llm_icon_for_tag');
    
    if (currentStage === 3) {
      // Basic Matching: Show original state after basic matching
      // - matched = distance matches that are matched
      // - unmatched_icon = distance unmatched_icon + llm_tag_for_icon (these were originally unmatched icons)
      // - unassigned_tag = distance unassigned_tag + llm_icon_for_tag (these were originally unassigned tags)
      matchedCount = distanceMatches.filter(m => m.match_status === 'matched').length;
      unmatchedIconsCount = distanceMatches.filter(m => m.match_status === 'unmatched_icon').length 
                         + tagForIconMatches.length; // Were originally unmatched icons
      unassignedTagsCount = distanceMatches.filter(m => m.match_status === 'unassigned_tag').length
                         + iconForTagMatches.length; // Were originally unassigned tags
    } else if (currentStage === 4) {
      // Tag Matching: Show state after tag matching
      // - matched = distance matched + llm_tag_for_icon matched
      // - unmatched_icon = remaining distance unmatched_icon
      // - unassigned_tag = distance unassigned_tag + llm_icon_for_tag (still unassigned at this stage)
      matchedCount = distanceMatches.filter(m => m.match_status === 'matched').length
                   + tagForIconMatches.filter(m => m.match_status === 'matched').length;
      unmatchedIconsCount = distanceMatches.filter(m => m.match_status === 'unmatched_icon').length;
      unassignedTagsCount = distanceMatches.filter(m => m.match_status === 'unassigned_tag').length
                         + iconForTagMatches.length; // Were originally unassigned tags
    } else {
      // Stage 5 (Icon Matching) or others: Show all current stats
      matchedCount = matchList.filter(m => m.match_status === 'matched').length;
      unmatchedIconsCount = matchList.filter(m => m.match_status === 'unmatched_icon').length;
      unassignedTagsCount = matchList.filter(m => m.match_status === 'unassigned_tag').length;
    }
    
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
      
      // Matching (stage-specific)
      matched: matchedCount,
      unmatchedIcons: unmatchedIconsCount,
      unassignedTags: unassignedTagsCount,
      total: matchedCount + unmatchedIconsCount + unassignedTagsCount,
    };
  }, [iconDetections, labelDetections, matches, currentStage, selectedSidebarLegendItem]);

  // Legend Counts - aggregate matches by icon template and tag
  const legendCounts = useMemo(() => {
    const matchList = matches || [];
    const iconsList = iconDetections || [];
    const tables = legendTables || [];
    
    // Build maps for lookups
    const iconMap = new Map(iconsList.map(d => [d.id, d]));
    const labelMap = new Map((labelDetections || []).map(d => [d.id, d]));
    
    // Helper to check if a detection is inside any legend table area
    // bbox_normalized format: [ymin, xmin, ymax, xmax] in 0-1000 scale
    const isInsideLegendTable = (detection) => {
      if (!detection?.bbox_normalized) return false;
      const [detY1, detX1, detY2, detX2] = detection.bbox_normalized;
      const detCenterX = (detX1 + detX2) / 2;
      const detCenterY = (detY1 + detY2) / 2;
      
      for (const table of tables) {
        // Skip if not on same page (compare page_id directly)
        if (table.page_id !== detection.page_id) continue;
        
        // Use bbox_normalized 
        if (!table.bbox_normalized || table.bbox_normalized.length < 4) continue;
        const [tableY1, tableX1, tableY2, tableX2] = table.bbox_normalized;
        
        // Check if detection center is inside legend table
        if (detCenterX >= tableX1 && detCenterX <= tableX2 &&
            detCenterY >= tableY1 && detCenterY <= tableY2) {
          console.log(`Detection inside legend table: det center (${detCenterX.toFixed(0)}, ${detCenterY.toFixed(0)}) inside table (${tableX1.toFixed(0)}-${tableX2.toFixed(0)}, ${tableY1.toFixed(0)}-${tableY2.toFixed(0)})`);
          return true;
        }
      }
      return false;
    };
    
    // Group matches by icon template and tag
    const countsMap = new Map(); // key: `${iconTemplateId}-${tagName}`, value: { ... }
    
    // Track which icon templates have at least one valid (non-legend-area) match
    const iconTemplatesWithMatches = new Set();
    
    // Debug: log legend tables info
    console.log(`[LegendCounts] Tables: ${tables.length}, checking detections against legend areas`);
    tables.forEach((t, i) => {
      console.log(`  Table ${i}: page_id=${t.page_id}, bbox_normalized=${JSON.stringify(t.bbox_normalized)}`);
    });
    
    let filteredOutCount = 0;
    matchList.forEach(match => {
      if (match.match_status !== 'matched') return;
      
      const icon = match.icon_detection_id ? iconMap.get(match.icon_detection_id) : null;
      const label = match.label_detection_id ? labelMap.get(match.label_detection_id) : null;
      
      // Skip detections inside legend table areas
      if (icon && isInsideLegendTable(icon)) {
        filteredOutCount++;
        return;
      }
      if (label && isInsideLegendTable(label)) {
        filteredOutCount++;
        return;
      }
      
      const tagName = match.llm_assigned_label || label?.tag_name;
      
      if (!icon && !tagName) return;
      
      // Get icon template info
      const iconTemplateId = icon?.icon_template_id || 'unknown';
      const legendItem = legendItems.find(item => item.icon_template?.id === iconTemplateId);
      const legendItemId = legendItem?.id;
      const iconDescription = legendItem?.description || 'Unknown Icon';
      
      const key = `${iconTemplateId}-${tagName || 'untagged'}`;
      
      if (!countsMap.has(key)) {
        countsMap.set(key, {
          key,
          iconTemplateId,
          legendItemId,
          iconDescription,
          tagName: tagName || '(no tag)',
          count: 0,
          matchIds: [],
          iconDetectionIds: new Set(), // Use Set to prevent duplicates
          labelDetectionIds: new Set(), // Use Set to prevent duplicates
        });
      }
      
      const entry = countsMap.get(key);
      entry.count += 1;
      entry.matchIds.push(match.id);
      if (match.icon_detection_id) entry.iconDetectionIds.add(match.icon_detection_id);
      if (match.label_detection_id) entry.labelDetectionIds.add(match.label_detection_id);
      
      // Mark this icon template as having valid matches
      iconTemplatesWithMatches.add(iconTemplateId);
    });
    
    console.log(`[LegendCounts] Filtered out ${filteredOutCount} detections in legend areas`);
    
    // Add zero-count entries for legend items without any valid matches
    legendItems.forEach(item => {
      if (!item.icon_template?.id) return;
      const iconTemplateId = item.icon_template.id;
      
      // Skip if this icon template already has matches in the countsMap
      if (iconTemplatesWithMatches.has(iconTemplateId)) return;
      
      const key = `${iconTemplateId}-(no matches)`;
      countsMap.set(key, {
        key,
        iconTemplateId,
        legendItemId: item.id,
        iconDescription: item.description || 'Unknown Icon',
        tagName: '(no matches)',
        count: 0,
        matchIds: [],
        iconDetectionIds: new Set(),
        labelDetectionIds: new Set(),
      });
    });
    
    // Convert to array, convert Sets to arrays, and sort: items with counts first, then by description, then by tag
    return Array.from(countsMap.values())
      .map(entry => ({
        ...entry,
        iconDetectionIds: Array.from(entry.iconDetectionIds),
        labelDetectionIds: Array.from(entry.labelDetectionIds),
        count: entry.iconDetectionIds.size, // Use unique icon count instead of match count
      }))
      .sort((a, b) => {
        // Items with count > 0 come first
        if (a.count > 0 && b.count === 0) return -1;
        if (a.count === 0 && b.count > 0) return 1;
        // Then sort by description
        const descCompare = a.iconDescription.localeCompare(b.iconDescription);
        if (descCompare !== 0) return descCompare;
        // Then by tag
        return (a.tagName || '').localeCompare(b.tagName || '');
      });
  }, [matches, iconDetections, labelDetections, legendItems, legendTables, pdfPages]);

  // Look up the actual selected legend row from legendCounts using the key
  // This ensures we always use fresh data when legendCounts recalculates
  const selectedLegendRow = useMemo(() => {
    if (!selectedLegendRowKey) return null;
    return legendCounts.find(row => row.key === selectedLegendRowKey) || null;
  }, [selectedLegendRowKey, legendCounts]);

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

    // Build lookup maps for filtering by selected legend item
    const selectedLegendItemTemplates = selectedSidebarLegendItem ? {
      iconTemplateId: selectedSidebarLegendItem.icon_template?.id,
      labelTemplateIds: (selectedSidebarLegendItem.label_templates || []).map(t => t.id),
    } : null;

    // Helper to check if a detection belongs to the selected legend item
    const matchesSelectedLegendItem = (det, type) => {
      if (!selectedLegendItemTemplates) return true; // No filter, show all
      if (type === 'icon') {
        return det.icon_template_id === selectedLegendItemTemplates.iconTemplateId;
      } else {
        return selectedLegendItemTemplates.labelTemplateIds.includes(det.label_template_id);
      }
    };

    // For detection and verification stages, show individual detections
    if (['detection', 'verification', 'overlap'].includes(stage)) {
      if (showIcons) {
        filteredIconDetections
          .filter(det => matchesSelectedLegendItem(det, 'icon'))
          .forEach(det => {
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
        filteredLabelDetections
          .filter(det => matchesSelectedLegendItem(det, 'label'))
          .forEach(det => {
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
      
      // Track drawn items to avoid duplicates (same tag/icon in multiple matches)
      const drawnIconIds = new Set();
      const drawnLabelIds = new Set();
      
      // Sort matches so distance matches are processed first (they have priority)
      const sortedMatches = [...filteredMatches].sort((a, b) => {
        if (a.match_method === 'distance' && b.match_method !== 'distance') return -1;
        if (a.match_method !== 'distance' && b.match_method === 'distance') return 1;
        return 0;
      });
      
      sortedMatches.forEach((match, index) => {
        const icon = match.icon_detection_id ? iconMap.get(match.icon_detection_id) : null;
        const label = match.label_detection_id ? labelMap.get(match.label_detection_id) : null;
        
        // Filter by selected legend item
        const iconMatchesFilter = icon ? matchesSelectedLegendItem(icon, 'icon') : true;
        const labelMatchesFilter = label ? matchesSelectedLegendItem(label, 'label') : true;
        
        // If neither icon nor label matches the filter, skip this match
        if (selectedLegendItemTemplates && !iconMatchesFilter && !labelMatchesFilter) {
          return;
        }
        
        // Determine colors based on stage and match type
        // For some stages, icon and label need different colors
        let iconColor, labelColor;
        const status = match.displayStatus || match.match_status;
        
        if (stage === 'matching') {
          // Basic Matching: Show original state
          if (match.match_method === 'llm_tag_for_icon') {
            // Icon was unmatched (orange), tag was unassigned (purple)
            iconColor = '#f59e0b';
            labelColor = '#a855f7';
          } else if (match.match_method === 'llm_icon_for_tag') {
            // Icon didn't exist yet, tag was unassigned (purple)
            iconColor = null; // Don't draw
            labelColor = '#a855f7';
          } else {
            // Distance matches - use actual status
            iconColor = labelColor = status === 'matched' ? '#22c55e' 
              : status === 'unmatched_icon' ? '#f59e0b' 
              : status === 'unassigned_tag' ? '#a855f7' 
              : '#f59e0b';
          }
        } else if (stage === 'tag-matching') {
          // Tag Matching: llm_icon_for_tag icons didn't exist yet
          if (match.match_method === 'llm_icon_for_tag') {
            iconColor = null; // Don't draw
            labelColor = '#a855f7'; // Tag was unassigned
          } else {
            iconColor = labelColor = status === 'matched' ? '#22c55e' 
              : status === 'unmatched_icon' ? '#f59e0b' 
              : status === 'unassigned_tag' ? '#a855f7' 
              : '#f59e0b';
          }
        } else {
          // Icon Matching: Show actual current state
          iconColor = labelColor = status === 'matched' ? '#22c55e' 
            : status === 'unmatched_icon' ? '#f59e0b' 
            : status === 'unassigned_tag' ? '#a855f7' 
            : '#f59e0b';
        }
        
        if (showIcons && icon && iconColor && !drawnIconIds.has(icon.id) && iconMatchesFilter) {
          drawnIconIds.add(icon.id);
          boxes.push({
            id: `match-icon-${match.id}`,
            bbox_normalized: icon.bbox_normalized,
            page_number: icon.page_number,
            color: iconColor,
            label: match.llm_assigned_label || null,
          });
        }
        if (showLabels && label && labelColor && !drawnLabelIds.has(label.id) && labelMatchesFilter) {
          drawnLabelIds.add(label.id);
          boxes.push({
            id: `match-label-${match.id}`,
            bbox_normalized: label.bbox_normalized,
            page_number: label.page_number,
            color: labelColor,
            label: label.tag_name || null,
          });
        }
      });
    }

    // For legend-counts stage, show selected row's icons only (not labels)
    if (stage === 'legend-counts' && selectedLegendRow) {
      console.log('[LegendCounts Overlay] Selected row:', selectedLegendRow.key, 'tagName:', selectedLegendRow.tagName, 'iconIds:', selectedLegendRow.iconDetectionIds.length);
      const iconMap = new Map((iconDetections || []).map(d => [d.id, d]));
      
      // Draw icons for selected legend row
      selectedLegendRow.iconDetectionIds.forEach(iconId => {
        const icon = iconMap.get(iconId);
        if (icon) {
          boxes.push({
            id: `legend-icon-${iconId}`,
            bbox_normalized: icon.bbox_normalized,
            page_number: icon.page_number,
            color: '#3b82f6', // Blue for selected legend items
            label: selectedLegendRow.tagName !== '(no matches)' ? selectedLegendRow.tagName : null,
          });
        }
      });
    }

    return boxes;
  }, [currentStage, showIcons, showLabels, filteredIconDetections, filteredLabelDetections, filteredMatches, selectedDetectionId, iconDetections, labelDetections, selectedLegendRow, selectedSidebarLegendItem]);

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

      // Build legend item IDs filter if "search selected only" is enabled
      const legendItemIdsToSearch = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;

      // Detect icons
      const iconMsg = searchSelectedOnly 
        ? `Detecting icons for ${selectedLegendItemIds.size} selected item(s)...`
        : 'Detecting icons across all pages...';
      setProcessingMessage(iconMsg);
      const iconResponse = await api.detectIcons(selectedProject.id, legendItemIdsToSearch);
      const iconCount = iconResponse.data?.length || 0;

      // Detect labels
      const labelMsg = searchSelectedOnly 
        ? `Detecting labels for ${selectedLegendItemIds.size} selected item(s)...`
        : 'Detecting labels/tags across all pages...';
      setProcessingMessage(labelMsg);
      const labelResponse = await api.detectLabels(selectedProject.id, legendItemIdsToSearch);
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
      // Build legend item IDs filter if "process selected only" is enabled
      const legendItemIdsToProcess = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;
      
      let iconResult = null;
      let labelResult = null;

      // Verify icons
      if ((iconDetections || []).length > 0) {
        const msg = searchSelectedOnly 
          ? `AI verifying icons for ${selectedLegendItemIds.size} selected item(s)...`
          : 'AI verifying icon detections...';
        setProcessingMessage(msg);
        const iconResponse = await api.verifyIconDetections(selectedProject.id, legendItemIdsToProcess);
        iconResult = iconResponse.data;
      }

      // Verify labels
      if ((labelDetections || []).length > 0) {
        const msg = searchSelectedOnly 
          ? `AI verifying labels for ${selectedLegendItemIds.size} selected item(s)...`
          : 'AI verifying label detections...';
        setProcessingMessage(msg);
        const labelResponse = await api.verifyLabelDetections(selectedProject.id, legendItemIdsToProcess);
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
      // Build legend item IDs filter if "process selected only" is enabled
      const legendItemIdsToProcess = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;
      
      const msg = searchSelectedOnly 
        ? `Resolving overlaps for ${selectedLegendItemIds.size} selected item(s)...`
        : 'Resolving overlapping tag detections...';
      setProcessingMessage(msg);
      const response = await api.resolveTagOverlaps(selectedProject.id, legendItemIdsToProcess);
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
      // Build legend item IDs filter if "process selected only" is enabled
      const legendItemIdsToProcess = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;
      
      const msg = searchSelectedOnly 
        ? `Matching for ${selectedLegendItemIds.size} selected item(s)...`
        : 'Matching icons with tags by distance...';
      setProcessingMessage(msg);
      const response = await api.matchIconsAndLabels(selectedProject.id, legendItemIdsToProcess);

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
      // Build legend item IDs filter if "process selected only" is enabled
      const legendItemIdsToProcess = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;
      
      const msg = searchSelectedOnly 
        ? `AI finding tags for ${selectedLegendItemIds.size} selected item(s)...`
        : 'AI finding tags for unlabeled icons...';
      setProcessingMessage(msg);
      const response = await api.matchTagsForIcons(selectedProject.id, legendItemIdsToProcess);
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
      // Build legend item IDs filter if "process selected only" is enabled
      const legendItemIdsToProcess = searchSelectedOnly && selectedLegendItemIds.size > 0
        ? Array.from(selectedLegendItemIds)
        : null;
      
      const msg = searchSelectedOnly 
        ? `AI finding icons for ${selectedLegendItemIds.size} selected item(s)...`
        : 'AI finding icons for unlabeled tags...';
      setProcessingMessage(msg);
      const response = await api.matchIconsForTags(selectedProject.id, legendItemIdsToProcess);
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
          {/* Process selected only option - shown for all processing stages (0-5) */}
          {currentStage <= 5 && (
            <div className="flex items-center gap-4 flex-wrap mb-3 pb-3 border-b border-gray-100">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={searchSelectedOnly}
                  onChange={(e) => setSearchSelectedOnly(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 rounded"
                />
                <span className="text-sm text-gray-700">Process selected items only</span>
              </label>
              {searchSelectedOnly && (
                <span className="text-sm text-indigo-600">
                  ({selectedLegendItemIds.size} of {legendItems.length} selected)
                </span>
              )}
            </div>
          )}

          {/* Detection stage stats */}
          {currentStage === 0 && (
            <div className="flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-2 bg-amber-50 px-3 py-1 rounded-lg border border-amber-200">
                <span className="text-amber-600 font-medium">🔷 {stageCounts.icons}</span>
                <span className="text-xs text-amber-500">icons detected</span>
              </div>
              <div className="flex items-center gap-2 bg-amber-50 px-3 py-1 rounded-lg border border-amber-200">
                <span className="text-amber-600 font-medium">📝 {stageCounts.labels}</span>
                <span className="text-xs text-amber-500">tags detected</span>
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

          {/* Legend Counts stage header */}
          {currentStage === 6 && (
            <div className="flex items-center gap-4 flex-wrap">
              <span className="text-sm font-medium text-gray-700">📊 Legend Counts Summary</span>
              <div className="ml-auto flex items-center gap-3">
                <div className="flex items-center gap-2 bg-blue-50 px-3 py-1 rounded-lg border border-blue-200">
                  <span className="text-blue-600 font-medium">{legendCounts.length}</span>
                  <span className="text-xs text-blue-500">unique pairs</span>
                </div>
                <div className="flex items-center gap-2 bg-green-50 px-3 py-1 rounded-lg border border-green-200">
                  <span className="text-green-600 font-medium">{legendCounts.reduce((sum, row) => sum + row.count, 0)}</span>
                  <span className="text-xs text-green-500">total matched</span>
                </div>
                {selectedLegendRow && (
                  <div className="flex items-center gap-2 bg-indigo-50 px-3 py-1 rounded-lg border border-indigo-200">
                    <span className="text-indigo-600 font-medium">"{selectedLegendRow.tagName}"</span>
                    <span className="text-xs text-indigo-500">selected ({selectedLegendRow.count})</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Matching stage filters (stages 3-5 only) */}
          {currentStage >= 3 && currentStage <= 5 && (
            <div className="flex items-center gap-4 flex-wrap">
              <span className="text-sm font-medium text-gray-700">Filter:</span>
              {MATCH_FILTERS
                .filter(f => {
                  // Basic Matching: only show All, Matched, Unmatched
                  if (currentStage === 3) {
                    return ['all', 'matched', 'unmatched'].includes(f.key);
                  }
                  // Tag Matching: show All, Matched, Unmatched, AI Tag→Icon
                  if (currentStage === 4) {
                    return ['all', 'matched', 'unmatched', 'llm_tag_for_icon'].includes(f.key);
                  }
                  // Icon Matching: show all filters
                  return true;
                })
                .map(f => {
                let count = 0;
                const matchList = matches || [];
                if (f.key === 'all') count = stageCounts.total;
                else if (f.key === 'matched') count = stageCounts.matched;
                else if (f.key === 'unmatched') count = stageCounts.unmatchedIcons + stageCounts.unassignedTags;
                else if (f.key === 'llm_tag_for_icon') count = matchList.filter(m => m.match_method === 'llm_tag_for_icon').length;
                else if (f.key === 'llm_icon_for_tag') count = matchList.filter(m => m.match_method === 'llm_icon_for_tag').length;
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

          {/* View toggles - hide for Legend Counts stage */}
          {currentStage !== 6 && (
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
          )}
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
                {selectedSidebarLegendItem 
                  ? `Showing: ${selectedSidebarLegendItem.description || 'Selected item'}`
                  : 'Click an item to filter overlays'}
              </p>
              {selectedSidebarLegendItem && (
                <button 
                  onClick={() => setSelectedSidebarLegendItem(null)}
                  className="mt-2 text-xs text-indigo-600 hover:underline"
                >
                  ✕ Clear filter (show all)
                </button>
              )}
              {searchSelectedOnly && (
                <div className="flex gap-2 mt-2 pt-2 border-t">
                  <span className="text-xs text-gray-500">Search filter:</span>
                  <button onClick={selectAllLegendItems} className="text-xs text-indigo-600 hover:underline">Select All</button>
                  <button onClick={deselectAllLegendItems} className="text-xs text-gray-500 hover:underline">Clear</button>
                </div>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {legendItems.map((item, index) => {
                const isCheckedForSearch = selectedLegendItemIds.has(item.id);
                const isSelectedForOverlay = selectedSidebarLegendItem?.id === item.id;
                const hasIcon = item.icon_template;
                const hasTags = item.label_templates?.length > 0;
                // Use cropped icon (preprocessed=false for reliability)
                const iconUrl = hasIcon ? api.getIconTemplateImage(item.id, false) : null;
                const tagNames = hasTags ? item.label_templates.map(t => t.tag_name || item.label_text).filter(Boolean) : [];
                
                // Calculate item-specific detection counts
                const itemIconTemplateId = item.icon_template?.id;
                const itemLabelTemplateIds = (item.label_templates || []).map(t => t.id);
                const itemIconCount = (iconDetections || []).filter(d => d.icon_template_id === itemIconTemplateId).length;
                const itemLabelCount = (labelDetections || []).filter(d => itemLabelTemplateIds.includes(d.label_template_id)).length;
                
                return (
                  <div
                    key={item.id}
                    onClick={() => {
                      // Toggle overlay filter
                      setSelectedSidebarLegendItem(isSelectedForOverlay ? null : item);
                      // Scroll to first detection if selecting
                      if (!isSelectedForOverlay) {
                        const firstIconDet = (iconDetections || []).find(d => d.icon_template_id === itemIconTemplateId);
                        const firstLabelDet = (labelDetections || []).find(d => itemLabelTemplateIds.includes(d.label_template_id));
                        const firstDet = firstIconDet || firstLabelDet;
                        if (firstDet) setScrollToPage(firstDet.page_number);
                      }
                    }}
                    className={`
                      p-3 rounded-lg border text-sm transition-all cursor-pointer
                      ${isSelectedForOverlay ? 'border-indigo-500 bg-indigo-50 shadow-sm' : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'}
                    `}
                  >
                    <div className="flex items-start gap-3">
                      {searchSelectedOnly && (
                        <input
                          type="checkbox"
                          checked={isCheckedForSearch}
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
                        
                        {/* Detection counts */}
                        <div className="flex gap-2 mt-2 text-[10px]">
                          <span className={`px-1.5 py-0.5 rounded ${itemIconCount > 0 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                            🔷 {itemIconCount} icons
                          </span>
                          <span className={`px-1.5 py-0.5 rounded ${itemLabelCount > 0 ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>
                            📝 {itemLabelCount} tags
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

        {/* Stages 1-5: Unified Legend Items sidebar */}
        {currentStage >= 1 && currentStage <= 5 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">Legend Items</h3>
              <p className="text-xs text-gray-600 mt-1">
                {selectedSidebarLegendItem 
                  ? `Showing: ${selectedSidebarLegendItem.description || 'Selected item'}`
                  : 'Click an item to filter overlays'}
              </p>
              {selectedSidebarLegendItem && (
                <button 
                  onClick={() => setSelectedSidebarLegendItem(null)}
                  className="mt-2 text-xs text-indigo-600 hover:underline"
                >
                  ✕ Clear filter (show all)
                </button>
              )}
              {searchSelectedOnly && (
                <div className="flex gap-2 mt-2 pt-2 border-t">
                  <span className="text-xs text-gray-500">Process filter:</span>
                  <button onClick={selectAllLegendItems} className="text-xs text-indigo-600 hover:underline">Select All</button>
                  <button onClick={deselectAllLegendItems} className="text-xs text-gray-500 hover:underline">Clear</button>
                </div>
              )}
              {/* Stage-specific stats */}
              <div className="text-xs mt-3 space-y-1 bg-white rounded p-2 border">
                {currentStage === 1 && (
                  <>
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
                      <span className="font-medium">{stageCounts.labelsAfterOverlap}</span>
                    </div>
                  </>
                )}
                {currentStage === 2 && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-500">🔷 Icons:</span>
                      <span className="font-medium">
                        <span className="text-green-600">{stageCounts.iconVerified}✓</span>
                        {' / '}
                        <span className="text-red-600">{stageCounts.iconRejected}✗</span>
                        {' / '}
                        <span className="text-yellow-600">{stageCounts.iconPending}⏳</span>
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">📝 Tags:</span>
                      <span className="font-medium">
                        <span className="text-green-600">{stageCounts.labelVerified}✓</span>
                        {' / '}
                        <span className="text-red-600">{stageCounts.labelsRejectedByLLM}✗</span>
                        {' / '}
                        <span className="text-yellow-600">{stageCounts.labelPending}⏳</span>
                      </span>
                    </div>
                  </>
                )}
                {currentStage >= 3 && (
                  <>
                    <div className="flex justify-between text-green-600">
                      <span>🔗 Matched:</span>
                      <span className="font-medium">{stageCounts.matched}</span>
                    </div>
                    <div className="flex justify-between text-amber-600">
                      <span>⚠️ Unmatched Icons:</span>
                      <span className="font-medium">{stageCounts.unmatchedIcons}</span>
                    </div>
                    <div className="flex justify-between text-purple-600">
                      <span>🏷️ Unassigned Tags:</span>
                      <span className="font-medium">{stageCounts.unassignedTags}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {legendItems.map((item, index) => {
                const isSelectedForOverlay = selectedSidebarLegendItem?.id === item.id;
                const isCheckedForProcess = selectedLegendItemIds.has(item.id);
                const hasIcon = item.icon_template;
                const hasTags = item.label_templates?.length > 0;
                const iconUrl = hasIcon ? api.getIconTemplateImage(item.id, false) : null;
                const tagNames = hasTags ? item.label_templates.map(t => t.tag_name || item.label_text).filter(Boolean) : [];
                
                // Calculate item-specific counts
                const itemIconTemplateId = item.icon_template?.id;
                const itemLabelTemplateIds = (item.label_templates || []).map(t => t.id);
                
                const itemIconCount = (iconDetections || []).filter(d => d.icon_template_id === itemIconTemplateId).length;
                const itemLabelCount = (labelDetections || []).filter(d => itemLabelTemplateIds.includes(d.label_template_id)).length;
                
                return (
                  <div
                    key={item.id}
                    onClick={() => {
                      setSelectedSidebarLegendItem(isSelectedForOverlay ? null : item);
                      // Find first detection for this item and scroll to it
                      if (!isSelectedForOverlay) {
                        const firstIconDet = (iconDetections || []).find(d => d.icon_template_id === itemIconTemplateId);
                        const firstLabelDet = (labelDetections || []).find(d => itemLabelTemplateIds.includes(d.label_template_id));
                        const firstDet = firstIconDet || firstLabelDet;
                        if (firstDet) setScrollToPage(firstDet.page_number);
                      }
                    }}
                    className={`
                      p-3 rounded-lg border text-sm transition-all cursor-pointer
                      ${isSelectedForOverlay ? 'border-indigo-500 bg-indigo-50 shadow-sm' : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'}
                    `}
                  >
                    <div className="flex items-start gap-3">
                      {searchSelectedOnly && (
                        <input
                          type="checkbox"
                          checked={isCheckedForProcess}
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
                        
                        {/* Detection counts */}
                        <div className="flex gap-2 mt-2 text-[10px]">
                          <span className={`px-1.5 py-0.5 rounded ${itemIconCount > 0 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                            🔷 {itemIconCount} icons
                          </span>
                          <span className={`px-1.5 py-0.5 rounded ${itemLabelCount > 0 ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}`}>
                            📝 {itemLabelCount} tags
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

        {/* Legend Counts stage: Summary table */}
        {currentStage === 6 && (
          <>
            <div className="p-4 border-b bg-gray-50">
              <h3 className="font-semibold text-gray-800">📊 Legend Counts</h3>
              <p className="text-xs text-gray-500 mt-1">
                Summary of all matched icons and tags. Click a row to highlight on PDF.
              </p>
              <div className="text-xs mt-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-500">Total rows:</span>
                  <span className="font-medium">{legendCounts.length}</span>
                </div>
                <div className="flex justify-between text-green-600">
                  <span>Total matched pairs:</span>
                  <span className="font-medium">{legendCounts.reduce((sum, row) => sum + row.count, 0)}</span>
                </div>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {legendCounts.length === 0 ? (
                <div className="text-center text-gray-500 py-8 text-sm p-4">
                  No matches found. Run matching stages first.
                </div>
              ) : (
                <table className="w-full text-xs">
                  <thead className="bg-gray-100 sticky top-0">
                    <tr>
                      <th className="p-2 text-left font-semibold text-gray-700">Icon</th>
                      <th className="p-2 text-left font-semibold text-gray-700">Tag</th>
                      <th className="p-2 text-center font-semibold text-gray-700">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {legendCounts.map((row, index) => (
                      <tr
                        key={row.key}
                        onClick={() => {
                          if (row.count === 0) return; // Don't select rows with no matches
                          setSelectedLegendRowKey(selectedLegendRowKey === row.key ? null : row.key);
                          // Scroll to first detection
                          if (row.iconDetectionIds.length > 0) {
                            const icon = (iconDetections || []).find(d => d.id === row.iconDetectionIds[0]);
                            if (icon) setScrollToPage(icon.page_number);
                          }
                        }}
                        className={`
                          border-b border-gray-100 transition-colors
                          ${row.count === 0 
                            ? 'bg-gray-50 opacity-60' 
                            : selectedLegendRowKey === row.key 
                              ? 'bg-blue-100 hover:bg-blue-150 cursor-pointer' 
                              : 'hover:bg-gray-50 cursor-pointer'}
                        `}
                      >
                        <td className="p-2">
                          <div className={`flex items-center gap-2 ${row.count === 0 ? 'opacity-50' : ''}`}>
                            {row.legendItemId ? (
                              <img 
                                src={api.getIconTemplateImage(row.legendItemId)}
                                alt={row.iconDescription}
                                className="w-8 h-8 object-contain border border-gray-200 rounded bg-white"
                                onError={(e) => { e.target.style.display = 'none'; }}
                              />
                            ) : (
                              <div className="w-8 h-8 bg-gray-200 rounded flex items-center justify-center text-gray-400">
                                ?
                              </div>
                            )}
                            <span className={`truncate max-w-[100px] ${row.count === 0 ? 'text-gray-400' : 'text-gray-700'}`} title={row.iconDescription}>
                              {row.iconDescription}
                            </span>
                          </div>
                        </td>
                        <td className="p-2">
                          <span className={`
                            px-2 py-0.5 rounded text-[10px] font-medium
                            ${row.count === 0 
                              ? 'bg-gray-100 text-gray-400' 
                              : row.tagName === '(no tag)' 
                                ? 'bg-gray-100 text-gray-500' 
                                : 'bg-blue-100 text-blue-700'}
                          `}>
                            {row.tagName}
                          </span>
                        </td>
                        <td className="p-2 text-center">
                          <span className={`font-bold ${row.count === 0 ? 'text-gray-400' : 'text-gray-800'}`}>
                            {row.count}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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

