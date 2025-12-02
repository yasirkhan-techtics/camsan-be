import React, { createContext, useContext, useState, useCallback } from 'react';
import api from '../utils/api';

const ProjectContext = createContext();

export const useProject = () => {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error('useProject must be used within a ProjectProvider');
  }
  return context;
};

// Initial AI status state
const initialAIStatus = {
  isRunning: false,
  currentStep: 0,
  totalSteps: 0,
  stepDescription: '',
  results: null,
  error: null,
};

export const ProjectProvider = ({ children }) => {
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [pdfPages, setPdfPages] = useState([]);
  const [legendTables, setLegendTables] = useState([]);
  const [legendItems, setLegendItems] = useState([]);
  const [detections, setDetections] = useState([]);
  const [iconDetections, setIconDetections] = useState([]);
  const [labelDetections, setLabelDetections] = useState([]);
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // AI verification state
  const [aiStatus, setAIStatus] = useState(initialAIStatus);
  const [verificationResults, setVerificationResults] = useState(null);

  const pageMap = React.useMemo(() => {
    const map = new Map();
    pdfPages.forEach((page) => {
      if (page?.id) {
        map.set(page.id, page);
      }
    });
    return map;
  }, [pdfPages]);

  const fetchProjects = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.getProjects();
      setProjects(response.data.projects || []);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching projects:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const selectProject = useCallback(async (projectId) => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.getProject(projectId);
      setSelectedProject(response.data);
      setPdfPages(response.data.pages || []);
      setLegendTables(response.data.legend_tables || []);
      setIconDetections([]);
      setLabelDetections([]);
      setMatches([]);
    } catch (err) {
      setError(err.message);
      console.error('Error selecting project:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const updateLegendBBox = useCallback(async (legendId, bbox) => {
    if (!selectedProject) return;
    try {
      await api.updateLegendBBox(selectedProject.id, legendId, bbox);
      // Refresh the project to get updated data
      await selectProject(selectedProject.id);
    } catch (err) {
      setError(err.message);
      console.error('Error updating legend bbox:', err);
      throw err;
    }
  }, [selectedProject, selectProject]);

  const fetchLegendItems = useCallback(async (legendId) => {
    if (!selectedProject) return;
    try {
      setLoading(true);
      const response = await api.getLegendItems(selectedProject.id, legendId);
      setLegendItems(response.data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching legend items:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedProject]);

  const normalizeDetectionBbox = useCallback(
    (bbox, page) => {
      if (!page?.width || !page?.height || !Array.isArray(bbox) || bbox.length < 4) {
        return null;
      }
      const [x, y, w, h] = bbox;
      return [
        (y / page.height) * 1000,
        (x / page.width) * 1000,
        ((y + h) / page.height) * 1000,
        ((x + w) / page.width) * 1000,
      ];
    },
    []
  );

  const enhanceDetections = useCallback(
    (items, type) => {
      if (!items) return [];
      return items
        .map((det) => {
          const page = pageMap.get(det.page_id);
          const normalized = det.bbox_normalized || normalizeDetectionBbox(det.bbox, page);
          if (!normalized) return null;
          return {
            ...det,
            type,
            bbox_normalized: normalized,
            page_number: page?.page_number || 1,
            // Explicitly preserve tag_name for label detections
            tag_name: det.tag_name || null,
          };
        })
        .filter(Boolean);
    },
    [pageMap, normalizeDetectionBbox]
  );

  const fetchDetections = useCallback(
    async (legendItemId) => {
      try {
        setLoading(true);
        const [iconResponse, labelResponse] = await Promise.all([
          api.getIconDetections({ legend_item_id: legendItemId }),
          api.getLabelDetections({ legend_item_id: legendItemId }),
        ]);
        const normalizedIcons = enhanceDetections(iconResponse.data, 'icon');
        const normalizedLabels = enhanceDetections(labelResponse.data, 'label');
        setDetections([...(normalizedIcons || []), ...(normalizedLabels || [])]);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching detections:', err);
      } finally {
        setLoading(false);
      }
    },
    [enhanceDetections]
  );

  const createDetection = useCallback(async (type, data) => {
    try {
      if (type === 'icon') {
        await api.createIconDetection(data);
      } else {
        await api.createLabelDetection(data);
      }
      // Refresh detections if we have a selected legend item
      if (data.legend_item_id) {
        await fetchDetections(data.legend_item_id);
      }
    } catch (err) {
      setError(err.message);
      console.error('Error creating detection:', err);
      throw err;
    }
  }, [fetchDetections]);

  const updateDetection = useCallback(async (type, id, data) => {
    try {
      if (type === 'icon') {
        await api.updateIconDetection(id, data);
      } else {
        await api.updateLabelDetection(id, data);
      }
    } catch (err) {
      setError(err.message);
      console.error('Error updating detection:', err);
      throw err;
    }
  }, []);

  const deleteDetection = useCallback(async (type, id) => {
    try {
      if (type === 'icon') {
        await api.deleteIconDetection(id);
      } else {
        await api.deleteLabelDetection(id);
      }
      // Remove from local state
      setDetections(prev => prev.filter(d => d.id !== id));
    } catch (err) {
      setError(err.message);
      console.error('Error deleting detection:', err);
      throw err;
    }
  }, []);

  const fetchProjectDetections = useCallback(async () => {
    if (!selectedProject) return;
    try {
      const [iconRes, labelRes] = await Promise.all([
        api.getIconDetections({ project_id: selectedProject.id }),
        api.getLabelDetections({ project_id: selectedProject.id }),
      ]);
      setIconDetections(enhanceDetections(iconRes.data, 'icon'));
      setLabelDetections(enhanceDetections(labelRes.data, 'label'));
    } catch (err) {
      setError(err.message);
      console.error('Error fetching project detections:', err);
    }
  }, [selectedProject, enhanceDetections]);

  const fetchMatches = useCallback(async () => {
    if (!selectedProject) return;
    try {
      const response = await api.getMatchedResults(selectedProject.id);
      setMatches(response.data || []);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching matches:', err);
    }
  }, [selectedProject]);

  const runDetectIcons = useCallback(async () => {
    if (!selectedProject) return;
    try {
      await api.detectIcons(selectedProject.id);
      await fetchProjectDetections();
    } catch (err) {
      setError(err.message);
      console.error('Error detecting icons:', err);
      throw err;
    }
  }, [selectedProject, fetchProjectDetections]);

  const runDetectLabels = useCallback(async () => {
    if (!selectedProject) return;
    try {
      await api.detectLabels(selectedProject.id);
      await fetchProjectDetections();
    } catch (err) {
      setError(err.message);
      console.error('Error detecting labels:', err);
      throw err;
    }
  }, [selectedProject, fetchProjectDetections]);

  const runMatchIconsLabels = useCallback(async () => {
    if (!selectedProject) return;
    try {
      await api.matchIconsAndLabels(selectedProject.id);
      await fetchMatches();
    } catch (err) {
      setError(err.message);
      console.error('Error matching icons and labels:', err);
      throw err;
    }
  }, [selectedProject, fetchMatches]);

  // Helper to update AI status
  const updateAIStatus = useCallback((updates) => {
    setAIStatus(prev => ({ ...prev, ...updates }));
  }, []);

  // Reset AI status
  const resetAIStatus = useCallback(() => {
    setAIStatus(initialAIStatus);
  }, []);

  // Dismiss AI status (keep results but hide)
  const dismissAIStatus = useCallback(() => {
    setAIStatus(prev => ({ ...prev, isRunning: false, results: null, error: null }));
  }, []);

  // Run icon detection with automatic LLM verification
  const runDetectIconsWithVerification = useCallback(async () => {
    if (!selectedProject) return;
    
    try {
      // Step 1: Detect icons
      updateAIStatus({
        isRunning: true,
        currentStep: 1,
        totalSteps: 2,
        stepDescription: 'Detecting icons using template matching...',
        results: null,
        error: null,
      });
      
      await api.detectIcons(selectedProject.id);
      await fetchProjectDetections();
      
      // Step 2: Verify with LLM
      updateAIStatus({
        currentStep: 2,
        stepDescription: 'AI Agent verifying icon detections...',
      });
      
      const verifyResponse = await api.verifyIconDetections(selectedProject.id);
      const iconVerification = verifyResponse.data;
      
      // Refresh detections after verification
      await fetchProjectDetections();
      
      // Update results
      setVerificationResults(prev => ({
        ...prev,
        iconVerification,
      }));
      
      updateAIStatus({
        isRunning: false,
        results: { iconVerification },
      });
      
      return iconVerification;
    } catch (err) {
      updateAIStatus({
        isRunning: false,
        error: err.response?.data?.detail || err.message || 'Icon detection failed',
      });
      throw err;
    }
  }, [selectedProject, fetchProjectDetections, updateAIStatus]);

  // Run label detection with automatic LLM verification
  const runDetectLabelsWithVerification = useCallback(async () => {
    if (!selectedProject) return;
    
    try {
      // Step 1: Detect labels
      updateAIStatus({
        isRunning: true,
        currentStep: 1,
        totalSteps: 2,
        stepDescription: 'Detecting tags using template matching...',
        results: null,
        error: null,
      });
      
      await api.detectLabels(selectedProject.id);
      await fetchProjectDetections();
      
      // Step 2: Verify with LLM
      updateAIStatus({
        currentStep: 2,
        stepDescription: 'AI Agent verifying tag detections...',
      });
      
      const verifyResponse = await api.verifyLabelDetections(selectedProject.id);
      const labelVerification = verifyResponse.data;
      
      // Refresh detections after verification
      await fetchProjectDetections();
      
      // Update results
      setVerificationResults(prev => ({
        ...prev,
        labelVerification,
      }));
      
      updateAIStatus({
        isRunning: false,
        results: { labelVerification },
      });
      
      return labelVerification;
    } catch (err) {
      updateAIStatus({
        isRunning: false,
        error: err.response?.data?.detail || err.message || 'Tag detection failed',
      });
      throw err;
    }
  }, [selectedProject, fetchProjectDetections, updateAIStatus]);

  // Run matching with automatic LLM matching for unmatched items
  const runMatchWithLLM = useCallback(async () => {
    if (!selectedProject) return;
    
    try {
      // Step 1: Distance-based matching (includes auto tag overlap resolution)
      updateAIStatus({
        isRunning: true,
        currentStep: 1,
        totalSteps: 2,
        stepDescription: 'Matching icons and tags by distance...',
        results: null,
        error: null,
      });
      
      await api.matchIconsAndLabels(selectedProject.id);
      await fetchMatches();
      
      // Step 2: LLM matching for unmatched items
      updateAIStatus({
        currentStep: 2,
        stepDescription: 'AI Agent matching remaining unmatched items...',
      });
      
      const llmMatchResponse = await api.llmMatchUnmatched(selectedProject.id);
      const llmMatching = llmMatchResponse.data;
      
      // Refresh matches after LLM matching
      await fetchMatches();
      await fetchProjectDetections();
      
      // Update results
      setVerificationResults(prev => ({
        ...prev,
        llmMatching,
      }));
      
      updateAIStatus({
        isRunning: false,
        results: { llmMatching },
      });
      
      return llmMatching;
    } catch (err) {
      updateAIStatus({
        isRunning: false,
        error: err.response?.data?.detail || err.message || 'Matching failed',
      });
      throw err;
    }
  }, [selectedProject, fetchMatches, fetchProjectDetections, updateAIStatus]);

  // Run full pipeline: detect icons, detect labels, match with LLM
  const runFullPipelineWithLLM = useCallback(async () => {
    if (!selectedProject) return;
    
    try {
      // Step 1: Detect icons
      updateAIStatus({
        isRunning: true,
        currentStep: 1,
        totalSteps: 6,
        stepDescription: 'Detecting icons using template matching...',
        results: null,
        error: null,
      });
      
      await api.detectIcons(selectedProject.id);
      
      // Step 2: Verify icons
      updateAIStatus({
        currentStep: 2,
        stepDescription: 'AI Agent verifying icon detections...',
      });
      
      const iconVerifyResponse = await api.verifyIconDetections(selectedProject.id);
      const iconVerification = iconVerifyResponse.data;
      
      // Step 3: Detect labels
      updateAIStatus({
        currentStep: 3,
        stepDescription: 'Detecting tags using template matching...',
      });
      
      await api.detectLabels(selectedProject.id);
      
      // Step 4: Verify labels
      updateAIStatus({
        currentStep: 4,
        stepDescription: 'AI Agent verifying tag detections...',
      });
      
      const labelVerifyResponse = await api.verifyLabelDetections(selectedProject.id);
      const labelVerification = labelVerifyResponse.data;
      
      // Step 5: Distance-based matching
      updateAIStatus({
        currentStep: 5,
        stepDescription: 'Matching icons and tags by distance...',
      });
      
      await api.matchIconsAndLabels(selectedProject.id);
      
      // Step 6: LLM matching for unmatched
      updateAIStatus({
        currentStep: 6,
        stepDescription: 'AI Agent matching remaining unmatched items...',
      });
      
      const llmMatchResponse = await api.llmMatchUnmatched(selectedProject.id);
      const llmMatching = llmMatchResponse.data;
      
      // Refresh all data
      await fetchProjectDetections();
      await fetchMatches();
      
      // Update results
      const results = {
        iconVerification,
        labelVerification,
        llmMatching,
      };
      setVerificationResults(results);
      
      updateAIStatus({
        isRunning: false,
        results,
      });
      
      return results;
    } catch (err) {
      updateAIStatus({
        isRunning: false,
        error: err.response?.data?.detail || err.message || 'Pipeline failed',
      });
      throw err;
    }
  }, [selectedProject, fetchProjectDetections, fetchMatches, updateAIStatus]);

  const value = {
    projects,
    selectedProject,
    pdfPages,
    legendTables,
    legendItems,
    detections,
    iconDetections,
    labelDetections,
    matches,
    loading,
    error,
    fetchProjects,
    selectProject,
    updateLegendBBox,
    fetchLegendItems,
    fetchDetections,
    createDetection,
    updateDetection,
    deleteDetection,
    fetchProjectDetections,
    fetchMatches,
    runDetectIcons,
    runDetectLabels,
    runMatchIconsLabels,
    // AI verification
    aiStatus,
    verificationResults,
    resetAIStatus,
    dismissAIStatus,
    runDetectIconsWithVerification,
    runDetectLabelsWithVerification,
    runMatchWithLLM,
    runFullPipelineWithLLM,
  };

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
};

