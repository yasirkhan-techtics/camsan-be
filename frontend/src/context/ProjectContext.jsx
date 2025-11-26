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
  };

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
};

