import axios from 'axios';

const API_BASE = '/api';

export const api = {
  // Projects
  getProjects: () => axios.get(`${API_BASE}/projects`),
  getProject: (id) => axios.get(`${API_BASE}/projects/${id}`),
  createProject: (formData) => axios.post(`${API_BASE}/projects`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  deleteProject: (id) => axios.delete(`${API_BASE}/projects/${id}`),
  
  // Pages
  getPages: (projectId) => axios.get(`${API_BASE}/pages/${projectId}`),
  processPages: (projectId) => axios.post(`${API_BASE}/pages/${projectId}/process`),
  
  // Legends
  detectLegends: (projectId) => axios.post(`${API_BASE}/projects/${projectId}/detect-legends`),
  createLegendTable: (projectId, data) => axios.post(`${API_BASE}/projects/${projectId}/legends`, data),
  getLegendItems: (projectId, legendId) => axios.get(`${API_BASE}/projects/${projectId}/legends/${legendId}/items`),
  extractLegendItems: (projectId, legendId) => axios.post(`${API_BASE}/projects/${projectId}/legends/${legendId}/extract-items`),
  updateLegendBBox: (projectId, legendId, bbox) => 
    axios.put(`${API_BASE}/projects/${projectId}/legends/${legendId}/bbox`, bbox),
  deleteLegendTable: (projectId, legendId) => 
    axios.delete(`${API_BASE}/projects/${projectId}/legends/${legendId}`),
  
  // Icons
  drawIconBbox: (legendItemId, bbox) => axios.post(`${API_BASE}/icons/legend-items/${legendItemId}/draw-icon-bbox`, bbox),
  getIconTemplate: (legendItemId) => axios.get(`${API_BASE}/icons/legend-items/${legendItemId}/icon-template`),
  preprocessIcon: (legendItemId) => axios.post(`${API_BASE}/icons/legend-items/${legendItemId}/preprocess-icon`),
  detectIcons: (projectId, legendItemIds = null) => axios.post(
    `${API_BASE}/icons/projects/${projectId}/detect-icons`,
    legendItemIds ? { legend_item_ids: legendItemIds } : {}
  ),
  getIconDetections: (params) => axios.get(`${API_BASE}/icons/detections`, { params }),
  createIconDetection: (data) => axios.post(`${API_BASE}/icons/detections`, data),
  updateIconDetection: (id, data) => axios.patch(`${API_BASE}/icons/detections/${id}`, data),
  deleteIconDetection: (id) => axios.delete(`${API_BASE}/icons/detections/${id}`),
  
  // Labels
  drawLabelBbox: (legendItemId, bbox, tagName = null, labelTemplateId = null) => 
    axios.post(`${API_BASE}/labels/legend-items/${legendItemId}/draw-label-bbox`, {
      ...bbox,
      tag_name: tagName,
      label_template_id: labelTemplateId,
    }),
  getLabelTemplate: (legendItemId) => axios.get(`${API_BASE}/labels/legend-items/${legendItemId}/label-template`),
  getLabelTemplates: (legendItemId) => axios.get(`${API_BASE}/labels/legend-items/${legendItemId}/label-templates`),
  deleteLabelTemplate: (labelTemplateId) => axios.delete(`${API_BASE}/labels/label-templates/${labelTemplateId}`),
  createLabelTemplate: (legendItemId) => axios.post(`${API_BASE}/labels/legend-items/${legendItemId}/label-template`),
  detectLabels: (projectId, legendItemIds = null) => axios.post(
    `${API_BASE}/labels/projects/${projectId}/detect-labels`,
    legendItemIds ? { legend_item_ids: legendItemIds } : {}
  ),
  getLabelDetections: (params) => axios.get(`${API_BASE}/labels/detections`, { params }),
  createLabelDetection: (data) => axios.post(`${API_BASE}/labels/detections`, data),
  updateLabelDetection: (id, data) => axios.patch(`${API_BASE}/labels/detections/${id}`, data),
  deleteLabelDetection: (id) => axios.delete(`${API_BASE}/labels/detections/${id}`),
  
  // File serving (storage-agnostic)
  getProjectPdf: (projectId) => `${API_BASE}/files/projects/${projectId}/pdf`,
  getPageImage: (pageId) => `${API_BASE}/files/pages/${pageId}/image`,
  getLegendTableImage: (legendTableId) => `${API_BASE}/files/legend-tables/${legendTableId}/image`,
  getIconTemplateImage: (legendItemId, preprocessed = false) => 
    `${API_BASE}/files/legend-items/${legendItemId}/icon-template${preprocessed ? '?preprocessed=true' : ''}`,
  getLabelTemplateImage: (legendItemId) => `${API_BASE}/files/legend-items/${legendItemId}/label-template`,

  // Matching
  matchIconsAndLabels: (projectId, legendItemIds = null) => axios.post(
    `${API_BASE}/icons/projects/${projectId}/match-icons-labels`,
    legendItemIds ? { legend_item_ids: legendItemIds } : {}
  ),
  getMatchedResults: (projectId) => axios.get(`${API_BASE}/icons/projects/${projectId}/matched-results`),

  // Detection settings
  getDetectionSettings: (projectId) => axios.get(`${API_BASE}/projects/${projectId}/detection-settings`),
  updateDetectionSettings: (projectId, data) => axios.put(`${API_BASE}/projects/${projectId}/detection-settings`, data),

  // LLM Verification
  verifyIconDetections: (projectId, legendItemIds = null, batchSize = 10) => 
    axios.post(`${API_BASE}/icons/projects/${projectId}/verify-icon-detections`, { 
      batch_size: batchSize,
      ...(legendItemIds ? { legend_item_ids: legendItemIds } : {})
    }),
  verifyLabelDetections: (projectId, legendItemIds = null, batchSize = 10) => 
    axios.post(`${API_BASE}/labels/projects/${projectId}/verify-label-detections`, { 
      batch_size: batchSize,
      ...(legendItemIds ? { legend_item_ids: legendItemIds } : {})
    }),
  
  // LLM Matching for unmatched items (combined - backward compatible)
  llmMatchUnmatched: (projectId) => 
    axios.post(`${API_BASE}/icons/projects/${projectId}/llm-match-unmatched`),
  
  // Phase 5: Tag matching for unlabeled icons
  matchTagsForIcons: (projectId, legendItemIds = null) => 
    axios.post(`${API_BASE}/icons/projects/${projectId}/match-tags-for-icons`,
      legendItemIds ? { legend_item_ids: legendItemIds } : {}
    ),
  
  // Phase 6: Icon matching for unlabeled tags
  matchIconsForTags: (projectId, legendItemIds = null) => 
    axios.post(`${API_BASE}/icons/projects/${projectId}/match-icons-for-tags`,
      legendItemIds ? { legend_item_ids: legendItemIds } : {}
    ),
  
  // Tag overlap resolution
  resolveTagOverlaps: (projectId, legendItemIds = null) => 
    axios.post(`${API_BASE}/labels/projects/${projectId}/resolve-tag-overlaps`,
      legendItemIds ? { legend_item_ids: legendItemIds } : {}
    ),
};

export default api;

