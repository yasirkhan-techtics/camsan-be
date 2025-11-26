/**
 * Convert normalized coordinates (0-1000 scale) to pixel coordinates
 * @param {Array} bbox_norm - [ymin, xmin, ymax, xmax] in 0-1000 scale
 * @param {number} pageWidth - Page width in pixels
 * @param {number} pageHeight - Page height in pixels
 * @returns {Object} - {x, y, w, h} in pixels
 */
export const toPixels = (bbox_norm, pageWidth, pageHeight) => {
  const [ymin, xmin, ymax, xmax] = bbox_norm;
  return {
    x: (xmin / 1000) * pageWidth,
    y: (ymin / 1000) * pageHeight,
    w: ((xmax - xmin) / 1000) * pageWidth,
    h: ((ymax - ymin) / 1000) * pageHeight
  };
};

/**
 * Convert pixel coordinates to normalized coordinates (0-1000 scale)
 * @param {number} x - X coordinate in pixels
 * @param {number} y - Y coordinate in pixels
 * @param {number} w - Width in pixels
 * @param {number} h - Height in pixels
 * @param {number} pageWidth - Page width in pixels
 * @param {number} pageHeight - Page height in pixels
 * @returns {Array} - [ymin, xmin, ymax, xmax] in 0-1000 scale
 */
export const toNormalized = (x, y, w, h, pageWidth, pageHeight) => {
  return [
    (y / pageHeight) * 1000,
    (x / pageWidth) * 1000,
    ((y + h) / pageHeight) * 1000,
    ((x + w) / pageWidth) * 1000
  ];
};

/**
 * Get the URL for serving files via API endpoints (storage-agnostic)
 * This function is deprecated - use api.getProjectPdf, api.getPageImage, etc. directly
 * @param {string} filePath - Local file path from the backend (not used anymore)
 * @returns {string} - Empty string (files should be accessed via API endpoints)
 * @deprecated Use specific API endpoints from api.js instead
 */
export const getFileUrl = (filePath) => {
  console.warn('⚠️ getFileUrl is deprecated. Use API endpoints from api.js instead.');
  return '';
};

