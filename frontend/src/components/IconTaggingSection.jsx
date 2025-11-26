import React, { useEffect, useMemo, useState } from 'react';
import PDFViewer from './PDFViewer';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'matched', label: 'Matched' },
  { key: 'unmatched', label: 'Unmatched' },
];

const IconTaggingSection = () => {
  const {
    selectedProject,
    iconDetections,
    labelDetections,
    matches,
    fetchProjectDetections,
    fetchMatches,
    runDetectIcons,
    runDetectLabels,
    runMatchIconsLabels,
  } = useProject();

  const [actionLoading, setActionLoading] = useState({
    icons: false,
    labels: false,
    match: false,
  });
  const [actionError, setActionError] = useState(null);
  const [selectedMatchId, setSelectedMatchId] = useState(null);
  const [filter, setFilter] = useState('all');
  const [scrollToPage, setScrollToPage] = useState(null);
  const [showIcons, setShowIcons] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

  useEffect(() => {
    if (!selectedProject) return;
    fetchProjectDetections();
    fetchMatches();
  }, [selectedProject, fetchProjectDetections, fetchMatches]);

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
      return {
        ...match,
        icon,
        label,
        isMatched: Boolean(label),
        page_number: icon?.page_number || label?.page_number || 1,
      };
    });
  }, [matches, iconMap, labelMap]);

  const filteredMatches = useMemo(() => {
    if (filter === 'matched') {
      return derivedMatches.filter((m) => m.isMatched);
    }
    if (filter === 'unmatched') {
      return derivedMatches.filter((m) => !m.isMatched);
    }
    return derivedMatches;
  }, [derivedMatches, filter]);

  const selectedMatch = derivedMatches.find((m) => m.id === selectedMatchId);

  const pdfUrl = selectedProject ? api.getProjectPdf(selectedProject.id) : null;

  const boundingBoxes = useMemo(() => {
    if (!selectedProject) return [];
    const boxes = [];

    if (selectedMatch) {
      if (showIcons && selectedMatch.icon) {
        boxes.push({
          id: `icon-${selectedMatch.icon.id}`,
          bbox_normalized: selectedMatch.icon.bbox_normalized,
          page_number: selectedMatch.icon.page_number,
          color: '#f97316',
          confidence: selectedMatch.icon.confidence,
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
  }, [selectedProject, selectedMatch, showIcons, showLabels, iconDetections, labelDetections]);

  const handleAction = async (type) => {
    if (!selectedProject) return;
    setActionError(null);
    setActionLoading((prev) => ({ ...prev, [type]: true }));
    try {
      if (type === 'icons') {
        await runDetectIcons();
      } else if (type === 'labels') {
        await runDetectLabels();
      } else if (type === 'match') {
        await runMatchIconsLabels();
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
    <div className="flex h-full">
      <div className="flex-1 flex flex-col">
        <div className="bg-white border-b px-4 py-3 flex flex-wrap gap-2 items-center">
          <button
            onClick={() => handleAction('icons')}
            className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50"
            disabled={actionLoading.icons}
          >
            {actionLoading.icons ? 'Detecting Icons...' : 'Detect Icons'}
          </button>
          <button
            onClick={() => handleAction('labels')}
            className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50"
            disabled={actionLoading.labels}
          >
            {actionLoading.labels ? 'Detect Tags...' : 'Detect Tags'}
          </button>
          <button
            onClick={() => handleAction('match')}
            className="px-4 py-2 rounded bg-primary text-white hover:bg-primary-dark disabled:opacity-50"
            disabled={actionLoading.match}
          >
            {actionLoading.match ? 'Matching...' : 'Match Icons & Tags'}
          </button>
          <button
            onClick={() => {
              fetchProjectDetections();
              fetchMatches();
            }}
            className="px-4 py-2 rounded border border-gray-300 hover:bg-gray-100"
          >
            Refresh Data
          </button>
          {actionError && (
            <div className="text-sm text-red-600 ml-auto">{actionError}</div>
          )}
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 bg-gray-50 p-4 overflow-hidden">
            <div className="flex items-center gap-4 mb-3 text-sm text-gray-700">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showIcons}
                  onChange={() => setShowIcons((prev) => !prev)}
                />
                Show Icons ({iconDetections.length})
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={() => setShowLabels((prev) => !prev)}
                />
                Show Tags ({labelDetections.length})
              </label>
              <span className="text-xs text-gray-500">
                {selectedMatch
                  ? 'Highlighting selected match'
                  : 'Showing all detections'}
              </span>
            </div>

            <div className="h-full border rounded bg-white overflow-hidden">
              <PDFViewer
                pdfUrl={pdfUrl}
                boundingBoxes={boundingBoxes}
                selectedBoxId={
                  selectedMatch
                    ? `icon-${selectedMatch.icon_detection_id}`
                    : null
                }
                scrollToPage={scrollToPage}
                isEditable={false}
                createMode={false}
              />
            </div>
          </div>

          <div className="w-96 border-l bg-white flex flex-col">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Matches</h3>
                <span className="text-sm text-gray-500">
                  {derivedMatches.length} total
                </span>
              </div>
              <div className="flex gap-2 mt-3">
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
            </div>
            <div className="flex-1 overflow-y-auto">
              {filteredMatches.length === 0 ? (
                <div className="p-6 text-center text-gray-500">
                  No matches to display.
                </div>
              ) : (
                filteredMatches.map((match) => {
                  const isSelected = selectedMatchId === match.id;
                  return (
                    <button
                      key={match.id}
                      onClick={() =>
                        setSelectedMatchId(
                          isSelected ? null : match.id
                        )
                      }
                      className={`w-full text-left px-4 py-3 border-b hover:bg-gray-50 ${
                        isSelected ? 'bg-blue-50' : ''
                      }`}
                    >
                      <div className="flex items-center justify-between text-sm">
                        <div className="font-medium text-gray-800">
                          Page {match.page_number}
                        </div>
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
                      <div className="mt-1 text-xs text-gray-600 space-y-0.5">
                        <div>
                          Icon: {match.icon_detection_id}{' '}
                          {match.icon
                            ? `(score ${(match.icon.confidence * 100).toFixed(1)}%)`
                            : ''}
                        </div>
                        <div>
                          Tag:{' '}
                          {match.label_detection_id ?? '—'}
                        </div>
                        <div>
                          Distance:{' '}
                          {match.distance
                            ? match.distance.toFixed(2)
                            : '—'}
                        </div>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IconTaggingSection;


