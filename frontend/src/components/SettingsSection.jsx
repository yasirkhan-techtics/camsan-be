import React, { useEffect, useState, useMemo } from 'react';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const numericFields = [
  { key: 'icon_scale_min', label: 'Icon Scale Min', step: 0.05 },
  { key: 'icon_scale_max', label: 'Icon Scale Max', step: 0.05 },
  { key: 'icon_match_threshold', label: 'Icon Match Threshold', step: 0.01 },
  { key: 'icon_rotation_step', label: 'Icon Rotation Step', step: 1 },
  { key: 'label_scale_min', label: 'Label (Tag) Scale Min', step: 0.05 },
  { key: 'label_scale_max', label: 'Label (Tag) Scale Max', step: 0.05 },
  { key: 'label_match_threshold', label: 'Label Match Threshold', step: 0.01 },
  { key: 'label_rotation_step', label: 'Label Rotation Step', step: 1 },
  { key: 'nms_threshold', label: 'NMS Threshold', step: 0.01 },
];

const SettingsSection = () => {
  const { selectedProject } = useProject();
  const [settings, setSettings] = useState(null);
  const [formValues, setFormValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const loadSettings = async (projectId) => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const response = await api.getDetectionSettings(projectId);
      const data = response.data;
      setSettings(data);
      setFormValues(
        numericFields.reduce((acc, field) => {
          acc[field.key] =
            data[field.key] !== null && data[field.key] !== undefined
              ? data[field.key].toString()
              : '';
          return acc;
        }, {})
      );
    } catch (err) {
      console.error('Error loading detection settings:', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedProject?.id) {
      loadSettings(selectedProject.id);
    } else {
      setSettings(null);
      setFormValues({});
      setError(null);
      setSuccess(null);
    }
  }, [selectedProject?.id]);

  const handleInputChange = (key, value) => {
    setFormValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const buildPayload = () => {
    const payload = {};
    Object.entries(formValues).forEach(([key, value]) => {
      if (value === '' || value === null || value === undefined) {
        return;
      }
      const numericValue = Number(value);
      if (!Number.isNaN(numericValue)) {
        payload[key] = numericValue;
      }
    });
    return payload;
  };

  const isDirty = useMemo(() => {
    if (!settings) return false;
    return numericFields.some((field) => {
      const current = formValues[field.key];
      const original = settings[field.key];
      if (current === '' || current === null || current === undefined) {
        return original !== null && original !== undefined;
      }
      return Number(current) !== Number(original);
    });
  }, [formValues, settings]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedProject?.id || !isDirty) return;

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const payload = buildPayload();
      const response = await api.updateDetectionSettings(
        selectedProject.id,
        payload
      );
      setSettings(response.data);
      setFormValues(
        numericFields.reduce((acc, field) => {
          acc[field.key] =
            response.data[field.key] !== null &&
            response.data[field.key] !== undefined
              ? response.data[field.key].toString()
              : '';
          return acc;
        }, {})
      );
      setSuccess('Settings saved successfully.');
    } catch (err) {
      console.error('Error saving detection settings:', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (!settings) return;
    setFormValues(
      numericFields.reduce((acc, field) => {
        acc[field.key] =
          settings[field.key] !== null && settings[field.key] !== undefined
            ? settings[field.key].toString()
            : '';
        return acc;
      }, {})
    );
    setError(null);
    setSuccess(null);
  };

  if (!selectedProject) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Select a project to configure detection settings.
      </div>
    );
  }

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-auto p-6 bg-gray-50">
        <div className="max-w-4xl mx-auto bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b">
            <h2 className="text-xl font-semibold">Detection Settings</h2>
            <p className="text-sm text-gray-500 mt-1">
              Tune the scale range and thresholds used by icon and tag
              (label) searches. Larger ranges increase recall but take
              longer to run.
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="p-6 space-y-6">
              {loading ? (
                <div className="text-center text-gray-500">
                  Loading settings...
                </div>
              ) : (
                <>
                  <div>
                    <h3 className="text-lg font-medium text-gray-800 mb-3">
                      Icon Detection
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {numericFields.slice(0, 4).map((field) => (
                        <label
                          key={field.key}
                          className="flex flex-col text-sm font-medium text-gray-700"
                        >
                          {field.label}
                          <input
                            type="number"
                            step={field.step}
                            value={formValues[field.key] ?? ''}
                            onChange={(e) =>
                              handleInputChange(field.key, e.target.value)
                            }
                            className="mt-1 px-3 py-2 border rounded focus:outline-none focus:ring focus:border-primary"
                          />
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-medium text-gray-800 mb-3">
                      Tag / Label Detection
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {numericFields.slice(4, 8).map((field) => (
                        <label
                          key={field.key}
                          className="flex flex-col text-sm font-medium text-gray-700"
                        >
                          {field.label}
                          <input
                            type="number"
                            step={field.step}
                            value={formValues[field.key] ?? ''}
                            onChange={(e) =>
                              handleInputChange(field.key, e.target.value)
                            }
                            className="mt-1 px-3 py-2 border rounded focus:outline-none focus:ring focus:border-primary"
                          />
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-medium text-gray-800 mb-3">
                      Advanced
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <label className="flex flex-col text-sm font-medium text-gray-700">
                        {numericFields[8].label}
                        <input
                          type="number"
                          step={numericFields[8].step}
                          value={formValues[numericFields[8].key] ?? ''}
                          onChange={(e) =>
                            handleInputChange(numericFields[8].key, e.target.value)
                          }
                          className="mt-1 px-3 py-2 border rounded focus:outline-none focus:ring focus:border-primary"
                        />
                      </label>
                    </div>
                  </div>
                </>
              )}

              {error && (
                <div className="p-3 rounded bg-red-50 text-red-700 text-sm">
                  {error}
                </div>
              )}

              {success && (
                <div className="p-3 rounded bg-green-50 text-green-700 text-sm">
                  {success}
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t bg-gray-50 flex items-center justify-between">
              <div className="text-sm text-gray-500">
                Icon scale defaults: 0.8 – 1.2 • Tag scale defaults: 0.6 – 1.3
              </div>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-4 py-2 border rounded text-gray-600 hover:bg-gray-100"
                  disabled={loading || saving || !isDirty}
                >
                  Reset
                </button>
                <button
                  type="submit"
                  disabled={!isDirty || saving || loading}
                  className="px-4 py-2 bg-primary text-white rounded hover:bg-primary-dark disabled:opacity-50"
                >
                  {saving ? 'Saving...' : 'Save Settings'}
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SettingsSection;

