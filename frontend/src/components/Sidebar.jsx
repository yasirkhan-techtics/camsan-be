import React, { useEffect, useState } from 'react';
import { useProject } from '../context/ProjectContext';
import api from '../utils/api';

const Sidebar = ({ isOpen, onToggle, onSelectProject }) => {
  const { projects, fetchProjects, loading } = useProject();
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectFile, setNewProjectFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [deletingProjectId, setDeletingProjectId] = useState(null);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  const handleCreateProject = async (e) => {
    e.preventDefault();
    if (!newProjectName.trim() || !newProjectFile) {
      alert('Please provide project name and PDF file');
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('name', newProjectName);
      formData.append('pdf_file', newProjectFile);

      await api.createProject(formData);
      
      // Reset form
      setNewProjectName('');
      setNewProjectFile(null);
      setShowNewProjectModal(false);
      
      // Refresh projects list
      await fetchProjects();
      
      alert('Project created successfully!');
    } catch (error) {
      console.error('Error creating project:', error);
      alert('Failed to create project: ' + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteProject = async (projectId, projectName, e) => {
    e.stopPropagation(); // Prevent project selection
    
    if (!confirm(`Are you sure you want to delete "${projectName}"? This will delete all associated files and cannot be undone.`)) {
      return;
    }

    setDeletingProjectId(projectId);
    try {
      await api.deleteProject(projectId);
      await fetchProjects();
      alert('Project deleted successfully!');
    } catch (error) {
      console.error('Error deleting project:', error);
      alert('Failed to delete project: ' + (error.response?.data?.detail || error.message));
    } finally {
      setDeletingProjectId(null);
    }
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          fixed inset-y-0 left-0 z-50
          w-64 bg-gray-800 text-white
          transform transition-transform duration-300 ease-in-out
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h1 className="text-xl font-bold">Projects</h1>
            <button
              onClick={onToggle}
              className="lg:hidden p-2 hover:bg-gray-700 rounded"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Project List */}
          <div className="flex-1 overflow-y-auto p-4">
            {loading ? (
              <div className="text-center text-gray-400">Loading...</div>
            ) : projects.length === 0 ? (
              <div className="text-center text-gray-400">No projects yet</div>
            ) : (
              <div className="space-y-2">
                {projects.map((project) => (
                  <div
                    key={project.id}
                    className="relative group"
                  >
                    <button
                      onClick={() => {
                        onSelectProject(project.id);
                      }}
                      className="w-full text-left p-3 rounded hover:bg-gray-700 transition-colors"
                      disabled={deletingProjectId === project.id}
                    >
                      <div className="font-medium truncate pr-8">{project.name}</div>
                      <div className="text-sm text-gray-400 mt-1">
                        Status: {project.status}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {new Date(project.created_at).toLocaleDateString()}
                      </div>
                    </button>
                    
                    {/* Delete Button */}
                    <button
                      onClick={(e) => handleDeleteProject(project.id, project.name, e)}
                      disabled={deletingProjectId === project.id}
                      className="absolute top-3 right-3 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-600 transition-all"
                      title="Delete project"
                    >
                      {deletingProjectId === project.id ? (
                        <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-700">
            <button 
              onClick={() => setShowNewProjectModal(true)}
              className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded transition-colors font-medium"
            >
              + New Project
            </button>
          </div>
        </div>
      </div>

      {/* New Project Modal */}
      {showNewProjectModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-[60] flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6">
            <h2 className="text-2xl font-bold mb-4 text-gray-900">Create New Project</h2>
            
            <form onSubmit={handleCreateProject}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Project Name
                </label>
                <input
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
                  placeholder="Enter project name"
                  required
                />
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  PDF File
                </label>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setNewProjectFile(e.target.files[0])}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900"
                  required
                />
                {newProjectFile && (
                  <p className="text-sm text-gray-600 mt-2">
                    Selected: {newProjectFile.name}
                  </p>
                )}
              </div>

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowNewProjectModal(false);
                    setNewProjectName('');
                    setNewProjectFile(null);
                  }}
                  disabled={uploading}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center"
                >
                  {uploading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Creating...
                    </>
                  ) : (
                    'Create Project'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default Sidebar;

