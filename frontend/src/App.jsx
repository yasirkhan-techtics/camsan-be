import React, { useState } from 'react';
import { ProjectProvider, useProject } from './context/ProjectContext';
import Sidebar from './components/Sidebar';
import LegendTablesSection from './components/LegendTablesSection';
import LegendItemsSection from './components/LegendItemsSection';
import ProcessingSection from './components/ProcessingSection';
import SettingsSection from './components/SettingsSection';
import api from './utils/api';
import './index.css';

const AppContent = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeSection, setActiveSection] = useState('legend-tables');
  const { selectedProject, selectProject, fetchProjects } = useProject();
  const [deletingProject, setDeletingProject] = useState(false);

  const handleSelectProject = (projectId) => {
    selectProject(projectId);
    setActiveSection('legend-tables');
    setSidebarOpen(false); // Hide sidebar when project is selected
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleDeleteCurrentProject = async () => {
    if (!selectedProject) return;
    
    if (!confirm(`Are you sure you want to delete "${selectedProject.name}"? This will delete all associated files and cannot be undone.`)) {
      return;
    }

    setDeletingProject(true);
    try {
      await api.deleteProject(selectedProject.id);
      selectProject(null); // Clear selection
      await fetchProjects();
      setSidebarOpen(true); // Show sidebar
      alert('Project deleted successfully!');
    } catch (error) {
      console.error('Error deleting project:', error);
      alert('Failed to delete project: ' + (error.response?.data?.detail || error.message));
    } finally {
      setDeletingProject(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - only show when open */}
      {sidebarOpen && (
        <Sidebar
          isOpen={sidebarOpen}
          onToggle={toggleSidebar}
          onSelectProject={handleSelectProject}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header with Toggle Button */}
        <div className="bg-white border-b px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={toggleSidebar}
                className="p-2 hover:bg-gray-100 rounded"
                title="Show sidebar"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            )}
            
            {selectedProject ? (
              <div>
                <h1 className="text-xl font-bold">{selectedProject.name}</h1>
                <p className="text-sm text-gray-600">Status: {selectedProject.status}</p>
              </div>
            ) : (
              <h1 className="text-xl font-bold">Select a project to get started</h1>
            )}
          </div>

          {/* Delete Project Button */}
          {selectedProject && (
            <button
              onClick={handleDeleteCurrentProject}
              disabled={deletingProject}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50 flex items-center gap-2"
              title="Delete project"
            >
              {deletingProject ? (
                <>
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Deleting...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Delete Project
                </>
              )}
            </button>
          )}
        </div>

        {/* Tabs for Sections */}
        {selectedProject && (
          <div className="bg-white border-b">
            <div className="flex space-x-1 px-4">
              <button
                onClick={() => setActiveSection('legend-tables')}
                className={`
                  px-4 py-2 font-medium border-b-2 transition-colors
                  ${activeSection === 'legend-tables'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-600 hover:text-gray-900'}
                `}
              >
                Legend Tables
              </button>
              <button
                onClick={() => setActiveSection('legend-items')}
                className={`
                  px-4 py-2 font-medium border-b-2 transition-colors
                  ${activeSection === 'legend-items'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-600 hover:text-gray-900'}
                `}
              >
                Legend Items
              </button>
              <button
                onClick={() => setActiveSection('processing')}
                className={`
                  px-4 py-2 font-medium border-b-2 transition-colors
                  ${activeSection === 'processing'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-600 hover:text-gray-900'}
                `}
              >
                Processing
              </button>
              <button
                onClick={() => setActiveSection('settings')}
                className={`
                  px-4 py-2 font-medium border-b-2 transition-colors
                  ${activeSection === 'settings'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-600 hover:text-gray-900'}
                `}
              >
                Settings
              </button>
            </div>
          </div>
        )}

        {/* Section Content */}
        <div className="flex-1 overflow-hidden">
          {!selectedProject ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-lg">No project selected</p>
                <p className="text-sm mt-2">Select a project from the sidebar to begin</p>
              </div>
            </div>
          ) : (
            <>
              {activeSection === 'legend-tables' && <LegendTablesSection />}
              {activeSection === 'legend-items' && <LegendItemsSection />}
              {activeSection === 'processing' && <ProcessingSection />}
              {activeSection === 'settings' && <SettingsSection />}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <ProjectProvider>
      <AppContent />
    </ProjectProvider>
  );
}

export default App;

