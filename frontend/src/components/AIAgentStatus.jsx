import React, { useState } from 'react';

/**
 * AI Agent Status Component
 * Shows animated progress and results for LLM verification pipeline
 */
const AIAgentStatus = ({ 
  isRunning, 
  currentStep, 
  totalSteps,
  stepDescription,
  results,
  error,
  onRetry,
  onDismiss 
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!isRunning && !results && !error) {
    return null;
  }

  const progressPercent = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

  return (
    <div className={`
      rounded-lg border overflow-hidden transition-all duration-300
      ${isRunning 
        ? 'bg-gradient-to-r from-indigo-50 via-purple-50 to-indigo-50 border-indigo-200 animate-gradient-x' 
        : error 
          ? 'bg-red-50 border-red-200'
          : 'bg-emerald-50 border-emerald-200'
      }
    `}>
      {/* Header */}
      <div 
        className="px-4 py-3 flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          {/* Status Icon */}
          {isRunning ? (
            <div className="relative">
              <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center">
                <svg className="w-5 h-5 text-indigo-600 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
              </span>
            </div>
          ) : error ? (
            <div className="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center">
              <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          ) : (
            <div className="w-8 h-8 rounded-full bg-emerald-100 flex items-center justify-center">
              <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
          )}

          {/* Title and Description */}
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-800">
                {isRunning ? 'AI Agent Working' : error ? 'Processing Failed' : 'AI Verification Complete'}
              </span>
              {isRunning && totalSteps > 0 && (
                <span className="text-sm text-indigo-600 font-medium">
                  Step {currentStep}/{totalSteps}
                </span>
              )}
            </div>
            <p className="text-sm text-gray-600">
              {isRunning 
                ? stepDescription || 'Processing...'
                : error 
                  ? error
                  : 'All verifications completed successfully'
              }
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {error && onRetry && (
            <button
              onClick={(e) => { e.stopPropagation(); onRetry(); }}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
            >
              Retry
            </button>
          )}
          {!isRunning && onDismiss && (
            <button
              onClick={(e) => { e.stopPropagation(); onDismiss(); }}
              className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          <button className="p-1.5 text-gray-400 hover:text-gray-600 transition-colors">
            <svg 
              className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Progress Bar (when running) */}
      {isRunning && totalSteps > 0 && (
        <div className="px-4 pb-2">
          <div className="h-1.5 bg-indigo-100 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Expanded Results */}
      {isExpanded && results && !isRunning && (
        <div className="px-4 pb-4 border-t border-emerald-200 mt-2 pt-3">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {results.iconVerification && (
              <>
                <ResultCard 
                  label="Icons Auto-Approved" 
                  value={results.iconVerification.auto_approved} 
                  color="emerald"
                  icon="check"
                />
                <ResultCard 
                  label="Icons LLM-Approved" 
                  value={results.iconVerification.llm_approved} 
                  color="blue"
                  icon="ai"
                />
                <ResultCard 
                  label="Icons Rejected" 
                  value={results.iconVerification.llm_rejected} 
                  color="red"
                  icon="x"
                />
                <ResultCard 
                  label="Total Icons" 
                  value={results.iconVerification.total_detections} 
                  color="gray"
                  icon="sum"
                />
              </>
            )}
            {results.labelVerification && (
              <>
                <ResultCard 
                  label="Tags Auto-Approved" 
                  value={results.labelVerification.auto_approved} 
                  color="emerald"
                  icon="check"
                />
                <ResultCard 
                  label="Tags LLM-Approved" 
                  value={results.labelVerification.llm_approved} 
                  color="blue"
                  icon="ai"
                />
                <ResultCard 
                  label="Tags Rejected" 
                  value={results.labelVerification.llm_rejected} 
                  color="red"
                  icon="x"
                />
                <ResultCard 
                  label="Total Tags" 
                  value={results.labelVerification.total_detections} 
                  color="gray"
                  icon="sum"
                />
              </>
            )}
            {results.llmMatching && (
              <>
                <ResultCard 
                  label="Icons Matched by AI" 
                  value={results.llmMatching.icons_matched} 
                  color="purple"
                  icon="link"
                />
                <ResultCard 
                  label="Tags Matched by AI" 
                  value={results.llmMatching.tags_matched} 
                  color="purple"
                  icon="link"
                />
                <ResultCard 
                  label="API Calls" 
                  value={results.llmMatching.api_calls_made} 
                  color="gray"
                  icon="api"
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const ResultCard = ({ label, value, color, icon }) => {
  const colorClasses = {
    emerald: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    blue: 'bg-blue-100 text-blue-700 border-blue-200',
    red: 'bg-red-100 text-red-700 border-red-200',
    purple: 'bg-purple-100 text-purple-700 border-purple-200',
    gray: 'bg-gray-100 text-gray-700 border-gray-200',
  };

  const icons = {
    check: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    x: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    ai: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    link: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
    sum: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
    api: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  };

  return (
    <div className={`p-3 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-center gap-2 mb-1">
        {icons[icon]}
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
};

export default AIAgentStatus;

