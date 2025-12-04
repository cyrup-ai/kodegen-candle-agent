//! Command execution result types with zero allocation patterns
//!
//! Provides blazing-fast result enumeration with owned strings allocated once
//! for maximum performance. Rich constructors and query methods included.

use serde::{Deserialize, Serialize};

use super::command_enums::OutputType;

// ============================================================================
// Typed Command Result Structures
// ============================================================================

/// Session creation result with zero allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResult {
    /// Session type identifier
    pub session_type: String,
    /// Session status
    pub status: String,
    /// Unique session identifier
    pub session_id: String,
}

/// Search result with typed collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Type of search performed
    pub search_type: String,
    /// Search result items (currently empty, typed for future expansion)
    pub results: Vec<serde_json::Value>,
    /// Total number of results found
    pub total_count: u64,
    /// Operation status
    pub status: String,
}

/// Command history result with typed entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryResult {
    /// Historical command entries (typed for future expansion)
    pub history: Vec<serde_json::Value>,
    /// Total number of entries
    pub total_entries: u64,
    /// Operation status
    pub status: String,
}

/// Debug information with nested configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugResult {
    /// Debug configuration and state
    pub debug_info: DebugInfo,
    /// Operation status
    pub status: String,
}

/// Nested debug information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    /// Whether debugging is enabled
    pub enabled: bool,
    /// Debug level (info, warn, error, etc.)
    pub level: String,
    /// Unix timestamp when debug info was captured
    pub timestamp: u64,
}

/// Execution statistics result with performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResult {
    /// Domain-specific execution statistics
    pub domain_stats: DomainStats,
    /// Operation status
    pub status: String,
}

/// Nested domain statistics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStats {
    /// Total commands executed
    pub total_commands: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Average execution time in milliseconds
    pub average_execution_time_ms: f64,
}

/// Export operation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Type of export performed
    pub export_type: String,
    /// Operation status
    pub status: String,
    /// Unix timestamp when export was performed
    pub timestamp: u64,
}

/// Settings operation result with configuration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsResult {
    /// Settings data (typed map for future expansion)
    pub settings: serde_json::Map<String, serde_json::Value>,
    /// Whether settings were updated
    pub updated: bool,
    /// Operation status
    pub status: String,
}

/// General search query result (used by `CommandExecutor`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySearchResult {
    /// Search query string
    pub query: String,
    /// Search scope description
    pub scope: String,
    /// Search result items (typed for future expansion)
    pub results: Vec<serde_json::Value>,
    /// Total number of results found
    pub total_found: u64,
}

/// Command execution result with zero allocation patterns where possible
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandExecutionResult {
    /// Simple success message (owned string allocated once)
    Success(String),
    
    // Typed data variants for each command category
    /// Session creation result
    Session(SessionResult),
    /// Search operation result
    Search(SearchResult),
    /// Command history result
    History(HistoryResult),
    /// Debug information result
    Debug(DebugResult),
    /// Execution statistics result
    Stats(StatsResult),
    /// Export operation result
    Export(ExportResult),
    /// Settings operation result
    Settings(SettingsResult),
    /// Query search result
    QuerySearch(QuerySearchResult),
    
    /// File result with path and metadata (owned strings allocated once)
    File {
        /// File path
        path: String,
        /// File size in bytes
        size_bytes: u64,
        /// MIME type of the file
        mime_type: String,
    },
    /// Multiple results (owned collection allocated once)
    Multiple(Vec<CommandExecutionResult>),
    /// Stream result for continuous output
    Stream {
        /// Stream identifier
        stream_id: String,
        /// Stream type
        stream_type: OutputType,
        /// Initial data if available
        initial_data: Option<String>,
    },
    /// Error result (owned string allocated once)
    Error(String),
}

impl CommandExecutionResult {
    /// Create success result with zero allocation constructor
    #[inline]
    pub fn success(message: impl Into<String>) -> Self {
        Self::Success(message.into())
    }

    /// Create file result with zero allocation constructor
    #[inline]
    pub fn file(path: impl Into<String>, size_bytes: u64, mime_type: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            size_bytes,
            mime_type: mime_type.into(),
        }
    }

    /// Create multiple results
    #[inline]
    #[must_use]
    pub fn multiple(results: Vec<CommandExecutionResult>) -> Self {
        Self::Multiple(results)
    }

    /// Create stream result with zero allocation constructor
    #[inline]
    pub fn stream(
        stream_id: impl Into<String>,
        stream_type: OutputType,
        initial_data: Option<String>,
    ) -> Self {
        Self::Stream {
            stream_id: stream_id.into(),
            stream_type,
            initial_data,
        }
    }

    /// Create error result with zero allocation constructor
    #[inline]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    /// Check if result indicates success
    #[inline]
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            Self::Success(_)
                | Self::Session(_)
                | Self::Search(_)
                | Self::History(_)
                | Self::Debug(_)
                | Self::Stats(_)
                | Self::Export(_)
                | Self::Settings(_)
                | Self::QuerySearch(_)
                | Self::File { .. }
                | Self::Multiple(_)
                | Self::Stream { .. }
        )
    }

    /// Check if result indicates error
    #[inline]
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}
