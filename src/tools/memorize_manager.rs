//! Memorize Session Manager - Async session pattern for long-running memorize operations
//!
//! Based on filesystem search pattern (one-shot async task lifecycle):
//! 1. Client calls memorize() → spawns background task → returns session_id
//! 2. Background task: resolve_content → generate_embedding → store_in_db
//! 3. Client polls check_memorize_status(session_id) to monitor progress
//! 4. Cleanup task removes old sessions (60s interval)

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use crate::builders::document::DocumentBuilder;
use uuid::Uuid;

use crate::memory::core::manager::pool::CoordinatorPool;
use crate::memory::core::primitives::metadata::MemoryMetadata;
use crate::domain::memory::primitives::types::MemoryTypeEnum;
use crate::domain::context::provider::{CandleContext, CandleFile, CandleFiles};
use crate::domain::context::CandleDocument as Document;
use tokio_stream::StreamExt;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

/// Helper function to get current Unix timestamp
/// Returns 0 if system clock is before UNIX epoch (defensive fallback)
fn unix_timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Cleanup interval in seconds
const CLEANUP_INTERVAL_SECS: u64 = 60;

/// Completed session retention time in seconds (30 seconds)
const COMPLETED_SESSION_RETENTION_SECS: u64 = 30;

/// Failed session retention time in seconds (5 minutes for debugging)
const FAILED_SESSION_RETENTION_SECS: u64 = 300;

// ============================================================================
// SESSION STATUS TYPES
// ============================================================================

/// Memorize operation status
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MemorizeStatus {
    /// Task is running
    InProgress,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
}

/// Progress information for memorize operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemorizeProgress {
    /// Current stage: "Loading content", "Generating embeddings", "Storing in database"
    pub stage: String,
    /// Number of files loaded (for directory/glob operations)
    pub files_loaded: usize,
    /// Total content size in bytes
    pub total_size_bytes: usize,
}

impl Default for MemorizeProgress {
    fn default() -> Self {
        Self {
            stage: "Initializing".to_string(),
            files_loaded: 0,
            total_size_bytes: 0,
        }
    }
}

// ============================================================================
// SESSION STRUCTURE
// ============================================================================

/// Active memorize session
pub struct MemorizeSession {
    /// Unique session ID (UUID v4)
    pub id: String,
    /// Library name for storage
    pub library: String,
    /// Original content input
    pub content_input: String,
    /// Current status
    pub status: Arc<RwLock<MemorizeStatus>>,
    /// Created memory ID (when completed)
    pub memory_id: Arc<RwLock<Option<String>>>,
    /// Error message (when failed)
    pub error: Arc<RwLock<Option<String>>>,
    /// Session start time
    pub start_time: Instant,
    /// Progress tracking
    pub progress: Arc<RwLock<MemorizeProgress>>,
    /// Last status check time (for cleanup)
    pub last_read_time: Arc<AtomicU64>,
}

impl MemorizeSession {
    /// Create new session
    pub fn new(id: String, library: String, content_input: String) -> Self {
        Self {
            id,
            library,
            content_input,
            status: Arc::new(RwLock::new(MemorizeStatus::InProgress)),
            memory_id: Arc::new(RwLock::new(None)),
            error: Arc::new(RwLock::new(None)),
            start_time: Instant::now(),
            progress: Arc::new(RwLock::new(MemorizeProgress::default())),
            last_read_time: Arc::new(AtomicU64::new(unix_timestamp_now())),
        }
    }

    /// Update progress stage
    pub async fn update_progress(&self, stage: &str, files_loaded: usize, total_size_bytes: usize) {
        let mut progress = self.progress.write().await;
        progress.stage = stage.to_string();
        progress.files_loaded = files_loaded;
        progress.total_size_bytes = total_size_bytes;
    }

    /// Mark session as completed
    pub async fn complete(&self, memory_id: String) {
        *self.status.write().await = MemorizeStatus::Completed;
        *self.memory_id.write().await = Some(memory_id);
        self.update_progress("Completed", 0, 0).await;
    }

    /// Mark session as failed
    pub async fn fail(&self, error_msg: String) {
        *self.status.write().await = MemorizeStatus::Failed;
        *self.error.write().await = Some(error_msg);
    }

    /// Update last read time (for cleanup tracking)
    pub fn touch(&self) {
        self.last_read_time.store(unix_timestamp_now(), Ordering::Relaxed);
    }
}

// ============================================================================
// STATUS RESPONSE (for check_memorize_status tool)
// ============================================================================

/// Response for check_memorize_status
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemorizeStatusResponse {
    /// Session ID
    pub session_id: String,
    /// Current status
    pub status: MemorizeStatus,
    /// Memory ID (when completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<String>,
    /// Library name
    pub library: String,
    /// Progress information
    pub progress: MemorizeProgress,
    /// Runtime in milliseconds
    pub runtime_ms: u64,
    /// Error message (when failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// SESSION MANAGER
// ============================================================================

/// Manager for memorize sessions
#[derive(Clone)]
pub struct MemorizeSessionManager {
    sessions: Arc<RwLock<HashMap<String, Arc<MemorizeSession>>>>,
    pool: Arc<CoordinatorPool>,
}

impl MemorizeSessionManager {
    /// Create new session manager
    pub fn new(pool: Arc<CoordinatorPool>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            pool,
        }
    }

    /// Start new memorize session (returns session_id immediately)
    pub async fn start_memorize_session(
        &self,
        library: String,
        content: String,
    ) -> anyhow::Result<String> {
        // Generate unique session ID using UUID v4
        let session_id = Uuid::new_v4().to_string();

        // Create session
        let session = Arc::new(MemorizeSession::new(
            session_id.clone(),
            library.clone(),
            content.clone(),
        ));

        // Store session
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), session.clone());

        // Spawn background task
        self.spawn_memorize_task(session.clone());

        Ok(session_id)
    }

    /// Get status for session
    pub async fn get_status(&self, session_id: &str) -> anyhow::Result<MemorizeStatusResponse> {
        let sessions = self.sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        // Update last read time
        session.touch();

        // Build response
        let status = session.status.read().await.clone();
        let memory_id = session.memory_id.read().await.clone();
        let error = session.error.read().await.clone();
        let progress = session.progress.read().await.clone();
        let runtime_ms = session.start_time.elapsed().as_millis() as u64;

        Ok(MemorizeStatusResponse {
            session_id: session.id.clone(),
            status,
            memory_id,
            library: session.library.clone(),
            progress,
            runtime_ms,
            error,
        })
    }

    /// Spawn background task to execute memorize operation
    fn spawn_memorize_task(&self, session: Arc<MemorizeSession>) {
        let pool = self.pool.clone();

        tokio::spawn(async move {
            log::info!(
                "Memorize task started for session {} (library: {})",
                session.id,
                session.library
            );

            // Stage 1: Loading content
            session.update_progress("Loading content", 0, 0).await;

            match Self::resolve_content(&session.content_input).await {
                Ok(resolved_content) => {
                    let content_size = resolved_content.len();
                    log::debug!(
                        "Content loaded for session {}: {} bytes",
                        session.id,
                        content_size
                    );

                    // Stage 2: Generating embeddings
                    session
                        .update_progress("Generating embeddings", 1, content_size)
                        .await;

                    // Get coordinator for library
                    match pool.get_coordinator(&session.library).await {
                        Ok(coordinator) => {
                            // Stage 3: Storing in database
                            session
                                .update_progress("Storing in database", 1, content_size)
                                .await;

                            // Store memory
                            let metadata = MemoryMetadata::default();
                            match coordinator
                                .add_memory(resolved_content, MemoryTypeEnum::LongTerm, Some(metadata))
                                .await
                            {
                                Ok(created) => {
                                    log::info!(
                                        "Memorize task completed for session {}: memory_id = {}",
                                        session.id,
                                        created.id()
                                    );
                                    session.complete(created.id().to_string()).await;
                                }
                                Err(e) => {
                                    log::error!(
                                        "Failed to store memory for session {}: {}",
                                        session.id,
                                        e
                                    );
                                    session
                                        .fail(format!("Failed to store memory: {}", e))
                                        .await;
                                }
                            }
                        }
                        Err(e) => {
                            log::error!(
                                "Failed to get coordinator for session {}: {}",
                                session.id,
                                e
                            );
                            session
                                .fail(format!("Failed to get coordinator: {}", e))
                                .await;
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to load content for session {}: {}", session.id, e);
                    session
                        .fail(format!("Failed to load content: {}", e))
                        .await;
                }
            }
        });
    }

    /// Smart content resolver (same as memorize.rs)
    async fn resolve_content(input: &str) -> anyhow::Result<String> {
        // 1. HTTP/HTTPS URL
        if input.starts_with("http://") || input.starts_with("https://") {
            let doc = Document::from_url(input).load_async().await;
            return Ok(doc.data);
        }

        // 2. GitHub URL/pattern
        if input.contains("github.com/") {
            let github_pattern = input
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .trim_start_matches("github.com/");

            let parts: Vec<&str> = github_pattern.split('/').collect();
            if parts.len() >= 2 {
                let repo = format!("{}/{}", parts[0], parts[1]);
                let path = if parts.len() > 4 && parts[2] == "blob" {
                    parts[4..].join("/")
                } else if parts.len() > 2 {
                    parts[2..].join("/")
                } else {
                    "README.md".to_string()
                };

                let doc = Document::from_github(&repo, &path).load_async().await;
                return Ok(doc.data);
            }

            return Err(anyhow::anyhow!("Invalid GitHub URL format: {}", input));
        }

        // 3. File/directory path (check before glob to avoid false positives)
        let path = std::path::Path::new(input);
        if path.exists() {
            if path.is_dir() {
                // Directory: glob all files
                let glob_pattern = format!("{}/**/*", input.trim_end_matches('/'));
                let context = CandleContext::<CandleFiles>::glob(&glob_pattern);
                let mut doc_stream = context.load();
                let mut all_content = Vec::new();

                while let Some(doc) = doc_stream.next().await {
                    let file_path = doc
                        .additional_props
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");

                    all_content.push(format!("=== {} ===\n{}", file_path, doc.data));
                }

                if all_content.is_empty() {
                    return Err(anyhow::anyhow!("No files found in directory: {}", input));
                }

                return Ok(all_content.join("\n\n"));
            }

            if path.is_file() {
                // Single file
                let context = CandleContext::<CandleFile>::of(path).await;
                let mut doc_stream = context.load();

                if let Some(doc) = doc_stream.next().await {
                    return Ok(doc.data);
                }

                return Err(anyhow::anyhow!("Failed to load file: {}", input));
            }
        }

        // 4. Glob pattern (only if contains wildcards and path doesn't exist)
        if input.contains('*') || input.contains('?') {
            let context = CandleContext::<CandleFiles>::glob(input);
            let mut doc_stream = context.load();
            let mut all_content = Vec::new();

            while let Some(doc) = doc_stream.next().await {
                let file_path = doc
                    .additional_props
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                all_content.push(format!("=== {} ===\n{}", file_path, doc.data));
            }

            if !all_content.is_empty() {
                return Ok(all_content.join("\n\n"));
            }

            // Glob matched nothing, fall through to literal text
        }

        // 5. Literal text (fallback - includes text with wildcards that don't match files)
        Ok(input.to_string())
    }

    /// Cleanup old sessions
    async fn cleanup_sessions(&self) {
        let now = unix_timestamp_now();

        let mut sessions = self.sessions.write().await;
        let mut to_remove = Vec::new();

        for (session_id, session) in sessions.iter() {
            let last_read = session.last_read_time.load(Ordering::Relaxed);
            let age_secs = now.saturating_sub(last_read);

            let status = session.status.read().await.clone();

            let should_remove = match status {
                MemorizeStatus::Completed => age_secs >= COMPLETED_SESSION_RETENTION_SECS,
                MemorizeStatus::Failed => age_secs >= FAILED_SESSION_RETENTION_SECS,
                MemorizeStatus::InProgress => false, // Never cleanup active sessions
            };

            if should_remove {
                to_remove.push(session_id.clone());
            }
        }

        for session_id in to_remove {
            log::debug!("Cleaning up memorize session: {}", session_id);
            sessions.remove(&session_id);
        }
    }

    /// Start cleanup task (call after all tools registered)
    pub fn start_cleanup_task(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(CLEANUP_INTERVAL_SECS));
            loop {
                interval.tick().await;
                self.cleanup_sessions().await;
            }
        });
    }
}
