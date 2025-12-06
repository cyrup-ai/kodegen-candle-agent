//! Check Memorize Status Tool - Monitor async memorize operations

use kodegen_mcp_schema::{Tool, ToolExecutionContext, ToolResponse, McpError};
use kodegen_mcp_schema::memory::{CheckMemorizeStatusArgs, CheckMemorizeStatusOutput, MemorizeProgress, MEMORY_CHECK_MEMORIZE_STATUS, CheckMemorizeStatusPrompts};
use std::sync::Arc;

use super::memorize_manager::{MemorizeSessionManager, MemorizeStatus};

// ============================================================================
// TOOL STRUCT
// ============================================================================

#[derive(Clone)]
pub struct CheckMemorizeStatusTool {
    manager: Arc<MemorizeSessionManager>,
}

impl CheckMemorizeStatusTool {
    pub fn new(manager: Arc<MemorizeSessionManager>) -> Self {
        Self { manager }
    }
}

// ============================================================================
// TOOL IMPLEMENTATION
// ============================================================================

impl Tool for CheckMemorizeStatusTool {
    type Args = CheckMemorizeStatusArgs;
    type Prompts = CheckMemorizeStatusPrompts;

    fn name() -> &'static str {
        MEMORY_CHECK_MEMORIZE_STATUS
    }

    fn description() -> &'static str {
        "Check the status of an async memorize operation started with memorize().\n\n\
         Returns current status, progress information, and memory_id when complete.\n\n\
         Status values:\n\
         - IN_PROGRESS: Task is still running (loading content, generating embeddings, storing)\n\
         - COMPLETED: Task finished successfully (memory_id available)\n\
         - FAILED: Task failed (error message available)\n\n\
         Poll this repeatedly (with delays) until status is COMPLETED or FAILED.\n\
         Progress includes current stage (Loading content, Generating embeddings, Storing in database)\n\
         and file counts for multi-file operations."
    }

    fn read_only() -> bool {
        true
    }

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_schema::ToolArgs>::Output>, McpError> {
        let response = self
            .manager
            .get_status(&args.session_id)
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to get status: {}", e)))?;

        // Terminal summary based on status
        let summary = match response.status {
            MemorizeStatus::InProgress => {
                format!(
                    "⏳ Memorization in progress\n\n\
                     Session: {}\n\
                     Library: {}\n\
                     Stage: {}\n\
                     Files loaded: {}\n\
                     Runtime: {:.1}s",
                    response.session_id,
                    response.library,
                    response.progress.stage,
                    response.progress.files_loaded,
                    response.runtime_ms as f64 / 1000.0
                )
            },
            MemorizeStatus::Completed => {
                format!(
                    "✓ Memorization completed\n\n\
                     Session: {}\n\
                     Library: {}\n\
                     Memory ID: {}\n\
                     Runtime: {:.1}s",
                    response.session_id,
                    response.library,
                    response.memory_id.as_deref().unwrap_or("unknown"),
                    response.runtime_ms as f64 / 1000.0
                )
            },
            MemorizeStatus::Failed => {
                format!(
                    "✗ Memorization failed\n\n\
                     Session: {}\n\
                     Library: {}\n\
                     Error: {}\n\
                     Runtime: {:.1}s",
                    response.session_id,
                    response.library,
                    response.error.as_deref().unwrap_or("Unknown error"),
                    response.runtime_ms as f64 / 1000.0
                )
            },
        };

        // Convert internal status to string
        let status_str = match response.status {
            MemorizeStatus::InProgress => "IN_PROGRESS",
            MemorizeStatus::Completed => "COMPLETED",
            MemorizeStatus::Failed => "FAILED",
        };

        Ok(ToolResponse::new(summary, CheckMemorizeStatusOutput {
            session_id: response.session_id,
            status: status_str.to_string(),
            memory_id: response.memory_id,
            library: response.library,
            progress: MemorizeProgress {
                stage: response.progress.stage,
                files_loaded: response.progress.files_loaded,
                total_size_bytes: response.progress.total_size_bytes,
            },
            runtime_ms: response.runtime_ms,
            error: response.error,
        }))
    }

}
