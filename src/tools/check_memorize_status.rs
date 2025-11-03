//! Check Memorize Status Tool - Monitor async memorize operations

use kodegen_mcp_tool::{Tool, error::McpError};
use kodegen_mcp_schema::claude_agent::{CheckMemorizeStatusArgs, CheckMemorizeStatusPromptArgs};
use rmcp::model::{PromptArgument, PromptMessage, PromptMessageContent, PromptMessageRole};
use serde_json::{Value, json};
use std::sync::Arc;

use super::memorize_manager::MemorizeSessionManager;

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
    type PromptArgs = CheckMemorizeStatusPromptArgs;

    fn name() -> &'static str {
        "check_memorize_status"
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

    async fn execute(&self, args: Self::Args) -> Result<Value, McpError> {
        let response = self
            .manager
            .get_status(&args.session_id)
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to get status: {}", e)))?;

        // Return structured JSON response
        Ok(json!({
            "session_id": response.session_id,
            "status": response.status,
            "memory_id": response.memory_id,
            "library": response.library,
            "progress": {
                "stage": response.progress.stage,
                "files_loaded": response.progress.files_loaded,
                "total_size_bytes": response.progress.total_size_bytes,
            },
            "runtime_ms": response.runtime_ms,
            "error": response.error,
        }))
    }

    fn prompt_arguments() -> Vec<PromptArgument> {
        vec![]
    }

    async fn prompt(&self, _args: Self::PromptArgs) -> Result<Vec<PromptMessage>, McpError> {
        Ok(vec![
            PromptMessage {
                role: PromptMessageRole::User,
                content: PromptMessageContent::text(
                    "How do I check the status of a memorize operation?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "Use check_memorize_status to monitor async memorize operations:\n\n\
                     1. Start memorization:\n\
                        memorize({\"library\": \"docs\", \"content\": \"github.com/user/repo\"})\n\
                        → Returns: {\"session_id\": \"abc-123\", \"status\": \"IN_PROGRESS\"}\n\n\
                     2. Check status (poll with delays):\n\
                        check_memorize_status({\"session_id\": \"abc-123\"})\n\
                        → Returns: {\n\
                          \"status\": \"IN_PROGRESS\",\n\
                          \"progress\": {\n\
                            \"stage\": \"Loading content\",\n\
                            \"files_loaded\": 45,\n\
                            \"total_size_bytes\": 125000\n\
                          },\n\
                          \"runtime_ms\": 2341\n\
                        }\n\n\
                     3. Continue polling until complete:\n\
                        check_memorize_status({\"session_id\": \"abc-123\"})\n\
                        → Returns: {\n\
                          \"status\": \"COMPLETED\",\n\
                          \"memory_id\": \"uuid-456\",\n\
                          \"library\": \"docs\",\n\
                          \"progress\": {\"stage\": \"Completed\"},\n\
                          \"runtime_ms\": 8753\n\
                        }\n\n\
                     Progress stages:\n\
                     - \"Initializing\": Session created\n\
                     - \"Loading content\": Fetching from URL/file/GitHub/etc\n\
                     - \"Generating embeddings\": Creating vector embeddings\n\
                     - \"Storing in database\": Saving to library\n\
                     - \"Completed\": Done (memory_id available)\n\n\
                     For large operations (full repos, many files), expect 5-30 seconds.\n\
                     Poll every 1-2 seconds until status is COMPLETED or FAILED.",
                ),
            },
        ])
    }
}
