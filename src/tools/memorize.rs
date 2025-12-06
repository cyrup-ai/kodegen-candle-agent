//! Memorize Tool - Store content in a named memory library (async session-based)

use kodegen_mcp_schema::{Tool, ToolExecutionContext, ToolResponse, McpError};
use kodegen_mcp_schema::memory::{MemorizeArgs, MemorizeOutput, MEMORY_MEMORIZE, MemorizePrompts};
use std::sync::Arc;

use super::memorize_manager::MemorizeSessionManager;

#[derive(Clone)]
pub struct MemorizeTool {
    manager: Arc<MemorizeSessionManager>,
}

impl MemorizeTool {
    pub fn new(manager: Arc<MemorizeSessionManager>) -> Self {
        Self { manager }
    }
}

impl Tool for MemorizeTool {
    type Args = MemorizeArgs;
    type Prompts = MemorizePrompts;

    fn name() -> &'static str {
        MEMORY_MEMORIZE
    }

    fn description() -> &'static str {
        "Start async memorization of content in a named memory library (returns immediately with session_id). \
         The content field intelligently detects and loads from: single file paths, directories (recursive), \
         glob patterns (*.rs, **/*.md), HTTP/HTTPS URLs, GitHub repos (github.com/user/repo with or without https://), \
         or literal text (fallback). Non-existent paths are treated as literal text. \
         For large operations (full repos, directories), this returns immediately and runs in background. \
         Use check_memorize_status(session_id) to monitor progress. When complete, memory_id is available. \
         Each library is a separate .db file for organizing memories by context. \
         Memories can be retrieved later using recall() by specifying the same library name."
    }

    fn read_only() -> bool {
        false
    }

    fn idempotent() -> bool {
        false // Creates new memories each time
    }

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_schema::ToolArgs>::Output>, McpError> {
        // Start async memorize session (returns immediately)
        let session_id = self
            .manager
            .start_memorize_session(args.library.clone(), args.content.clone())
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to start memorize session: {}", e)))?;

        let summary = format!(
            "✓ Memorization started\n\n\
             Session: {}\n\
             Library: {}\n\
             Status: IN_PROGRESS\n\n\
             Use check_memorize_status to monitor progress",
            session_id, args.library
        );

        Ok(ToolResponse::new(summary, MemorizeOutput {
            session_id,
            status: "IN_PROGRESS".to_string(),
            library: args.library,
            message: "Memorization started in background. Use check_memorize_status to monitor progress.".to_string(),
        }))
    }

}
