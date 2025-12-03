//! Memorize Tool - Store content in a named memory library (async session-based)

use kodegen_mcp_tool::{Tool, ToolExecutionContext, ToolResponse, error::McpError};
use kodegen_mcp_schema::claude_agent::{MemorizeArgs, MemorizeOutput, MemorizePromptArgs, MEMORY_MEMORIZE};
use rmcp::model::{PromptArgument, PromptMessage};
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
    type PromptArgs = MemorizePromptArgs;

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

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_tool::ToolArgs>::Output>, McpError> {
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

    fn prompt_arguments() -> Vec<PromptArgument> {
        vec![]
    }

    async fn prompt(&self, _args: Self::PromptArgs) -> Result<Vec<PromptMessage>, McpError> {
        use rmcp::model::{PromptMessageRole, PromptMessageContent};

        Ok(vec![
            PromptMessage {
                role: PromptMessageRole::User,
                content: PromptMessageContent::text(
                    "How do I use the memorize tool to store important knowledge for later retrieval?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "The memorize tool stores content in named memory libraries (physical database files) using an async session pattern. \
                     Each library is a separate .db file for organizing memories by context (e.g., work, personal, projects).\n\n\
                     Usage pattern (async):\n\
                     1. Start memorization: memorize({\"library\": \"work\", \"content\": \"github.com/user/repo\"})\n\
                        → Returns: {\"session_id\": \"abc-123\", \"status\": \"IN_PROGRESS\"}\n\
                     2. Check progress: check_memorize_status({\"session_id\": \"abc-123\"})\n\
                        → Returns: {\"status\": \"IN_PROGRESS\", \"progress\": {\"stage\": \"Loading content\"}}\n\
                     3. Wait and check again until status is \"COMPLETED\"\n\
                        → Returns: {\"status\": \"COMPLETED\", \"memory_id\": \"uuid-456\"}\n\n\
                     Content types supported:\n\
                     - File paths: \"/path/to/file.txt\" (single file)\n\
                     - Directories: \"/path/to/dir\" (recursive - loads all files)\n\
                     - Glob patterns: \"src/**/*.rs\" or \"*.md\" (wildcards)\n\
                     - HTTP/HTTPS URLs: \"https://example.com/doc.html\" or \"http://...\"\n\
                     - GitHub: \"github.com/user/repo\" (with or without https:// prefix)\n\
                       Examples: \"github.com/user/repo\", \"https://github.com/user/repo/blob/main/file.md\"\n\
                     - Literal text: \"Important note to remember\" (when nothing else matches)\n\n\
                     Note: Non-existent file paths are treated as literal text.\n\n\
                     Library organization:\n\
                     - Each library is a separate database file at $XDG_CONFIG_HOME/kodegen/memory/{library}.db\n\
                     - Libraries auto-create on first use\n\
                     - Use libraries to separate contexts: work vs personal, project boundaries, etc.\n\n\
                     The content is automatically converted to embeddings for semantic search via recall().",
                ),
            },
        ])
    }
}
