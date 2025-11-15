//! List Memory Libraries Tool - List all unique library names

use kodegen_mcp_tool::{Tool, error::McpError};
use kodegen_mcp_schema::claude_agent::{ListMemoryLibrariesArgs, ListMemoryLibrariesPromptArgs, MEMORY_LIST_LIBRARIES};
use rmcp::model::{PromptArgument, PromptMessage, Content};
use serde_json::json;
use std::sync::Arc;

use crate::memory::core::manager::pool::CoordinatorPool;

#[derive(Clone)]
pub struct ListMemoryLibrariesTool {
    pool: Arc<CoordinatorPool>,
}

impl ListMemoryLibrariesTool {
    pub fn new(pool: Arc<CoordinatorPool>) -> Self {
        Self { pool }
    }
}

impl Tool for ListMemoryLibrariesTool {
    type Args = ListMemoryLibrariesArgs;
    type PromptArgs = ListMemoryLibrariesPromptArgs;

    fn name() -> &'static str {
        MEMORY_LIST_LIBRARIES
    }

    fn description() -> &'static str {
        "List all memory library database files by scanning the filesystem. \
         Returns library names found in the memory directory (all .db files). \
         Use this to discover what libraries are available for recall."
    }

    fn read_only() -> bool {
        true
    }

    async fn execute(&self, _args: Self::Args) -> Result<Vec<Content>, McpError> {
        // Use pool's list_libraries() which scans filesystem
        let libraries = self.pool.list_libraries()
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to list libraries: {}", e)))?;

        let count = libraries.len();

        let mut contents = Vec::new();

        // Terminal summary
        let summary = if libraries.is_empty() {
            "✓ No memory libraries found\n\n\
             Create a library by using memorize with a new library name".to_string()
        } else {
            let library_list = libraries.iter()
                .map(|lib| format!("  • {}", lib))
                .collect::<Vec<_>>()
                .join("\n");
            
            format!(
                "✓ Memory libraries found ({})\n\n{}",
                count, library_list
            )
        };
        contents.push(Content::text(summary));

        // JSON metadata
        let metadata = json!({
            "libraries": libraries,
            "count": count
        });
        let json_str = serde_json::to_string_pretty(&metadata)
            .unwrap_or_else(|_| "{}".to_string());
        contents.push(Content::text(json_str));

        Ok(contents)
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
                    "How do I use list_memory_libraries to see what knowledge is available?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "The list_memory_libraries tool shows all library database files that exist on disk. \
                     It scans the memory directory for .db files, NOT database contents.\n\n\
                     Basic usage:\n\
                     list_memory_libraries({})\n\n\
                     No parameters required - returns all libraries found on filesystem.\n\n\
                     Response format:\n\
                     {\n\
                       \"libraries\": [\n\
                         \"personal\",\n\
                         \"project_x\",\n\
                         \"work\"\n\
                       ],\n\
                       \"count\": 3\n\
                     }\n\n\
                     Libraries are database files at: $XDG_CONFIG_HOME/kodegen/memory/{library}.db\n\
                     Results are sorted alphabetically.\n\n\
                     When to use:\n\
                     - Discovery: \"What libraries exist?\"\n\
                     - Before recall: Check which libraries are available\n\
                     - User asks: \"What memory libraries do I have?\"\n\n\
                     Empty response:\n\
                     If no libraries exist yet, you'll get:\n\
                     {\n\
                       \"libraries\": [],\n\
                       \"count\": 0\n\
                     }\n\n\
                     Libraries are created automatically when you first memorize() with a new library name.",
                ),
            },
        ])
    }
}
