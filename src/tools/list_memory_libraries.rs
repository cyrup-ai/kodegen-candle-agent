//! List Memory Libraries Tool - List all unique library names

use kodegen_mcp_schema::{Tool, ToolExecutionContext, ToolResponse, McpError};
use kodegen_mcp_schema::memory::{ListMemoryLibrariesArgs, ListMemoryLibrariesOutput, MEMORY_LIST_LIBRARIES};
use kodegen_mcp_schema::memory::list_libraries::MemoryListLibrariesPrompts;
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
    type Prompts = MemoryListLibrariesPrompts;

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

    async fn execute(&self, _args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_schema::ToolArgs>::Output>, McpError> {
        // Use pool's list_libraries() which scans filesystem
        let libraries = self.pool.list_libraries()
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to list libraries: {}", e)))?;

        let count = libraries.len();

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

        Ok(ToolResponse::new(summary, ListMemoryLibrariesOutput {
            libraries,
            count,
        }))
    }

}
