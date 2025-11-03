//! Recall Tool - Retrieve relevant memories from a library using semantic search

use kodegen_mcp_tool::{Tool, error::McpError};
use kodegen_mcp_schema::claude_agent::{RecallArgs, RecallPromptArgs};
use rmcp::model::{PromptArgument, PromptMessage};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;

use crate::memory::core::manager::pool::CoordinatorPool;
use crate::memory::core::ops::filter::MemoryFilter;

#[derive(Clone)]
pub struct RecallTool {
    pool: Arc<CoordinatorPool>,
}

impl RecallTool {
    pub fn new(pool: Arc<CoordinatorPool>) -> Self {
        Self { pool }
    }
}

impl Tool for RecallTool {
    type Args = RecallArgs;
    type PromptArgs = RecallPromptArgs;

    fn name() -> &'static str {
        "recall"
    }

    fn description() -> &'static str {
        "Retrieve relevant memories from a library using semantic search. \
         Searches for content similar to the provided context and returns the most relevant results. \
         Uses vector similarity (cosine) to find semantically related memories."
    }

    fn read_only() -> bool {
        true
    }

    async fn execute(&self, args: Self::Args) -> Result<Value, McpError> {
        let start = Instant::now();

        // Get coordinator for specified library
        let coordinator = self.pool.get_coordinator(&args.library)
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Failed to get coordinator for library '{}': {}", args.library, e)))?;

        // Create filter WITHOUT library tag (library already selected via coordinator)
        let filter = MemoryFilter::new();

        // Search using coordinator's public API
        let results = coordinator
            .search_memories(&args.context, args.limit, Some(filter))
            .await
            .map_err(|e| McpError::Other(anyhow::anyhow!("Search failed: {}", e)))?;

        // Convert to simplified format with 4 core fields: similarity, importance, score, rank
        let memories: Vec<Value> = results
            .into_iter()
            .enumerate()
            .map(|(index, memory)| {
                // Extract similarity (raw cosine) from metadata.custom
                let similarity = memory.metadata.custom
                    .get("similarity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;

                // Get importance (boosted by entanglement/quality in coordinator)
                let importance = memory.importance();

                // Calculate score = similarity × importance
                let score = similarity * importance;

                // Rank is 1-indexed position in already-sorted results
                let rank = index + 1;

                json!({
                    "id": memory.id(),
                    "content": memory.content().to_string(),
                    "created_at": memory.creation_time(),
                    "similarity": similarity,
                    "importance": importance,
                    "score": score,
                    "rank": rank
                })
            })
            .collect();

        let count = memories.len();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(json!({
            "memories": memories,
            "library": args.library,
            "count": count,
            "elapsed_ms": elapsed_ms
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
                    "How do I use the recall tool to retrieve relevant memories using semantic search?",
                ),
            },
            PromptMessage {
                role: PromptMessageRole::Assistant,
                content: PromptMessageContent::text(
                    "The recall tool retrieves memories from a specific library using semantic search. \
                     Searches are scoped to one library (database file) at a time.\n\n\
                     Basic usage:\n\
                     1. Search work library: recall({\"library\": \"work\", \"context\": \"authentication\", \"limit\": 5})\n\
                     2. Search personal library: recall({\"library\": \"personal\", \"context\": \"recipes\", \"limit\": 3})\n\
                     3. Search project library: recall({\"library\": \"project_x\", \"context\": \"API design\", \"limit\": 10})\n\n\
                     Library scoping:\n\
                     - Results come ONLY from the specified library's database file\n\
                     - Memories in other libraries are NOT searched\n\
                     - To search multiple libraries, make multiple recall() calls\n\n\
                     Semantic search capability:\n\
                     - Finds conceptually similar content, not just keyword matches\n\
                     - Uses 1024-dimensional vector embeddings\n\
                     - Results ranked by relevance_score\n\n\
                     Response format:\n\
                     {\n\
                       \"memories\": [{\"id\": \"...\", \"content\": \"...\", \"relevance_score\": 0.85}],\n\
                       \"library\": \"work\",\n\
                       \"count\": 3\n\
                     }\n\n\
                     Parameters:\n\
                     - library: Which database file to search (required)\n\
                     - context: Your search query (required)\n\
                     - limit: Maximum results (optional, default: 10)",
                ),
            },
        ])
    }
}
