//! Recall Tool - Retrieve relevant memories from a library using semantic search

use kodegen_mcp_schema::{Tool, ToolExecutionContext, ToolResponse, McpError};
use kodegen_mcp_schema::memory::{RecallArgs, RecallOutput, RecalledMemory, MEMORY_RECALL, MemoryRecallPrompts};
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
    type Prompts = MemoryRecallPrompts;

    fn name() -> &'static str {
        MEMORY_RECALL
    }

    fn description() -> &'static str {
        "Retrieve relevant memories from a library using semantic search. \
         Searches for content similar to the provided context and returns the most relevant results. \
         Uses vector similarity (cosine) to find semantically related memories."
    }

    fn read_only() -> bool {
        true
    }

    async fn execute(&self, args: Self::Args, _ctx: ToolExecutionContext) -> Result<ToolResponse<<Self::Args as kodegen_mcp_schema::ToolArgs>::Output>, McpError> {
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

        // Convert to typed RecalledMemory structs
        let memories: Vec<RecalledMemory> = results
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

                RecalledMemory {
                    id: memory.id().to_string(),
                    content: memory.content().to_string(),
                    created_at: memory.creation_time().to_string(),
                    similarity,
                    importance,
                    score,
                    rank,
                }
            })
            .collect();

        let count = memories.len();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Terminal summary
        let summary = if memories.is_empty() {
            format!(
                "✓ No memories found\n\n\
                 Library: {}\n\
                 Query: {}\n\
                 Search time: {:.0}ms",
                args.library, args.context, elapsed_ms
            )
        } else {
            let top_results = memories.iter()
                .take(5)
                .enumerate()
                .map(|(i, m)| {
                    // Truncate content to 50 chars
                    let truncated = if m.content.len() > 50 {
                        format!("{}...", &m.content[..50])
                    } else {
                        m.content.clone()
                    };
                    format!("  {}. [{:.2}] {}", i + 1, m.similarity, truncated)
                })
                .collect::<Vec<_>>()
                .join("\n");

            format!(
                "✓ Memories recalled ({} results)\n\n\
                 Library: {}\n\
                 Search time: {:.0}ms\n\n\
                 Top results:\n{}",
                count, args.library, elapsed_ms, top_results
            )
        };

        Ok(ToolResponse::new(summary, RecallOutput {
            memories,
            library: args.library,
            count,
            elapsed_ms,
        }))
    }

}
