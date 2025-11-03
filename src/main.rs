//! Candle-Agent Category HTTP Server
//!
//! Serves memory tools via HTTP/HTTPS transport using kodegen_server_http.

use anyhow::{Result, anyhow};
use kodegen_server_http::{run_http_server, Managers, RouterSet, register_tool};
use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};
use std::sync::Arc;

use kodegen_candle_agent::capability::registry::TextEmbeddingModel;
use kodegen_candle_agent::memory::core::manager::pool::CoordinatorPool;
use kodegen_candle_agent::tools::{
    MemorizeTool, MemorizeSessionManager, CheckMemorizeStatusTool,
    RecallTool, ListMemoryLibrariesTool
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize CoordinatorPool before run_http_server (async initialization)
    let pool = initialize_coordinator_pool().await?;

    run_http_server("candle-agent", move |_config, _tracker| {
        let mut tool_router = ToolRouter::new();
        let mut prompt_router = PromptRouter::new();
        let managers = Managers::new();

        // Create memorize session manager
        let memorize_manager = Arc::new(MemorizeSessionManager::new(pool.clone()));

        // Register memory tools (4 tools now - memorize and check_memorize_status use manager)
        (tool_router, prompt_router) = register_tool(
            tool_router,
            prompt_router,
            MemorizeTool::new(memorize_manager.clone()),
        );

        (tool_router, prompt_router) = register_tool(
            tool_router,
            prompt_router,
            CheckMemorizeStatusTool::new(memorize_manager.clone()),
        );

        (tool_router, prompt_router) = register_tool(
            tool_router,
            prompt_router,
            RecallTool::new(pool.clone()),
        );

        (tool_router, prompt_router) = register_tool(
            tool_router,
            prompt_router,
            ListMemoryLibrariesTool::new(pool.clone()),
        );

        // CRITICAL: Start cleanup task after all tools registered
        memorize_manager.start_cleanup_task();

        Ok(RouterSet::new(tool_router, prompt_router, managers))
    })
    .await
}

async fn initialize_coordinator_pool() -> Result<Arc<CoordinatorPool>> {
    // Get embedding model from registry (Stella 400M variant - registered by default)
    use kodegen_candle_agent::capability::registry::FromRegistry;
    let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5")
        .ok_or_else(|| anyhow!("Stella embedding model not found in registry"))?;

    // Create coordinator pool - coordinators created lazily per library
    let pool = CoordinatorPool::new(emb_model);

    Ok(Arc::new(pool))
}
