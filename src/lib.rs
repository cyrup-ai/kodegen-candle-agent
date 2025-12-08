//! Fluent AI Candle Library
#![recursion_limit = "256"]
#![feature(impl_trait_in_fn_trait_return)]
#![feature(impl_trait_in_assoc_type)]
#![feature(negative_impls)]
#![feature(auto_traits)]
#![feature(fn_traits)]

//! This crate provides Candle ML framework integration for AI services.
//! All Candle-prefixed domain types, builders, and providers are defined here
//! to ensure complete independence from the main cyrup packages.

// Initialize performance optimizations on library load
use std::sync::Once;
static INIT: Once = Once::new();

/// Initialize library-wide performance optimizations
pub fn init_candle() {
    INIT.call_once(|| {
        // Initialize timestamp caching for high-performance operations
        domain::memory::cache::initialize_timestamp_cache();

        // Initialize memory node pool for zero-allocation memory operations
        // Using 1000 nodes with 768-dimensional embeddings (typical for BERT-base)
        domain::memory::pool::initialize_memory_node_pool(1000, 768);

        // Force initialization of static model registries (LazyLock)
        // This ensures static models are available in test contexts
        let _ = capability::registry::model_count();
    });
}

pub mod macros;

// Candle-specific modules (minimal set for core functionality)
/// Async stream utilities using tokio streams
pub mod async_stream;
/// Candle builders for zero-allocation construction patterns
pub mod builders;
/// Candle macros for ARCHITECTURE.md syntax support
/// Candle capabilities organized by what models can do
pub mod capability;
/// CLI module for interactive chat applications
pub mod cli;
/// Chat functionality is now available through domain::chat
/// Core components (engine, generation, etc.)
pub mod core;
/// Candle domain types (replaces cyrup_domain dependency)
pub mod domain;
/// Extension integration for Raycast and Alfred (macOS)
pub mod extensions;
/// Image processing utilities
pub mod image;
/// Memory system with cognitive features and vector storage
pub mod memory;
/// MCP tools for memory operations
pub mod tools;
/// Prompt processing utilities
pub mod prompt;
/// Shared Tokio runtime for avoiding multiple runtime creation
pub mod runtime;
/// Utility modules for common operations
pub mod util;
/// Real workflow execution system with streams-only architecture
pub mod workflow;

// Essential Candle re-exports for public API (minimal set)
// Domain types will be added as they become available

// Prelude - All types needed for ARCHITECTURE.md syntax
pub mod prelude {
    pub use crate::builders::{CandleAgentBuilder, CandleAgentRoleBuilder, CandleFluentAi};
    // Vision builder for image description
    pub use crate::builders::CandleVisionBuilder;
    // Embedding builder for text embeddings
    pub use crate::builders::EmbeddingBuilder;
    pub use crate::domain::Embedding;
    // Re-export generation types from modular structure
    pub use crate::core::generation::{
        CandleLlamaModel, CandleModel, GenerationStatistics, SamplingConfig, SimdMetrics,
        SpecialTokens, TextGenerator, TokenHistory,
    };
    pub use crate::core::{
        Engine, EngineConfig, EngineError, EngineResult, ModelArchitecture, ModelConfig,
        ModelConfigError,
    };
    pub use crate::domain::chat::CandleChatLoop;
    pub use crate::domain::chat::message::CandleMessageChunk;
    pub use crate::domain::{
        agent::CandleAgent,
        chat::message::types::CandleMessageRole,
        context::{
            FinishReason,
            chunks::CandleStringChunk,
            provider::{CandleContext, CandleDirectory, CandleFile, CandleFiles, CandleGithub},
        },
        image_generation::{
            ImageGenerationChunk, ImageGenerationConfig, ImageGenerationModel, tensor_to_image,
        },
        tool::{CandleToolRouter, CyloBackendConfig, RouterError},
    };

    // Re-export workspace MCP types for convenience
    pub use kodegen_mcp_client::KodegenClient;
    pub use kodegen_mcp_schema::Tool;
    pub use rmcp::model::Tool as ToolInfo;

    // Real workflow execution types - streams-only architecture
    pub use crate::workflow::{CandleExecutableWorkflow, CandleWorkflowStep, candle_workflow};

    // Pool infrastructure for transparent worker management
    pub use crate::capability::registry::pool::{Pool, PoolError, init_maintenance};

    pub struct CandleLibrary;

    impl CandleLibrary {
        pub fn named(_name: &str) -> Self {
            Self
        }
    }

    // Re-export tool implementation that provides static methods

    // Helper function for ARCHITECTURE.md example
    pub fn process_turn() -> CandleChatLoop {
        CandleChatLoop::Reprompt("continue".to_string())
    }
}

// Re-export everything from prelude at root level for convenience
// Re-export tokio_stream for convenience
pub use tokio_stream::{Stream, StreamExt};

// Re-export our stream utilities
pub use crate::async_stream::{empty, from_iter, once, spawn_stream};
// SIMD operations from cyrup-simd for high-performance ML workloads
pub use kodegen_simd;
pub use prelude::*;

// Pool infrastructure (part of registry)
pub use capability::registry::pool::{Pool, PoolError, init_maintenance};

// ============================================================================
// EMBEDDED SERVER FUNCTION
// ============================================================================

/// Start the candle-agent HTTP server programmatically for embedded mode
///
/// Returns a ServerHandle for graceful shutdown control.
/// This function is non-blocking - the server runs in background tasks.
///
/// # Arguments
/// * `addr` - Socket address to bind to (e.g., "127.0.0.1:30441")
/// * `tls_cert` - Optional path to TLS certificate file
/// * `tls_key` - Optional path to TLS private key file
///
/// # Returns
/// ServerHandle for graceful shutdown, or error if startup fails
pub async fn start_server(
    addr: std::net::SocketAddr,
    tls_cert: Option<std::path::PathBuf>,
    tls_key: Option<std::path::PathBuf>,
) -> anyhow::Result<kodegen_server_http::ServerHandle> {
    // Bind to the address first
    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    // Convert separate cert/key into Option<(cert, key)> tuple
    let tls_config = match (tls_cert, tls_key) {
        (Some(cert), Some(key)) => Some((cert, key)),
        _ => None,
    };

    // Delegate to start_server_with_listener
    start_server_with_listener(listener, tls_config).await
}

/// Start candle-agent HTTP server using pre-bound listener (TOCTOU-safe)
///
/// This variant is used by kodegend to eliminate TOCTOU race conditions
/// during port cleanup. The listener is already bound to a port.
///
/// # Arguments
/// * `listener` - Pre-bound TcpListener (port already reserved)
/// * `tls_config` - Optional (cert_path, key_path) for HTTPS
///
/// # Returns
/// ServerHandle for graceful shutdown, or error if startup fails
pub async fn start_server_with_listener(
    listener: tokio::net::TcpListener,
    tls_config: Option<(std::path::PathBuf, std::path::PathBuf)>,
) -> anyhow::Result<kodegen_server_http::ServerHandle> {
    use kodegen_server_http::{ServerBuilder, Managers, RouterSet, register_tool};
    use rmcp::handler::server::router::{prompt::PromptRouter, tool::ToolRouter};

    let mut builder = ServerBuilder::new()
        .category(kodegen_config::CATEGORY_CANDLE_AGENT)
        .register_tools(|| async {
            // Initialize CoordinatorPool (retrieves model from lazy registry, creates empty pool)
            let pool = initialize_coordinator_pool().await?;

            let mut tool_router = ToolRouter::new();
            let mut prompt_router = PromptRouter::new();
            let managers = Managers::new();

            // Create memorize session manager
            let memorize_manager = std::sync::Arc::new(crate::tools::MemorizeSessionManager::new(pool.clone()));

            // Register memory tools (4 tools)
            (tool_router, prompt_router) = register_tool(
                tool_router,
                prompt_router,
                crate::tools::MemorizeTool::new(memorize_manager.clone()),
            );

            (tool_router, prompt_router) = register_tool(
                tool_router,
                prompt_router,
                crate::tools::CheckMemorizeStatusTool::new(memorize_manager.clone()),
            );

            (tool_router, prompt_router) = register_tool(
                tool_router,
                prompt_router,
                crate::tools::RecallTool::new(pool.clone()),
            );

            (tool_router, prompt_router) = register_tool(
                tool_router,
                prompt_router,
                crate::tools::ListMemoryLibrariesTool::new(pool.clone()),
            );

            // Start cleanup task for memorize sessions
            memorize_manager.start_cleanup_task();

            Ok(RouterSet::new(tool_router, prompt_router, managers))
        })
        .with_listener(listener);

    if let Some((cert, key)) = tls_config {
        builder = builder.with_tls_config(cert, key);
    }

    builder.serve().await
}

// Helper function for pool initialization
async fn initialize_coordinator_pool() -> anyhow::Result<std::sync::Arc<crate::memory::core::manager::pool::CoordinatorPool>> {
    use crate::capability::registry::FromRegistry;
    use crate::capability::registry::TextEmbeddingModel;
    
    // Get embedding model from lazy static registry (Stella 400M variant)
    let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5")
        .ok_or_else(|| anyhow::anyhow!("Stella embedding model not found in registry"))?;

    // Create empty coordinator pool - coordinators created lazily per library
    let pool = crate::memory::core::manager::pool::CoordinatorPool::new(emb_model);

    Ok(std::sync::Arc::new(pool))
}
