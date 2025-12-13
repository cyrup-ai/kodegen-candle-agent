//! # Candle Agent Memory Tools Example
//!
//! Demonstrates async memory operations with the kodegen-candle-agent server:
//! - Async `memorize` operations with session-based polling
//! - Progress tracking during embedding generation
//! - Listing all memory libraries
//! - Semantic similarity search with `recall`
//! - Content deduplication via SHA-256 hash matching
//! - Multi-library organization
//!
//! ## Common Module Dependency
//!
//! This example requires the `common` module located at:
//! - **Path:** `examples/common/mod.rs`
//! - **Size:** ~12KB of shared utility code
//! - **Purpose:** HTTP server lifecycle management and logging
//!
//! ### API Surface Required from `common`
//!
//! ```ignore
//! /// Spawns kodegen-candle-agent HTTP server and connects to it
//! /// Returns: (KodegenConnection, ServerHandle)
//! async fn connect_to_local_http_server() -> Result<(KodegenConnection, ServerHandle)>;
//!
//! /// HTTP client wrapper that logs all requests/responses to JSONL file
//! /// Useful for debugging and understanding MCP protocol
//! struct LoggingClient {
//!     fn new(client: KodegenClient, log_path: impl AsRef<Path>) -> Result<Self>;
//!     async fn call_tool(&self, name: &str, args: Value) -> Result<CallToolResult>;
//!     async fn call_tool_typed<T: DeserializeOwned>(&self, name: &str, args: Value) -> Result<T>;
//! }
//!
//! /// Finds workspace root using `cargo metadata`
//! /// Returns: Static reference to cached PathBuf
//! fn find_workspace_root() -> Result<&'static PathBuf>;
//!
//! /// Server process handle with graceful shutdown
//! struct ServerHandle {
//!     async fn shutdown(&mut self) -> Result<()>;
//! }
//! ```
//!
//! ### Why Common Module Exists
//!
//! The `common` module is shared infrastructure for all candle-agent HTTP server examples.
//! It handles:
//! - **Server spawning:** Uses `cargo run` to compile and start the HTTP server
//! - **Connection retry:** Polls server until ready (up to 5 minutes for first compile)
//! - **Port cleanup:** Kills existing processes on the port before starting
//! - **Request logging:** Captures all MCP tool calls to `tmp/mcp-client/memory.log`
//! - **Graceful shutdown:** Sends SIGTERM and waits for clean exit
//!
//! This infrastructure is non-trivial (12KB) and would be duplicated across examples
//! if inlined. Keeping it in a shared module follows DRY principles.
//!
//! ## Running This Example
//!
//! ```bash
//! # From workspace root
//! cd /Volumes/samsung_t9/kodegen-workspace
//!
//! # Run the example (will auto-start the HTTP server)
//! cargo run --example candle_agent_demo
//!
//! # The common module will:
//! # 1. Spawn kodegen-candle-agent HTTP server on port 20438
//! # 2. Wait for server to be ready (compiles on first run)
//! # 3. Connect via HTTP and run the example
//! # 4. Shut down server gracefully when done
//! ```
//!
//! ## Copying This Example
//!
//! If you want to use this example in your own project:
//!
//! 1. **Copy both files:**
//!    - `examples/candle_agent_demo.rs`
//!    - `examples/common/mod.rs`
//!
//! 2. **Add dependencies to `Cargo.toml`:**
//!    ```toml
//!    [dev-dependencies]
//!    kodegen-mcp-client = "0.1"
//!    kodegen-config = "0.1"
//!    anyhow = "1.0"
//!    serde = { version = "1.0", features = ["derive"] }
//!    serde_json = "1.0"
//!    tokio = { version = "1.0", features = ["full"] }
//!    tracing = "0.1"
//!    tracing-subscriber = "0.3"
//!    ```
//!
//! 3. **Ensure kodegen-candle-agent binary is available:**
//!    - Either build it in your workspace
//!    - Or modify `common/mod.rs` to point to the binary location
//!
//! ## Log Output
//!
//! All HTTP requests and responses are logged to:
//! ```
//! tmp/mcp-client/memory.log
//! ```
//!
//! This JSONL file contains:
//! - Timestamp
//! - Tool name (e.g., "memory_memorize")
//! - Request arguments
//! - Response data
//! - Duration in milliseconds
//!
//! Useful for debugging MCP protocol issues or understanding the async flow.

mod common;

use anyhow::Context;
use clap::Parser;
use kodegen_config::KodegenConfig;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing::{info, error};

/// Command-line arguments for the candle agent demo
#[derive(Parser, Debug)]
#[command(author, version, about = "Candle agent memory tools demonstration")]
struct Args {
    /// Clean up databases and logs from previous runs before starting
    /// 
    /// By default, the demo preserves existing data for safety.
    /// Use this flag to start with a clean slate for reproducible testing.
    #[arg(long, default_value = "false")]
    clean: bool,
}

#[derive(serde::Deserialize, Debug)]
struct MemorizeResponse {
    session_id: String,
    #[serde(rename = "status")]
    _status: String,
    library: String,
    #[serde(rename = "message")]
    _message: String,
}

#[derive(serde::Deserialize, Debug)]
struct CheckMemorizeStatusResponse {
    #[serde(rename = "session_id")]
    _session_id: String,
    status: String,
    memory_id: Option<String>,
    #[serde(rename = "library")]
    _library: String,
    progress: MemorizeProgress,
    runtime_ms: u64,
    error: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct MemorizeProgress {
    stage: String,
    #[serde(rename = "files_loaded")]
    _files_loaded: usize,
    #[serde(rename = "total_size_bytes")]
    _total_size_bytes: usize,
}

#[derive(serde::Deserialize, Debug)]
struct ListLibrariesResponse {
    libraries: Vec<String>,
    count: usize,
}

#[derive(serde::Deserialize, Debug)]
struct Memory {
    id: String,
    content: String,
    #[allow(dead_code)]
    created_at: String,
    /// Raw cosine similarity (0.0-1.0) - shows pure vector match quality
    similarity: f64,
    /// Boosted importance (includes entanglement/quality boosts)
    importance: f64,
    /// Final ranking score (similarity × importance)
    score: f64,
    /// Position in ranked results (1 = best)
    rank: usize,
}

#[derive(serde::Deserialize, Debug)]
struct RecallResponse {
    memories: Vec<Memory>,
    library: String,
    count: usize,
}

/// Configuration for async operation polling behavior
#[derive(Debug, Clone)]
struct PollingConfig {
    /// Initial polling interval (first attempt)
    initial_interval: std::time::Duration,
    /// Maximum polling interval (backoff cap)
    max_interval: std::time::Duration,
    /// Total timeout duration
    max_duration: std::time::Duration,
    /// Enable exponential backoff
    enable_backoff: bool,
    /// Log progress every N attempts
    progress_log_interval: u32,
}

impl Default for PollingConfig {
    fn default() -> Self {
        Self {
            initial_interval: std::time::Duration::from_millis(200),  // Start fast
            max_interval: std::time::Duration::from_secs(2),          // Cap at 2s
            max_duration: std::time::Duration::from_secs(60),         // 60s total
            enable_backoff: true,                          // Adaptive polling
            progress_log_interval: 4,                      // Log every 4 attempts
        }
    }
}

impl PollingConfig {
    /// Create configuration from environment variables
    /// 
    /// Supported environment variables:
    /// - `KODEGEN_POLL_INITIAL_MS`: Initial polling interval in milliseconds (default: 200)
    /// - `KODEGEN_POLL_MAX_MS`: Maximum polling interval in milliseconds (default: 2000)
    /// - `KODEGEN_POLL_TIMEOUT_SECS`: Total timeout in seconds (default: 60)
    /// - `KODEGEN_POLL_BACKOFF`: Enable exponential backoff (default: true)
    fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("KODEGEN_POLL_INITIAL_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                config.initial_interval = std::time::Duration::from_millis(ms);
            }
        }

        if let Ok(val) = std::env::var("KODEGEN_POLL_MAX_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                config.max_interval = std::time::Duration::from_millis(ms);
            }
        }

        if let Ok(val) = std::env::var("KODEGEN_POLL_TIMEOUT_SECS") {
            if let Ok(secs) = val.parse::<u64>() {
                config.max_duration = std::time::Duration::from_secs(secs);
            }
        }

        if let Ok(val) = std::env::var("KODEGEN_POLL_BACKOFF") {
            if let Ok(enable) = val.parse::<bool>() {
                config.enable_backoff = enable;
            }
        }

        config
    }
}

/// Helper to wait for async memorize operation to complete
async fn wait_for_memorize_completion(
    client: &common::LoggingClient,
    session_id: &str,
    config: &PollingConfig,
    cancel_token: CancellationToken,
) -> anyhow::Result<String> {
    // Exponential backoff configuration
    let mut current_interval = config.initial_interval;
    let max_interval = config.max_interval;
    let max_duration = config.max_duration;
    let start = tokio::time::Instant::now();
    let mut attempt = 0;

    loop {
        attempt += 1;
        
        // Check for cancellation at start of each iteration
        if cancel_token.is_cancelled() {
            return Err(anyhow::anyhow!(
                "Operation cancelled by user (session_id: {})",
                session_id
            ));
        }

        // Check for absolute timeout
        if start.elapsed() > max_duration {
            let elapsed_secs = start.elapsed().as_secs();
            
            error!(
                session_id = %session_id,
                elapsed_secs = elapsed_secs,
                max_duration_secs = max_duration.as_secs(),
                "Timeout waiting for memorize completion"
            );
            
            return Err(anyhow::anyhow!(
                "Timeout waiting for memorize completion after {:?} ({} attempts)",
                config.max_duration,
                attempt
            ));
        }

        // Poll status
        let status: CheckMemorizeStatusResponse = client
            .call_tool_typed(
                kodegen_config::MEMORY_CHECK_MEMORIZE_STATUS,
                json!({ "session_id": session_id }),
            )
            .await
            .with_context(|| {
                format!(
                    "Failed to check memorize status (session_id: {}, elapsed: {}ms)",
                    session_id,
                    start.elapsed().as_millis()
                )
            })?;

        match status.status.as_str() {
            "COMPLETED" => {
                return status.memory_id.ok_or_else(|| {
                    error!(
                        session_id = %session_id,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        runtime_ms = status.runtime_ms,
                        "Status COMPLETED but no memory_id returned"
                    );
                    anyhow::anyhow!(
                        "Status COMPLETED but no memory_id returned (session_id: {}, elapsed: {}ms, runtime: {}ms)",
                        session_id,
                        start.elapsed().as_millis(),
                        status.runtime_ms
                    )
                });
            }
            "FAILED" => {
                let error_msg = status.error.as_deref().unwrap_or("Unknown error");
                
                error!(
                    session_id = %session_id,
                    error_msg = %error_msg,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    runtime_ms = status.runtime_ms,
                    stage = %status.progress.stage,
                    "Memorize operation failed"
                );
                
                return Err(anyhow::anyhow!(
                    "Memorize failed: {} (session_id: {}, elapsed: {}ms, runtime: {}ms, stage: {})",
                    error_msg,
                    session_id,
                    start.elapsed().as_millis(),
                    status.runtime_ms,
                    status.progress.stage
                ));
            }
            "IN_PROGRESS" => {
                // Log progress every N attempts
                if attempt % config.progress_log_interval == 0 {
                    info!(
                        "      ⏳ Still processing... Stage: {}, Runtime: {}ms (session_id: {}, elapsed: {}ms, attempt: {})",
                        status.progress.stage, 
                        status.runtime_ms,
                        session_id,
                        start.elapsed().as_millis(),
                        attempt
                    );
                }
                
                // Sleep with cancellation support using tokio::select!
                tokio::select! {
                    _ = tokio::time::sleep(current_interval) => {
                        // Sleep completed normally, continue loop
                    },
                    _ = cancel_token.cancelled() => {
                        // Cancellation requested during sleep
                        return Err(anyhow::anyhow!(
                            "Operation cancelled during polling (session_id: {})",
                            session_id
                        ));
                    }
                }
                
                // Exponential backoff: double the interval, cap at max_interval (if enabled)
                if config.enable_backoff {
                    current_interval = (current_interval * 2).min(max_interval);
                }
            }
            other => {
                // Structured log for machine-readable aggregation
                error!(
                    session_id = %session_id,
                    status = %other,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    runtime_ms = status.runtime_ms,
                    stage = %status.progress.stage,
                    "Unknown memorize status encountered during polling"
                );
                
                return Err(anyhow::anyhow!(
                    "Unknown status '{}' (session_id: {}, elapsed: {}ms, runtime: {}ms, stage: {})",
                    other,
                    session_id,
                    start.elapsed().as_millis(),
                    status.runtime_ms,
                    status.progress.stage
                ));
            }
        }
    }
}

/// Clean up database files from previous runs
/// 
/// SurrealKV databases are stored as directories, so we use remove_dir_all.
fn cleanup_databases(memory_dir: &std::path::Path) -> anyhow::Result<()> {
    let db_files = ["rust_patterns.db", "debugging_insights.db"];
    
    for db_file in &db_files {
        let db_path = memory_dir.join(db_file);
        if db_path.exists() {
            std::fs::remove_dir_all(&db_path)
                .with_context(|| format!("Failed to delete {}", db_path.display()))?;
            info!("🗑️  Cleaned up previous database: {}", db_file);
        }
    }
    
    Ok(())
}

/// Clean up log file from previous runs
fn cleanup_log_file(log_path: &std::path::Path) -> anyhow::Result<()> {
    if log_path.exists() {
        std::fs::remove_file(log_path)
            .with_context(|| format!("Failed to delete {}", log_path.display()))?;
        info!("🗑️  Cleaned up previous log file: {}", log_path.display());
    }
    
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command-line arguments
    let args = Args::parse();
    
    // Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    info!("Starting candle-agent memory tools example (async session pattern)");

    // Create cancellation token for graceful shutdown
    let cancel_token = CancellationToken::new();
    
    // Spawn Ctrl+C signal handler (follows pattern from packages/kodegen/src/main.rs)
    let signal_token = cancel_token.clone();
    tokio::spawn(async move {
        if let Err(e) = tokio::signal::ctrl_c().await {
            eprintln!("Failed to listen for Ctrl+C: {}", e);
            return;
        }
        info!("\n🛑 Ctrl+C received - cancelling operations...");
        signal_token.cancel();
    });

    // Prepare database directory
    let memory_dir = KodegenConfig::data_dir()
        .context("Failed to get data directory")?
        .join("memory");

    // Clean up databases only if --clean flag is provided
    if args.clean {
        info!("🧹 Cleanup mode enabled (--clean flag provided)");
        cleanup_databases(&memory_dir)?;
    } else {
        info!("💾 Preserving existing data (use --clean to start fresh)");
    }

    // Connect to kodegen-candle-agent server
    let (conn, mut server) = common::connect_to_local_http_server().await?;

    // Wrap client with logging
    let workspace_root = common::find_workspace_root()
        .context("Failed to find workspace root")?;
    let log_path = workspace_root.join("tmp/mcp-client/memory.log");

    // Clean up log file only if --clean flag is provided
    if args.clean {
        cleanup_log_file(&log_path)?;
    }

    let client = common::LoggingClient::new(conn.client(), log_path)
        .await
        .context("Failed to create logging client")?;

    info!("Connected to server: {:?}", client.server_info());

    // Run example with cancellation support wrapped in tokio::select!
    let result = tokio::select! {
        res = run_memory_example(&client, cancel_token.clone()) => res,
        _ = cancel_token.cancelled() => {
            info!("Example cancelled by user");
            Err(anyhow::anyhow!("Operation cancelled by user"))
        }
    };

    // Always attempt cleanup, collecting all errors
    let mut cleanup_errors = Vec::new();

    // Try to shutdown logging client
    if let Err(e) = client.shutdown().await {
        let error_msg = format!("Failed to shutdown logging client: {:#}", e);
        tracing::error!("{}", error_msg);
        cleanup_errors.push(error_msg);
    }

    // Try to close connection
    if let Err(e) = conn.close().await {
        let error_msg = format!("Failed to close connection: {:#}", e);
        tracing::error!("{}", error_msg);
        cleanup_errors.push(error_msg);
    }

    // Try to shutdown server
    if let Err(e) = server.shutdown().await {
        let error_msg = format!("Failed to shutdown server: {:#}", e);
        tracing::error!("{}", error_msg);
        cleanup_errors.push(error_msg);
    }

    // Combine results: preserve original error, note cleanup failures
    if !cleanup_errors.is_empty() {
        // If example also failed, preserve that error with cleanup context
        if let Err(example_err) = result {
            return Err(example_err.context(format!(
                "Example failed, and cleanup also had {} error(s): {}", 
                cleanup_errors.len(),
                cleanup_errors.join("; ")
            )));
        }
        
        // Example succeeded but cleanup failed
        return Err(anyhow::anyhow!(
            "Example succeeded but cleanup failed: {}", 
            cleanup_errors.join("; ")
        ));
    }

    // Both example and cleanup succeeded
    result
}

async fn run_memory_example(
    client: &common::LoggingClient,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    info!("========================================");
    info!("  Memory Tools Demonstration");
    info!("  (Async Session Pattern)");
    info!("========================================\n");

    // Create polling configuration from environment variables
    let config = PollingConfig::from_env();

    // ========================================================================
    // PHASE 1: Create memories in two different libraries (concurrent)
    // ========================================================================
    info!("PHASE 1: Creating memories in two libraries (concurrent)\n");

    // Start all operations concurrently using tokio::try_join!
    let (mem1_id, mem2_id, mem3_id, mem4_id) = tokio::try_join!(
        // Memory 1: Rust pattern #1
        async {
            info!("1. Storing Rust pattern #1");
            let session: MemorizeResponse = client
                .call_tool_typed(
                    kodegen_config::MEMORY_MEMORIZE,
                    json!({
                        "library": "rust_patterns",
                        "content": "Error handling pattern using Result<T, E> with the ? operator for clean propagation"
                    }),
                )
                .await
                .context("Failed to start memorize #1")?;
            info!("   → Session started: {}", session.session_id);
            let mem_id = wait_for_memorize_completion(client, &session.session_id, &config, cancel_token.clone())
                .await
                .with_context(|| format!("Failed to complete memorize for session '{}'", session.session_id))?;
            info!("   ✅ Created memory: {}", mem_id);
            Ok::<String, anyhow::Error>(mem_id)
        },
        
        // Memory 2: Rust pattern #2
        async {
            info!("2. Storing Rust pattern #2");
            let session: MemorizeResponse = client
                .call_tool_typed(
                    kodegen_config::MEMORY_MEMORIZE,
                    json!({
                        "library": "rust_patterns",
                        "content": "Async/await pattern for file I/O operations using tokio::fs with proper error handling"
                    }),
                )
                .await
                .context("Failed to start memorize #2")?;
            info!("   → Session started: {}", session.session_id);
            let mem_id = wait_for_memorize_completion(client, &session.session_id, &config, cancel_token.clone())
                .await
                .with_context(|| format!("Failed to complete memorize for session '{}'", session.session_id))?;
            info!("   ✅ Created memory: {}", mem_id);
            Ok::<String, anyhow::Error>(mem_id)
        },
        
        // Memory 3: Debugging insight #1
        async {
            info!("3. Storing debugging insight #1");
            let session: MemorizeResponse = client
                .call_tool_typed(
                    kodegen_config::MEMORY_MEMORIZE,
                    json!({
                        "library": "debugging_insights",
                        "content": "React re-renders happen when props or state change - use React.memo to prevent unnecessary renders"
                    }),
                )
                .await
                .context("Failed to start memorize #3")?;
            info!("   → Session started: {}", session.session_id);
            let mem_id = wait_for_memorize_completion(client, &session.session_id, &config, cancel_token.clone())
                .await
                .with_context(|| format!("Failed to complete memorize for session '{}'", session.session_id))?;
            info!("   ✅ Created memory: {}", mem_id);
            Ok::<String, anyhow::Error>(mem_id)
        },
        
        // Memory 4: Debugging insight #2
        async {
            info!("4. Storing debugging insight #2");
            let session: MemorizeResponse = client
                .call_tool_typed(
                    kodegen_config::MEMORY_MEMORIZE,
                    json!({
                        "library": "debugging_insights",
                        "content": "SQL N+1 query problem - use eager loading with JOIN instead of lazy loading to reduce DB calls"
                    }),
                )
                .await
                .context("Failed to start memorize #4")?;
            info!("   → Session started: {}", session.session_id);
            let mem_id = wait_for_memorize_completion(client, &session.session_id, &config, cancel_token.clone())
                .await
                .with_context(|| format!("Failed to complete memorize for session '{}'", session.session_id))?;
            info!("   ✅ Created memory: {}", mem_id);
            Ok::<String, anyhow::Error>(mem_id)
        },
    )?;

    info!("\n✅ All 4 memories created concurrently:");
    info!("   rust_patterns[1]: {}", mem1_id);
    info!("   rust_patterns[2]: {}", mem2_id);
    info!("   debugging_insights[1]: {}", mem3_id);
    info!("   debugging_insights[2]: {}", mem4_id);

    // ========================================================================
    // PHASE 2: List all libraries
    // ========================================================================
    info!("\nPHASE 2: Listing all memory libraries\n");

    info!("5. Calling list_memory_libraries()");
    let libraries: ListLibrariesResponse = client
        .call_tool_typed(kodegen_config::MEMORY_LIST_LIBRARIES, json!({}))
        .await
        .context("Failed to list memory libraries")?;

    info!("   ✅ Found {} libraries:", libraries.count);
    for lib in &libraries.libraries {
        info!("      - {}", lib);
    }

    // ========================================================================
    // PHASE 3: Recall from each library (semantic search)
    // ========================================================================
    info!("\nPHASE 3: Recalling memories using semantic search\n");

    info!("6. Recalling from 'rust_patterns' (context: 'error handling')");
    let recall1: RecallResponse = client
        .call_tool_typed(
            kodegen_config::MEMORY_RECALL,
            json!({
                "library": "rust_patterns",
                "context": "error handling",
                "limit": 5
            }),
        )
        .await
        .context("Failed to recall from rust_patterns")?;

    info!("   ✅ Found {} memories in '{}':", recall1.count, recall1.library);
    for memory in &recall1.memories {
        info!("      #{} - ID: {}", memory.rank, memory.id);
        info!("        Similarity: {:.3} (raw cosine)", memory.similarity);
        info!("        Importance: {:.3} (boosted)", memory.importance);
        info!("        Score: {:.3} (similarity × importance)", memory.score);
        info!("        Content: {}", memory.content);
    }

    info!("\n7. Recalling from 'debugging_insights' (context: 'performance optimization')");
    let recall2: RecallResponse = client
        .call_tool_typed(
            kodegen_config::MEMORY_RECALL,
            json!({
                "library": "debugging_insights",
                "context": "performance optimization",
                "limit": 5
            }),
        )
        .await
        .context("Failed to recall from debugging_insights")?;

    info!("   ✅ Found {} memories in '{}':", recall2.count, recall2.library);
    for memory in &recall2.memories {
        info!("      #{} - ID: {}", memory.rank, memory.id);
        info!("        Similarity: {:.3} (raw cosine)", memory.similarity);
        info!("        Importance: {:.3} (boosted)", memory.importance);
        info!("        Score: {:.3} (similarity × importance)", memory.score);
        info!("        Content: {}", memory.content);
    }

    // ========================================================================
    // PHASE 4: Deduplication test - Store duplicate content (async)
    // ========================================================================
    info!("\nPHASE 4: Testing deduplication (duplicate content detection)\n");

    let duplicate_content = "This is a duplicate test string for deduplication verification";

    info!("8. Storing duplicate content in 'rust_patterns' (first time)");
    let dup_session1: MemorizeResponse = client
        .call_tool_typed(
            kodegen_config::MEMORY_MEMORIZE,
            json!({
                "library": "rust_patterns",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #1")?;
    info!("   → Session started: {}", dup_session1.session_id);
    let dup1_id = wait_for_memorize_completion(client, &dup_session1.session_id, &config, cancel_token.clone())
        .await
        .with_context(|| format!("Failed to complete memorize for session '{}'", dup_session1.session_id))?;
    info!("   ✅ First insertion - Memory ID: {}", dup1_id);

    info!("9. Storing SAME content in 'rust_patterns' (second time - should deduplicate)");
    let dup_session2: MemorizeResponse = client
        .call_tool_typed(
            kodegen_config::MEMORY_MEMORIZE,
            json!({
                "library": "rust_patterns",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #2")?;
    info!("   → Session started: {}", dup_session2.session_id);
    let dup2_id = wait_for_memorize_completion(client, &dup_session2.session_id, &config, cancel_token.clone())
        .await
        .with_context(|| format!("Failed to complete memorize for session '{}'", dup_session2.session_id))?;
    info!("   ✅ Second insertion - Memory ID: {}", dup2_id);

    if dup1_id == dup2_id {
        info!("\n   🎉 DEDUPLICATION VERIFIED:");
        info!("      Same memory_id returned: {}", dup1_id);
        info!("      Content hash matched - importance reset, but same entry preserved!");
    } else {
        info!("\n   ⚠️  Different memory IDs - deduplication may not have worked");
        info!("      First:  {}", dup1_id);
        info!("      Second: {}", dup2_id);
    }

    info!("\n10. Storing SAME content in 'debugging_insights' (different library)");
    let dup_session3: MemorizeResponse = client
        .call_tool_typed(
            kodegen_config::MEMORY_MEMORIZE,
            json!({
                "library": "debugging_insights",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #3")?;
    info!("   → Session started: {}", dup_session3.session_id);
    let dup3_id = wait_for_memorize_completion(client, &dup_session3.session_id, &config, cancel_token.clone())
        .await
        .with_context(|| format!("Failed to complete memorize for session '{}'", dup_session3.session_id))?;
    info!("   ✅ Third insertion (different library) - Memory ID: {}", dup3_id);

    if dup1_id == dup3_id {
        info!("\n   🎉 CROSS-LIBRARY DEDUPLICATION VERIFIED:");
        info!("      Same memory_id across libraries: {}", dup1_id);
        info!("      Content hash is global - same content = same memory entry!");
    } else {
        info!("\n   ℹ️  Different memory ID in different library");
        info!("      First (rust_patterns):      {}", dup1_id);
        info!("      Third (debugging_insights): {}", dup3_id);
    }

    // ========================================================================
    // Final summary
    // ========================================================================
    info!("\n========================================");
    info!("  Example completed successfully!");
    info!("========================================");
    info!("Demonstrated features:");
    info!("  ✅ Async memorize with session-based polling");
    info!("  ✅ Progress tracking during embedding generation");
    info!("  ✅ List all memory libraries");
    info!("  ✅ Recall using semantic similarity search");
    info!("  ✅ Deduplication via content hash");
    info!("  ✅ Multi-library organization");

    Ok(())
}
