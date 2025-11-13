mod common;

use anyhow::Context;
use serde_json::json;
use tracing::info;

#[derive(serde::Deserialize, Debug)]
struct MemorizeResponse {
    session_id: String,
    _status: String,
    library: String,
    _message: String,
}

#[derive(serde::Deserialize, Debug)]
struct CheckMemorizeStatusResponse {
    _session_id: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_id: Option<String>,
    _library: String,
    progress: MemorizeProgress,
    runtime_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct MemorizeProgress {
    stage: String,
    _files_loaded: usize,
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

/// Helper to wait for async memorize operation to complete
async fn wait_for_memorize_completion(
    client: &common::LoggingClient,
    session_id: &str,
) -> anyhow::Result<String> {
    let poll_interval = tokio::time::Duration::from_millis(500);
    let max_attempts = 60; // 30 seconds max

    for attempt in 1..=max_attempts {
        let status: CheckMemorizeStatusResponse = client
            .call_tool_typed(
                "memory_check_memorize_status",
                json!({ "session_id": session_id }),
            )
            .await
            .context("Failed to check memorize status")?;

        match status.status.as_str() {
            "COMPLETED" => {
                return status.memory_id.ok_or_else(|| {
                    anyhow::anyhow!("Status COMPLETED but no memory_id returned")
                });
            }
            "FAILED" => {
                let error_msg = status.error.unwrap_or_else(|| "Unknown error".to_string());
                return Err(anyhow::anyhow!("Memorize failed: {}", error_msg));
            }
            "IN_PROGRESS" => {
                if attempt % 4 == 0 {
                    // Log progress every 2 seconds
                    info!(
                        "      ⏳ Still processing... Stage: {}, Runtime: {}ms",
                        status.progress.stage, status.runtime_ms
                    );
                }
                tokio::time::sleep(poll_interval).await;
            }
            other => {
                return Err(anyhow::anyhow!("Unknown status: {}", other));
            }
        }
    }

    Err(anyhow::anyhow!(
        "Timeout waiting for memorize completion (30s)"
    ))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_env_filter("info").init();

    info!("Starting candle-agent memory tools example (async session pattern)");

    // Clean up database files from previous runs for reproducible testing
    let home_dir = dirs::home_dir().context("Failed to get home directory")?;
    let memory_dir = home_dir.join("Library/Application Support/kodegen/memory");

    let db_files = ["rust_patterns.db", "debugging_insights.db"];
    for db_file in &db_files {
        let db_path = memory_dir.join(db_file);
        if db_path.exists() {
            // SurrealKV databases are directories, not files
            std::fs::remove_dir_all(&db_path)
                .with_context(|| format!("Failed to delete {}", db_path.display()))?;
            info!("🗑️  Cleaned up previous database: {}", db_file);
        }
    }

    // Connect to kodegen-candle-agent server
    let (conn, mut server) = common::connect_to_local_http_server().await?;

    // Wrap client with logging
    let workspace_root = common::find_workspace_root()
        .context("Failed to find workspace root")?;
    let log_path = workspace_root.join("tmp/mcp-client/memory.log");

    // Clean up log file from previous runs for reproducible testing
    if log_path.exists() {
        std::fs::remove_file(&log_path)
            .with_context(|| format!("Failed to delete {}", log_path.display()))?;
        info!("🗑️  Cleaned up previous log file: {}", log_path.display());
    }

    let client = common::LoggingClient::new(conn.client(), log_path)
        .await
        .context("Failed to create logging client")?;

    info!("Connected to server: {:?}", client.server_info());

    // Run example with cleanup
    let result = run_memory_example(&client).await;

    // Always close connection, regardless of example result
    conn.close().await?;
    server.shutdown().await?;

    // Propagate any error from the example
    result
}

async fn run_memory_example(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("========================================");
    info!("  Memory Tools Demonstration");
    info!("  (Async Session Pattern)");
    info!("========================================\n");

    // ========================================================================
    // PHASE 1: Create memories in two different libraries (async pattern)
    // ========================================================================
    info!("PHASE 1: Creating memories in two libraries (async)\n");

    // Library 1: rust_patterns
    info!("1. Storing Rust pattern #1");
    let session1: MemorizeResponse = client
        .call_tool_typed(
            "memory_memorize",
            json!({
                "library": "rust_patterns",
                "content": "Error handling pattern using Result<T, E> with the ? operator for clean propagation"
            }),
        )
        .await
        .context("Failed to start memorize #1")?;
    info!("   → Session started: {}", session1.session_id);
    let mem1_id = wait_for_memorize_completion(client, &session1.session_id).await?;
    info!("   ✅ Created memory: {} in '{}'", mem1_id, session1.library);

    info!("2. Storing Rust pattern #2");
    let session2: MemorizeResponse = client
        .call_tool_typed(
            "memory_memorize",
            json!({
                "library": "rust_patterns",
                "content": "Async/await pattern for file I/O operations using tokio::fs with proper error handling"
            }),
        )
        .await
        .context("Failed to start memorize #2")?;
    info!("   → Session started: {}", session2.session_id);
    let mem2_id = wait_for_memorize_completion(client, &session2.session_id).await?;
    info!("   ✅ Created memory: {} in '{}'", mem2_id, session2.library);

    // Library 2: debugging_insights
    info!("3. Storing debugging insight #1");
    let session3: MemorizeResponse = client
        .call_tool_typed(
            "memory_memorize",
            json!({
                "library": "debugging_insights",
                "content": "React re-renders happen when props or state change - use React.memo to prevent unnecessary renders"
            }),
        )
        .await
        .context("Failed to start memorize #3")?;
    info!("   → Session started: {}", session3.session_id);
    let mem3_id = wait_for_memorize_completion(client, &session3.session_id).await?;
    info!("   ✅ Created memory: {} in '{}'", mem3_id, session3.library);

    info!("4. Storing debugging insight #2");
    let session4: MemorizeResponse = client
        .call_tool_typed(
            "memory_memorize",
            json!({
                "library": "debugging_insights",
                "content": "SQL N+1 query problem - use eager loading with JOIN instead of lazy loading to reduce DB calls"
            }),
        )
        .await
        .context("Failed to start memorize #4")?;
    info!("   → Session started: {}", session4.session_id);
    let mem4_id = wait_for_memorize_completion(client, &session4.session_id).await?;
    info!("   ✅ Created memory: {} in '{}'", mem4_id, session4.library);

    // ========================================================================
    // PHASE 2: List all libraries
    // ========================================================================
    info!("\nPHASE 2: Listing all memory libraries\n");

    info!("5. Calling list_memory_libraries()");
    let libraries: ListLibrariesResponse = client
        .call_tool_typed("memory_list_libraries", json!({}))
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
            "memory_recall",
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
            "memory_recall",
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
            "memory_memorize",
            json!({
                "library": "rust_patterns",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #1")?;
    info!("   → Session started: {}", dup_session1.session_id);
    let dup1_id = wait_for_memorize_completion(client, &dup_session1.session_id).await?;
    info!("   ✅ First insertion - Memory ID: {}", dup1_id);

    info!("9. Storing SAME content in 'rust_patterns' (second time - should deduplicate)");
    let dup_session2: MemorizeResponse = client
        .call_tool_typed(
            "memory_memorize",
            json!({
                "library": "rust_patterns",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #2")?;
    info!("   → Session started: {}", dup_session2.session_id);
    let dup2_id = wait_for_memorize_completion(client, &dup_session2.session_id).await?;
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
            "memory_memorize",
            json!({
                "library": "debugging_insights",
                "content": duplicate_content
            }),
        )
        .await
        .context("Failed to start memorize duplicate #3")?;
    info!("   → Session started: {}", dup_session3.session_id);
    let dup3_id = wait_for_memorize_completion(client, &dup_session3.session_id).await?;
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
