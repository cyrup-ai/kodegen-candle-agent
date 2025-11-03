//! HTTP handlers for the memory API
//! This module contains the actual handler functions for each endpoint

use axum::{
    Json as JsonBody,
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use futures_util::StreamExt;
use surrealdb_types::{Datetime, Value};

use super::models::{CreateMemoryRequest, HealthResponse, MemoryResponse, SearchRequest};
use super::routes::AppState;
use crate::memory::core::primitives::node::MemoryNode;
use crate::memory::manager::surreal::MemoryManager;

/// Create a new memory
pub async fn create_memory(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<CreateMemoryRequest>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate request
    if request.content.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Create memory node from request
    let content = crate::memory::core::primitives::types::MemoryContent::new(&request.content);
    let memory_node = MemoryNode::new(request.memory_type, content);

    // Create memory using the manager
    let pending_memory = state.memory_manager.create_memory(memory_node);
    match pending_memory.await {
        Ok(memory) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content.text,
                memory_type: memory.memory_type,
                metadata: request.metadata,
                user_id: request.user_id,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Err(e) => {
            log::error!("Failed to create memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get a memory by ID
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate ID format
    if id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Retrieve memory using the manager
    match state.memory_manager.get_memory(&id).await {
        Ok(Some(memory)) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content.text,
                metadata: serde_json::to_value(&memory.metadata).ok(),
                memory_type: memory.memory_type,
                user_id: None,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            log::error!("Failed to get memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Update a memory
pub async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
    JsonBody(request): JsonBody<CreateMemoryRequest>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate inputs
    if id.is_empty() || request.content.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Update memory using the manager
    // Create updated memory node
    let content = crate::memory::core::primitives::types::MemoryContent::new(&request.content);
    let updated_memory = MemoryNode::with_id(id.clone(), request.memory_type, content);

    let pending_memory = state.memory_manager.update_memory(updated_memory);
    match pending_memory.await {
        Ok(memory) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content.text,
                metadata: serde_json::to_value(&memory.metadata).ok(),
                memory_type: memory.memory_type,
                user_id: None,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Err(e) => {
            log::error!("Failed to update memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Delete a memory
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    // Validate ID format
    if id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Delete memory using the manager
    match state.memory_manager.delete_memory(&id).await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            log::error!("Failed to delete memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memories
pub async fn search_memories(
    State(state): State<AppState>,
    JsonBody(request): JsonBody<SearchRequest>,
) -> Result<Json<Vec<MemoryResponse>>, StatusCode> {
    // Track search latency for metrics
    let start = std::time::Instant::now();
    
    // Validate search request
    if request.query.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Perform search using the manager
    let mut memory_stream = state.memory_manager.search_by_content(&request.query);

    // Collect memories from stream
    let mut memories: Vec<MemoryNode> = vec![];
    while let Some(result) = memory_stream.next().await {
        match result {
            Ok(memory) => memories.push(memory),
            Err(_) => continue, // Skip failed items
        }
    }

    // Process the memories directly
    let responses: Vec<MemoryResponse> = memories
        .into_iter()
        .map(|memory| MemoryResponse {
            id: memory.id,
            content: memory.content.text,
            metadata: serde_json::to_value(&memory.metadata).ok(),
            memory_type: memory.memory_type,
            user_id: None,
            created_at: memory.created_at,
            updated_at: memory.updated_at,
        })
        .collect();
    
    // Calculate and store search latency for metrics
    let latency_seconds = start.elapsed().as_secs_f64();
    
    // Store latency in shared state for metrics endpoint
    if let Ok(mut latency) = state.last_search_latency.write() {
        *latency = latency_seconds;
    } else {
        log::warn!("Failed to acquire write lock for latency state");
    }
    
    log::debug!("Search completed in {:.3}s, returned {} results", latency_seconds, responses.len());
    
    Ok(Json(responses))
}

/// Health check endpoint
pub async fn get_health(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    // Perform actual health check using the memory manager
    let status = if state.memory_manager.health_check().await.is_ok() {
        "healthy".to_string()
    } else {
        "unhealthy".to_string()
    };

    Json(HealthResponse {
        status,
        timestamp: Datetime::now(),
    })
}

/// Metrics endpoint
pub async fn get_metrics(
    State(state): State<AppState>,
) -> Result<String, StatusCode> {
    // Collect actual metrics from the memory manager
    let mut output = String::with_capacity(1024);

    // Get total memory count using efficient SurrealDB COUNT query
    let total_count: u64 = {
        let mut query_result = state.memory_manager
            .database()
            .query("SELECT count() AS total FROM memory")
            .await
            .map_err(|e| {
                log::error!("Failed to count memories: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        
        let count_result: Option<u64> = query_result
            .take("total")
            .map_err(|e| {
                log::error!("Failed to parse count result: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        
        count_result.unwrap_or(0)
    };

    let is_healthy = state.memory_manager.health_check().await.is_ok();

    // Memory health status
    output.push_str(
        "# HELP memory_manager_healthy Memory manager health status (1=healthy, 0=unhealthy)\n",
    );
    output.push_str("# TYPE memory_manager_healthy gauge\n");
    output.push_str(&format!(
        "memory_manager_healthy {}\n",
        if is_healthy { 1 } else { 0 }
    ));

    // Total count metric
    output.push_str("# HELP memory_total_count Total number of memories\n");
    output.push_str("# TYPE memory_total_count gauge\n");
    output.push_str(&format!("memory_total_count {}\n", total_count));

    // Per-type memory count metrics
    output.push_str("# HELP memory_count_by_type Memory count per type\n");
    output.push_str("# TYPE memory_count_by_type gauge\n");
    
    use crate::memory::primitives::types::MemoryTypeEnum;
    for memory_type in [
        MemoryTypeEnum::Episodic,
        MemoryTypeEnum::Semantic,
        MemoryTypeEnum::Procedural,
        MemoryTypeEnum::Working,
        MemoryTypeEnum::LongTerm,
    ] {
        let type_str = format!("{:?}", memory_type);
        let count: Option<u64> = state.memory_manager
            .database()
            .query(format!("SELECT count() AS total FROM memory WHERE memory_type = '{}'", type_str))
            .await
            .ok()
            .and_then(|mut result| result.take("total").ok())
            .flatten();
        
        if let Some(count) = count {
            output.push_str(&format!(
                "memory_count_by_type{{type=\"{}\"}} {}\n",
                type_str.to_lowercase(), count
            ));
        }
    }

    // Get last search latency from shared state
    let search_latency = state.last_search_latency
        .read()
        .map(|l| *l)
        .unwrap_or_else(|_| {
            log::warn!("Failed to acquire read lock for latency state");
            0.0
        });

    output.push_str("# HELP memory_search_latency_seconds Most recent search latency\n");
    output.push_str("# TYPE memory_search_latency_seconds gauge\n");
    output.push_str(&format!("memory_search_latency_seconds {:.6}\n", search_latency));

    // Get storage size from SurrealDB system information
    let storage_size_bytes: u64 = {
        match state.memory_manager
            .database()
            .query("INFO FOR ROOT")
            .await
        {
            Ok(mut response) => {
                // Parse the system info response as SurrealDB Value, then convert to JSON
                match response.take::<Value>(0) {
                    Ok(surreal_value) => {
                        // Convert SurrealDB Value to serde_json::Value for easier parsing
                        if let Ok(json_value) = serde_json::to_value(&surreal_value) {
                            json_value
                                .get("system")
                                .and_then(|sys| sys.get("memory_usage"))
                                .and_then(|mem| mem.as_u64())
                                .unwrap_or(0)
                        } else {
                            0
                        }
                    }
                    Err(_) => {
                        log::debug!("INFO FOR ROOT returned unexpected format");
                        0
                    }
                }
            }
            Err(e) => {
                log::warn!("Failed to get SurrealDB storage stats: {}", e);
                0
            }
        }
    };

    output.push_str("# HELP memory_storage_size_bytes SurrealDB memory usage in bytes\n");
    output.push_str("# TYPE memory_storage_size_bytes gauge\n");
    output.push_str(&format!("memory_storage_size_bytes {}\n", storage_size_bytes));

    Ok(output)
}
