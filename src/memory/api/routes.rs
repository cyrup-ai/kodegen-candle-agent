//! API routes for the memory system
//! This module defines the HTTP routes and endpoints

use std::sync::{Arc, RwLock};

use axum::{
    Router,
    routing::{delete, get, post, put},
};

use super::handlers::{
    create_memory, delete_memory, get_health, get_memory, get_metrics, search_memories,
    update_memory,
};
use crate::memory::SurrealMemoryManager;

/// Combined application state
#[derive(Clone)]
pub struct AppState {
    pub memory_manager: Arc<SurrealMemoryManager>,
    pub last_search_latency: Arc<RwLock<f64>>,
}

/// Create the main API router
pub fn create_router(memory_manager: Arc<SurrealMemoryManager>) -> Router {
    // Create combined application state
    let state = AppState {
        memory_manager,
        last_search_latency: Arc::new(RwLock::new(0.0_f64)),
    };
    
    Router::new()
        // Memory operations
        .route("/memories", post(create_memory))
        .route("/memories/:id", get(get_memory))
        .route("/memories/:id", put(update_memory))
        .route("/memories/:id", delete(delete_memory))
        .route("/memories/search", post(search_memories))
        // Health and monitoring
        .route("/health", get(get_health))
        .route("/metrics", get(get_metrics))
        // Inject combined application state
        .with_state(state)
}
