//! Coordinator pool for managing multiple memory libraries
//!
//! The CoordinatorPool manages a collection of MemoryCoordinator instances,
//! one per library (physical .db file). This enables proper library isolation
//! where each library is a separate database file rather than using tags for
//! library separation.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::capability::registry::TextEmbeddingModel;
use crate::memory::core::manager::coordinator::MemoryCoordinator;
use crate::memory::utils::{Error, Result};

/// Pool of MemoryCoordinators, one per library
///
/// Each library corresponds to a physical database file at:
/// `$XDG_CONFIG_HOME/kodegen/memory/{library_name}.db`
///
/// The pool provides:
/// - Lazy initialization: Coordinators created on first access
/// - Caching: Reuses existing coordinators for subsequent requests
/// - Filesystem scanning: Lists available libraries by scanning .db files
pub struct CoordinatorPool {
    coordinators: Arc<RwLock<HashMap<String, Arc<MemoryCoordinator>>>>,
    embedding_model: TextEmbeddingModel,
}

impl CoordinatorPool {
    /// Create a new coordinator pool with the specified embedding model
    ///
    /// The pool starts empty - coordinators are created lazily when first accessed.
    ///
    /// # Arguments
    /// * `embedding_model` - Text embedding model used by all coordinators in the pool
    ///
    /// # Example
    /// ```no_run
    /// use kodegen_candle_agent::capability::registry::{FromRegistry, TextEmbeddingModel};
    /// use kodegen_candle_agent::memory::core::manager::pool::CoordinatorPool;
    ///
    /// # fn example() {
    /// let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5").unwrap();
    /// let pool = CoordinatorPool::new(emb_model);
    /// # }
    /// ```
    pub fn new(embedding_model: TextEmbeddingModel) -> Self {
        Self {
            coordinators: Arc::new(RwLock::new(HashMap::new())),
            embedding_model,
        }
    }

    /// Get a coordinator for the specified library, creating if needed
    ///
    /// If the coordinator already exists in the pool, returns the cached instance.
    /// Otherwise, creates a new coordinator connected to the library's .db file.
    ///
    /// # Arguments
    /// * `library_name` - Name of the library (becomes filename: {library_name}.db)
    ///
    /// # Returns
    /// Arc to the MemoryCoordinator for this library
    ///
    /// # Errors
    /// Returns error if coordinator creation fails (e.g., database connection error)
    ///
    /// # Example
    /// ```no_run
    /// # use kodegen_candle_agent::capability::registry::{FromRegistry, TextEmbeddingModel};
    /// # use kodegen_candle_agent::memory::core::manager::pool::CoordinatorPool;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5").unwrap();
    /// # let pool = CoordinatorPool::new(emb_model);
    /// let work_coordinator = pool.get_coordinator("work").await?;
    /// let personal_coordinator = pool.get_coordinator("personal").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_coordinator(&self, library_name: &str) -> Result<Arc<MemoryCoordinator>> {
        // Check cache first (read lock - allows concurrent reads)
        {
            let coordinators = self.coordinators.read().await;
            if let Some(coordinator) = coordinators.get(library_name) {
                log::debug!(
                    "Reusing cached coordinator for library '{}'",
                    library_name
                );
                return Ok(coordinator.clone());
            }
        }

        // Not in cache - create new coordinator (requires write lock)
        log::info!(
            "Creating new coordinator for library '{}' (first access)",
            library_name
        );

        let coordinator =
            MemoryCoordinator::from_library(library_name, self.embedding_model.clone()).await?;
        let coordinator_arc = Arc::new(coordinator);

        // Cache it
        {
            let mut coordinators = self.coordinators.write().await;
            // Double-check in case another task created it while we were waiting for lock
            if let Some(existing) = coordinators.get(library_name) {
                log::debug!(
                    "Coordinator for library '{}' was created by another task, using that one",
                    library_name
                );
                return Ok(existing.clone());
            }
            coordinators.insert(library_name.to_string(), coordinator_arc.clone());
            log::info!(
                "Cached coordinator for library '{}' (total: {} libraries)",
                library_name,
                coordinators.len()
            );
        }

        Ok(coordinator_arc)
    }

    /// List all available libraries by scanning the filesystem
    ///
    /// Scans `$XDG_CONFIG_HOME/kodegen/memory/` for .db files and returns
    /// their names (without .db extension) in sorted order.
    ///
    /// # Returns
    /// Sorted vector of library names
    ///
    /// # Errors
    /// Returns error if directory reading fails
    ///
    /// # Example
    /// ```no_run
    /// # use kodegen_candle_agent::capability::registry::{FromRegistry, TextEmbeddingModel};
    /// # use kodegen_candle_agent::memory::core::manager::pool::CoordinatorPool;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5").unwrap();
    /// # let pool = CoordinatorPool::new(emb_model);
    /// let libraries = pool.list_libraries().await?;
    /// for library in libraries {
    ///     println!("Found library: {}", library);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_libraries(&self) -> Result<Vec<String>> {
        // Construct memory directory path
        let memory_dir = kodegen_config::KodegenConfig::data_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("memory");

        // If directory doesn't exist, no libraries available
        if !memory_dir.exists() {
            log::debug!(
                "Memory directory does not exist: {}",
                memory_dir.display()
            );
            return Ok(Vec::new());
        }

        log::debug!("Scanning memory directory: {}", memory_dir.display());

        // Scan for .db files
        let mut libraries = Vec::new();
        let mut entries = tokio::fs::read_dir(&memory_dir).await.map_err(|e| {
            Error::Internal(format!(
                "Failed to read memory directory '{}': {}",
                memory_dir.display(),
                e
            ))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            Error::Internal(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();

            // Check if it's a .db file
            if path.extension().and_then(|s| s.to_str()) == Some("db")
                && let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    libraries.push(name.to_string());
                    log::debug!("Found library: {}", name);
                }
        }

        // Sort alphabetically for consistent ordering
        libraries.sort();

        log::info!("Found {} libraries in filesystem", libraries.len());

        Ok(libraries)
    }

    /// Shutdown all coordinators in the pool gracefully
    ///
    /// Drains the coordinator pool and calls shutdown_workers() on each.
    /// This should be called before dropping the pool to ensure clean shutdown.
    ///
    /// # Example
    /// ```no_run
    /// # use kodegen_candle_agent::capability::registry::{FromRegistry, TextEmbeddingModel};
    /// # use kodegen_candle_agent::memory::core::manager::pool::CoordinatorPool;
    /// # async fn example() {
    /// # let emb_model = TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5").unwrap();
    /// # let pool = CoordinatorPool::new(emb_model);
    /// // ... use pool ...
    /// pool.shutdown_all().await;
    /// # }
    /// ```
    pub async fn shutdown_all(&self) {
        log::info!("Shutting down all coordinators in pool");

        let mut coordinators = self.coordinators.write().await;
        let count = coordinators.len();

        for (name, coordinator) in coordinators.drain() {
            // Try to unwrap Arc to get mutable access
            // If Arc has multiple references, we can't shutdown (just log warning)
            match Arc::try_unwrap(coordinator) {
                Ok(mut coord) => {
                    coord.shutdown_workers();
                    log::info!("Shutdown coordinator for library: {}", name);
                }
                Err(arc) => {
                    log::warn!(
                        "Coordinator for library '{}' has {} remaining references, cannot shutdown cleanly",
                        name,
                        Arc::strong_count(&arc)
                    );
                }
            }
        }

        log::info!("Shutdown complete ({} coordinators)", count);
    }

    /// Get the number of cached coordinators in the pool
    ///
    /// Useful for monitoring and debugging.
    pub async fn pool_size(&self) -> usize {
        self.coordinators.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        use crate::capability::registry::FromRegistry;
        let emb_model =
            TextEmbeddingModel::from_registry("dunzhang/stella_en_400M_v5")
                .expect("test should successfully load stella_en_400M_v5 model from registry");
        let _pool = CoordinatorPool::new(emb_model);
        // Pool created successfully
    }
}
