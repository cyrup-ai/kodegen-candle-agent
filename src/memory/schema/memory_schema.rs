//! Database schema for memory nodes

use serde::{Deserialize, Serialize};
use surrealdb::types::{Datetime, RecordId, SurrealValue};
use uuid::Uuid;

use crate::memory::core::primitives::types::MemoryTypeEnum;

/// Database schema for memory nodes
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MemoryNodeSchema {
    /// Unique identifier
    pub id: RecordId,
    /// Content of the memory
    pub content: String,
    /// Content hash for fast deduplication and lookup
    pub content_hash: i64,
    /// Type of memory
    pub memory_type: MemoryTypeEnum,
    /// Metadata associated with the memory
    pub metadata: MemoryMetadataSchema,
    /// Raw cosine similarity score from vector search (0.0 to 1.0)
    /// Only populated when retrieved via vector search
    #[serde(default)]
    pub similarity_score: Option<f32>,
    /// Importance-weighted vector similarity score (cosine * importance)
    /// Used for ranking, only populated when retrieved via vector search
    #[serde(default)]
    pub vector_score: Option<f32>,
    /// Related memories from graph traversal (1-hop neighbors)
    /// Only populated when retrieved via hybrid search with expansion
    #[serde(default)]
    pub related_memories: Option<Vec<MemoryNodeSchema>>,
}

/// Database schema for memory metadata
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MemoryMetadataSchema {
    /// Creation time
    pub created_at: Datetime,
    /// Last accessed time
    pub last_accessed_at: Datetime,
    /// Importance score (0.0 to 1.0)
    pub importance: f32,
    /// Vector embedding
    pub embedding: Option<Vec<f32>>,
    /// Classification tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Keywords extracted from content
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Custom metadata
    #[serde(default)]
    pub custom: serde_json::Value,
}

/// Public memory type for API access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique identifier
    pub id: String,
    /// Content of the memory
    pub content: String,
    /// Type of memory
    pub memory_type: String,
    /// Creation time
    pub created_at: Datetime,
    /// Last updated time
    pub updated_at: Datetime,
    /// Last accessed time
    pub last_accessed_at: Datetime,
    /// Importance score (0.0 to 1.0)
    pub importance: f32,
    /// Vector embedding
    pub embedding: Option<Vec<f32>>,
    /// Tags
    pub tags: Vec<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl Memory {
    /// Create a new memory instance
    pub fn new(content: String, memory_type: MemoryTypeEnum) -> Self {
        let now = Datetime::now();
        let id = Uuid::new_v4().simple().to_string();

        Self {
            id,
            content,
            memory_type: memory_type.to_string(),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            importance: 0.5,
            embedding: None,
            tags: Vec::new(),
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Update the last accessed time
    pub fn touch(&mut self) {
        self.last_accessed_at = Datetime::now();
    }

    /// Set the embedding vector
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
        self.updated_at = Datetime::now();
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        if let serde_json::Value::Object(ref mut map) = self.metadata {
            map.insert(key, value);
            self.updated_at = Datetime::now();
        }
    }

    /// Remove metadata by key
    pub fn remove_metadata(&mut self, key: &str) {
        if let serde_json::Value::Object(ref mut map) = self.metadata {
            map.remove(key);
            self.updated_at = Datetime::now();
        }
    }
}
