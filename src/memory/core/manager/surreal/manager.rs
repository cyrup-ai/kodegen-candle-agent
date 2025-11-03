//! SurrealDB memory manager implementation.
//!
//! This module provides the core SurrealDBMemoryManager struct with
//! initialization, database utilities, migrations, and export/import functionality.

use std::sync::Arc;

use surrealdb::Surreal;
use surrealdb::engine::any::Any;

use crate::capability::registry::TextEmbeddingModel;
use crate::memory::migration::{
    BuiltinMigrations, DataExporter, DataImporter, ExportFormat, ImportFormat, MigrationManager,
};
use crate::memory::primitives::{MemoryNode, MemoryRelationship};
use crate::memory::schema::memory_schema::MemoryNodeSchema;
use crate::memory::utils::error::Error;
use std::path::Path;

use super::Result;
use super::types::ExportData;

/// SurrealDB-backed memory manager implementation
#[derive(Debug)]
pub struct SurrealDBMemoryManager {
    pub(in crate::memory::core) db: Surreal<Any>,
    pub(super) embedding_model: Option<TextEmbeddingModel>,
}

impl SurrealDBMemoryManager {
    /// Create a new SurrealDBMemoryManager with an existing database connection
    pub fn new(db: Surreal<Any>) -> Self {
        Self {
            db,
            embedding_model: None,
        }
    }

    /// Create a new manager with an embedding model for auto-embedding generation
    pub fn with_embedding_model(db: Surreal<Any>, embedding_model: TextEmbeddingModel) -> Self {
        Self {
            db,
            embedding_model: Some(embedding_model),
        }
    }

    /// Alternative constructor using Arc<TextEmbeddingModel>
    pub fn with_embeddings(db: Surreal<Any>, embedding_model: Arc<TextEmbeddingModel>) -> Self {
        Self {
            db,
            embedding_model: Some((*embedding_model).clone()),
        }
    }

    /// Get a reference to the underlying database connection
    pub fn database(&self) -> &Surreal<Any> {
        &self.db
    }

    /// Initialize the database schema and indexes
    ///
    /// This method sets up:
    /// - Memory table with content and metadata fields
    /// - Relationship table with source/target references
    /// - Quantum signature table for cognitive states
    /// - MTREE index for vector similarity search
    /// - Entanglement graph edges
    pub async fn initialize(&self) -> Result<()> {
        // Define the memory table schema
        self.db
            .query(
                "
                DEFINE TABLE IF NOT EXISTS memory SCHEMAFULL;
                DEFINE FIELD IF NOT EXISTS content ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS content_hash ON memory TYPE int;
                DEFINE FIELD IF NOT EXISTS memory_type ON memory TYPE string;
                DEFINE FIELD IF NOT EXISTS created_at ON memory TYPE datetime;
                DEFINE FIELD IF NOT EXISTS updated_at ON memory TYPE datetime;
                DEFINE FIELD IF NOT EXISTS metadata ON memory FLEXIBLE TYPE object;
                DEFINE FIELD IF NOT EXISTS metadata.created_at ON memory TYPE datetime;
                DEFINE FIELD IF NOT EXISTS metadata.last_accessed_at ON memory TYPE datetime;
                DEFINE FIELD IF NOT EXISTS metadata.importance ON memory TYPE float;
                DEFINE FIELD IF NOT EXISTS metadata.embedding ON memory FLEXIBLE TYPE option<array<float>>;
                DEFINE FIELD IF NOT EXISTS metadata.tags ON memory TYPE array<string>;
                DEFINE FIELD IF NOT EXISTS metadata.keywords ON memory TYPE array<string>;
                DEFINE FIELD IF NOT EXISTS metadata.custom ON memory FLEXIBLE TYPE option<object>;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define memory table: {:?}", e)))?;

        // Define the relationship table
        self.db
            .query(
                "
                DEFINE TABLE IF NOT EXISTS relationship SCHEMAFULL;
                DEFINE FIELD IF NOT EXISTS source_id ON relationship TYPE string;
                DEFINE FIELD IF NOT EXISTS target_id ON relationship TYPE string;
                DEFINE FIELD IF NOT EXISTS relationship_type ON relationship TYPE string;
                DEFINE FIELD IF NOT EXISTS created_at ON relationship TYPE datetime;
                DEFINE FIELD IF NOT EXISTS updated_at ON relationship TYPE datetime;
                DEFINE FIELD IF NOT EXISTS strength ON relationship TYPE option<float>;
                DEFINE FIELD IF NOT EXISTS metadata ON relationship FLEXIBLE TYPE option<object>;
                ",
            )
            .await
            .map_err(|e| {
                Error::Database(format!("Failed to define relationship table: {:?}", e))
            })?;

        // Define quantum signature table
        self.db
            .query(
                "
                DEFINE TABLE IF NOT EXISTS quantum_signature SCHEMAFULL;
                DEFINE FIELD IF NOT EXISTS memory_id ON quantum_signature TYPE string;
                DEFINE FIELD IF NOT EXISTS coherence_fingerprint ON quantum_signature TYPE object;
                DEFINE FIELD IF NOT EXISTS entanglement_bonds ON quantum_signature TYPE array;
                DEFINE FIELD IF NOT EXISTS superposition_contexts ON quantum_signature TYPE array<string>;
                DEFINE FIELD IF NOT EXISTS collapse_probability ON quantum_signature TYPE float;
                DEFINE FIELD IF NOT EXISTS quantum_entropy ON quantum_signature TYPE float;
                DEFINE FIELD IF NOT EXISTS created_at ON quantum_signature TYPE datetime;
                DEFINE FIELD IF NOT EXISTS decoherence_rate ON quantum_signature TYPE float;
                ",
            )
            .await
            .map_err(|e| {
                Error::Database(format!("Failed to define quantum_signature table: {:?}", e))
            })?;

        // Define MTREE index for vector similarity search
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS memory_embedding_mtree ON memory 
                FIELDS metadata.embedding 
                MTREE DIMENSION 1024 
                DIST COSINE 
                TYPE F32;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define MTREE index: {:?}", e)))?;

        // CRITICAL INDEX 1: Content deduplication (UNIQUE)
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS memory_content_hash_uniq ON memory 
                FIELDS content_hash 
                UNIQUE;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define content_hash index: {:?}", e)))?;

        // CRITICAL INDEX 2: Memory type filtering
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS memory_type_idx ON memory 
                FIELDS memory_type;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define memory_type index: {:?}", e)))?;

        // CRITICAL INDEX 3: Temporal ordering
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS memory_created_at_idx ON memory 
                FIELDS created_at;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define created_at index: {:?}", e)))?;

        // CRITICAL INDEX 4: Relationship source lookup
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS relationship_source_idx ON relationship 
                FIELDS source_id;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define relationship_source index: {:?}", e)))?;

        // CRITICAL INDEX 5: Relationship target lookup
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS relationship_target_idx ON relationship 
                FIELDS target_id;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define relationship_target index: {:?}", e)))?;

        // CRITICAL INDEX 6: Quantum signature mapping (UNIQUE)
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS quantum_signature_memory_id_uniq ON quantum_signature 
                FIELDS memory_id 
                UNIQUE;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define quantum_signature index: {:?}", e)))?;

        // Define analyzer for full-text search
        self.db
            .query(
                "
                DEFINE ANALYZER IF NOT EXISTS simple TOKENIZERS blank,class FILTERS lowercase;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define simple analyzer: {:?}", e)))?;

        // CRITICAL INDEX 7: Full-text content search
        self.db
            .query(
                "
                DEFINE INDEX IF NOT EXISTS memory_content_search ON memory
                FIELDS content
                SEARCH ANALYZER simple BM25;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define content_search index: {:?}", e)))?;

        // Define entanglement edges (graph relations)
        self.db
            .query(
                "
                DEFINE TABLE IF NOT EXISTS entangled SCHEMAFULL TYPE RELATION FROM memory TO memory;
                DEFINE FIELD IF NOT EXISTS entanglement_type ON entangled TYPE string;
                DEFINE FIELD IF NOT EXISTS strength ON entangled TYPE float;
                DEFINE FIELD IF NOT EXISTS created_at ON entangled TYPE datetime;
                ",
            )
            .await
            .map_err(|e| {
                Error::Database(format!("Failed to define entanglement edges: {:?}", e))
            })?;

        // Define causal relation edges (separate from general entanglement)
        self.db
            .query(
                "
                DEFINE TABLE IF NOT EXISTS caused SCHEMAFULL TYPE RELATION FROM memory TO memory;
                DEFINE FIELD IF NOT EXISTS strength ON caused TYPE float;
                DEFINE FIELD IF NOT EXISTS temporal_distance ON caused TYPE int;
                DEFINE FIELD IF NOT EXISTS created_at ON caused TYPE datetime;
                ",
            )
            .await
            .map_err(|e| Error::Database(format!("Failed to define causal edges: {:?}", e)))?;

        Ok(())
    }

    /// Execute a raw SurrealQL query
    ///
    /// Useful for custom queries and administrative operations.
    pub async fn execute_query(&self, query: &str) -> Result<serde_json::Value> {
        let mut response = self
            .db
            .query(query)
            .await
            .map_err(|e| Error::Database(format!("{:?}", e)))?;

        let result: Vec<serde_json::Value> = response
            .take(0)
            .map_err(|e| Error::Database(format!("{:?}", e)))?;

        Ok(serde_json::Value::Array(result))
    }

    /// Health check for database connection
    pub async fn health_check(&self) -> Result<bool> {
        match self.db.health().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Run all pending migrations
    pub async fn run_migrations(&self) -> Result<()> {
        let db_arc = Arc::new(self.db.clone());
        let mut migration_mgr = MigrationManager::new(db_arc)
            .await
            .map_err(|e| Error::Database(format!("Migration manager creation failed: {:?}", e)))?;

        // Add all built-in migrations to the manager
        for migration in BuiltinMigrations::all() {
            migration_mgr.add_migration(migration);
        }

        // Execute all pending migrations
        migration_mgr
            .migrate()
            .await
            .map_err(|e| Error::Database(format!("Migration failed: {:?}", e)))?;

        Ok(())
    }

    /// Export all memories and relationships to a file
    pub async fn export_memories(&self, path: &Path, format: ExportFormat) -> Result<()> {
        // Fetch all memories
        let query = "SELECT * FROM memory";
        let mut response = self
            .db
            .query(query)
            .await
            .map_err(|e| Error::Database(format!("Export query failed: {:?}", e)))?;

        let memory_schemas: Vec<MemoryNodeSchema> = response
            .take(0)
            .map_err(|e| Error::Database(format!("Failed to parse memories: {:?}", e)))?;

        let memories: Vec<MemoryNode> = memory_schemas.into_iter().map(Self::from_schema).collect();

        // Fetch all relationships
        let query = "SELECT * FROM relationship";
        let mut response = self
            .db
            .query(query)
            .await
            .map_err(|e| Error::Database(format!("Export query failed: {:?}", e)))?;

        let relationships: Vec<MemoryRelationship> = response
            .take(0)
            .map_err(|e| Error::Database(format!("Failed to parse relationships: {:?}", e)))?;

        // Create export data structure
        let export_data = ExportData {
            memories,
            relationships,
        };

        // Use DataExporter to write to file
        let exporter = DataExporter::new(format);
        exporter
            .export_to_file(&[export_data], path)
            .await
            .map_err(|e| Error::Other(format!("Export failed: {:?}", e)))
    }

    /// Import memories and relationships from a file
    pub async fn import_memories(&self, path: &Path, format: ImportFormat) -> Result<()> {
        // Use DataImporter for format-aware import
        let importer = DataImporter::new();

        // Import based on format
        let import_data_vec: Vec<ExportData> = match format {
            ImportFormat::Json => importer
                .import_json(path)
                .await
                .map_err(|e| Error::Other(format!("JSON import failed: {:?}", e)))?,
            ImportFormat::Csv => importer
                .import_csv(path)
                .await
                .map_err(|e| Error::Other(format!("CSV import failed: {:?}", e)))?,
        };

        let import_data = import_data_vec
            .into_iter()
            .next()
            .ok_or_else(|| Error::Other("No data in import file".to_string()))?;

        // Validation: Check for duplicate memory IDs
        let mut memory_ids = std::collections::HashSet::new();
        for memory in &import_data.memories {
            if !memory_ids.insert(memory.id.clone()) {
                return Err(Error::Other(format!(
                    "Duplicate memory ID found: {}",
                    memory.id
                )));
            }
        }

        // Validation: Verify relationship references exist
        for relationship in &import_data.relationships {
            if !memory_ids.contains(&relationship.source_id) {
                return Err(Error::Other(format!(
                    "Relationship references non-existent source_id: {}",
                    relationship.source_id
                )));
            }
            if !memory_ids.contains(&relationship.target_id) {
                return Err(Error::Other(format!(
                    "Relationship references non-existent target_id: {}",
                    relationship.target_id
                )));
            }
        }

        // Insert memories
        for memory in import_data.memories {
            // Use CREATE to insert with explicit ID
            let content = super::types::MemoryNodeCreateContent::from(&memory);

            let query = "
                CREATE memory CONTENT {
                    id: $id,
                    content: $content,
                    content_hash: $content_hash,
                    memory_type: $memory_type,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    metadata: $metadata
                }
            ";

            self.db
                .query(query)
                .bind(("id", memory.id.clone()))
                .bind(("content", content.content))
                .bind(("content_hash", content.content_hash))
                .bind(("memory_type", format!("{:?}", content.memory_type)))
                .bind(("created_at", memory.created_at))
                .bind(("updated_at", memory.updated_at))
                .bind(("metadata", content.metadata))
                .await
                .map_err(|e| Error::Database(format!("Failed to import memory: {:?}", e)))?;
        }

        // Insert relationships
        for relationship in import_data.relationships {
            let content = super::types::RelationshipCreateContent::from(&relationship);

            let query = "
                CREATE relationship CONTENT {
                    id: $id,
                    source_id: $source_id,
                    target_id: $target_id,
                    relationship_type: $relationship_type,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    strength: $strength,
                    metadata: $metadata
                }
            ";

            self.db
                .query(query)
                .bind(("id", relationship.id))
                .bind(("source_id", content.source_id))
                .bind(("target_id", content.target_id))
                .bind(("relationship_type", content.relationship_type))
                .bind(("created_at", content.created_at))
                .bind(("updated_at", content.updated_at))
                .bind(("strength", content.strength))
                .bind(("metadata", content.metadata))
                .await
                .map_err(|e| Error::Database(format!("Failed to import relationship: {:?}", e)))?;
        }

        Ok(())
    }

    /// Convert SurrealDB schema to domain MemoryNode
    pub(super) fn from_schema(schema: MemoryNodeSchema) -> MemoryNode {
        use crate::memory::core::primitives::metadata::MemoryMetadata;
        use crate::memory::core::primitives::types::MemoryContent;
        use crate::memory::monitoring::operations::OperationStatus;
        use surrealdb_types::ToSql;

        // Extract just the ID portion from the RecordId, stripping SurrealDB angle brackets
        let key_str = schema.id.key.to_sql();
        let id_str = key_str.trim_start_matches('⟨').trim_end_matches('⟩').to_string();

        let mut metadata = MemoryMetadata::with_memory_type(schema.memory_type);
        metadata.created_at = schema.metadata.created_at;
        metadata.last_accessed_at = Some(schema.metadata.last_accessed_at);
        metadata.importance = schema.metadata.importance;
        metadata.embedding = schema.metadata.embedding.clone();
        metadata.tags = schema.metadata.tags.clone();
        metadata.keywords = schema.metadata.keywords.clone();
        metadata.custom = schema.metadata.custom.clone();

        // Store raw similarity from SQL query
        if let Some(sim) = schema.similarity_score {
            metadata.custom["similarity"] = serde_json::to_value(sim).unwrap_or_default();
        }

        // Store related_memories in custom metadata for recall tool (from hybrid search)
        if let Some(related) = schema.related_memories {
            // Convert related memories to simplified format for API response
            let related_json: Vec<serde_json::Value> = related
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id.key.to_sql().trim_start_matches('⟨').trim_end_matches('⟩'),
                        "content": r.content,
                        "relevance_score": r.vector_score
                    })
                })
                .collect();
            metadata.custom["related_memories"] = serde_json::json!(related_json);
        }

        MemoryNode {
            id: id_str,
            content: MemoryContent::new(&schema.content),
            content_hash: schema.content_hash,
            memory_type: schema.memory_type,
            created_at: schema.metadata.created_at,
            updated_at: schema.metadata.last_accessed_at,
            embedding: schema.metadata.embedding,
            evaluation_status: OperationStatus::Success,
            metadata,
            relevance_score: schema.vector_score,
        }
    }
}
