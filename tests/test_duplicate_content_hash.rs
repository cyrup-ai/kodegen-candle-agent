/// Test for duplicate content_hash upsert behavior
/// Verifies that re-ingesting content with the same hash resets importance to 1.0

#[cfg(test)]
mod test_duplicate_content_hash {
    use kodegen_candle_agent::domain::memory::primitives::types::MemoryTypeEnum;
    use kodegen_candle_agent::domain::memory::serialization::content_hash;
    use kodegen_candle_agent::memory::core::manager::surreal::manager::SurrealDBMemoryManager;
    use kodegen_candle_agent::memory::core::manager::surreal::trait_def::MemoryManager;
    use kodegen_candle_agent::memory::primitives::MemoryNode;
    use kodegen_candle_agent::memory::utils::current_timestamp_ms;
    
    async fn create_test_memory(content: &str, memory_type: MemoryTypeEnum) -> MemoryNode {
        MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: kodegen_candle_agent::memory::primitives::MemoryContent {
                text: content.to_string(),
                source: None,
                metadata: None,
            },
            content_hash: content_hash(content),
            memory_type,
            created_at: current_timestamp_ms(),
            updated_at: current_timestamp_ms(),
            metadata: kodegen_candle_agent::memory::primitives::MemoryMetadata {
                importance: 1.0,
                last_accessed_at: current_timestamp_ms(),
                access_count: 0,
                embedding: None,
                tags: vec![],
                links: vec![],
            },
        }
    }
    
    #[tokio::test]
    async fn test_duplicate_content_hash_resets_importance() {
        // Initialize SurrealDB in-memory instance
        let manager = SurrealDBMemoryManager::new(None)
            .await
            .expect("Failed to create memory manager");
        
        // Create initial memory with importance 1.0
        let content = "This is a test fact about Rust memory management";
        let memory1 = create_test_memory(content, MemoryTypeEnum::Fact).await;
        
        // Insert first memory
        let result1 = manager.create_memory(memory1).await;
        assert!(result1.is_ok(), "First insert should succeed");
        let inserted1 = result1.unwrap();
        assert_eq!(inserted1.metadata.importance, 1.0);
        
        // Simulate importance decay by updating
        let mut decayed = inserted1.clone();
        decayed.metadata.importance = 0.3; // Simulated decay
        let _update_result = manager.update_memory(decayed).await;
        
        // Verify importance was decayed
        let retrieved = manager.get_memory(&inserted1.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.as_ref().unwrap().metadata.importance, 0.3);
        
        // Create second memory with SAME content (duplicate content_hash)
        let memory2 = create_test_memory(content, MemoryTypeEnum::Fact).await;
        
        // Insert duplicate - should detect and reset importance to 1.0
        let result2 = manager.create_memory(memory2).await;
        assert!(result2.is_ok(), "Duplicate insert should succeed (upsert)");
        let upserted = result2.unwrap();
        
        // Verify importance was reset to maximum (1.0)
        assert_eq!(upserted.metadata.importance, 1.0, 
            "Importance should be reset to maximum (1.0) on duplicate content re-ingestion");
        
        // Verify it's the same record (same ID)
        assert_eq!(upserted.id, inserted1.id, 
            "Upsert should return the same record, not create a new one");
    }
}
