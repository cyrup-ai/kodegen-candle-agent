//! Enhanced history management and search system
//!
//! This module provides comprehensive history management with SIMD-optimized full-text search,
//! lock-free tag management, and zero-allocation streaming export capabilities using
//! blazing-fast algorithms and elegant ergonomic APIs.

use std::sync::Arc;

use std::pin::Pin;
use atomic_counter::AtomicCounter;
use tokio_stream::{Stream, StreamExt};

// Submodules
pub mod algorithms;
pub mod export;
pub mod index;
pub mod manager;
pub mod query;
pub mod ranking;
pub mod tagger;
pub mod types;

// Re-export public types
pub use export::HistoryExporter as CandleHistoryExporter;
pub use export::{HistoryExporter, SearchExporter};
pub use index::ChatSearchIndex;
// Additional search capabilities with Candle prefixes
pub use index::ChatSearchIndex as CandleChatSearchIndex;
// Re-export migrated components with Candle prefixes
pub use manager::CandleEnhancedHistoryManager;
pub use query::QueryProcessor;
pub use ranking::ResultRanker;
pub use tagger::{CandleConversationTag, CandleConversationTagger, CandleTaggingStatistics};
pub use types::*;

use crate::domain::chat::message::CandleSearchChatMessage as SearchChatMessage;

/// Main search interface combining all components
pub struct ChatSearcher {
    /// Search index
    index: Arc<ChatSearchIndex>,
    /// Query processor
    query_processor: QueryProcessor,
    /// Result ranker
    ranker: ResultRanker,
    /// Result exporter
    exporter: SearchExporter,
}

impl ChatSearcher {
    /// Create a new chat searcher
    pub fn new(index: Arc<ChatSearchIndex>) -> Self {
        Self {
            index,
            query_processor: QueryProcessor::new(),
            ranker: ResultRanker::new(),
            exporter: SearchExporter::new(),
        }
    }

    /// Search messages with SIMD optimization (streaming individual results)
    #[must_use]
    pub fn search_stream(
        &self,
        query: SearchQuery,
    ) -> Pin<Box<dyn Stream<Item = SearchResult> + Send>> {
        let self_clone = self.clone();
        let query_terms = query.terms.clone();
        let query_operator = query.operator.clone();
        let query_fuzzy_matching = query.fuzzy_matching;

        Box::pin(crate::async_stream::spawn_stream(move |tx| async move {
            // START TIMING
            let start_time = std::time::Instant::now();
            
            let results = match query_operator {
                QueryOperator::And => {
                    let stream = self_clone
                        .index
                        .search_and_stream(&query_terms, query_fuzzy_matching);
                    tokio::pin!(stream);
                    let mut vec = Vec::new();
                    while let Some(result) = stream.next().await {
                        vec.push(result);
                    }
                    vec
                }
                QueryOperator::Or => {
                    let stream = self_clone
                        .index
                        .search_or_stream(&query_terms, query_fuzzy_matching);
                    tokio::pin!(stream);
                    let mut vec = Vec::new();
                    while let Some(result) = stream.next().await {
                        vec.push(result);
                    }
                    vec
                }
                QueryOperator::Not => {
                    let stream = self_clone
                        .index
                        .search_not_stream(&query_terms, query_fuzzy_matching);
                    tokio::pin!(stream);
                    let mut vec = Vec::new();
                    while let Some(result) = stream.next().await {
                        vec.push(result);
                    }
                    vec
                }
                QueryOperator::Phrase => {
                    let stream = self_clone
                        .index
                        .search_phrase_stream(&query_terms, query_fuzzy_matching);
                    tokio::pin!(stream);
                    let mut vec = Vec::new();
                    while let Some(result) = stream.next().await {
                        vec.push(result);
                    }
                    vec
                }
                QueryOperator::Proximity { distance } => {
                    let stream = self_clone.index.search_proximity_stream(
                        &query_terms,
                        distance,
                        query_fuzzy_matching,
                    );
                    tokio::pin!(stream);
                    let mut vec = Vec::new();
                    while let Some(result) = stream.next().await {
                        vec.push(result);
                    }
                    vec
                }
            };

            // Apply enhanced filtering, sorting and pagination
            let filtered_results = Self::apply_filters(results, &query);
            let sorted_results = Self::apply_sorting(filtered_results, &query.sort_order);
            let paginated_results =
                Self::apply_pagination(sorted_results, query.offset, query.max_results);

            // Stream results
            for result in paginated_results {
                let _ = tx.send(result);
            }

            // END TIMING - Update query statistics
            let query_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            self_clone.update_query_statistics(query_duration_ms);
        }))
    }

    /// Apply comprehensive filtering system (date, user, session, tag, content)
    fn apply_filters(results: Vec<SearchResult>, query: &SearchQuery) -> Vec<SearchResult> {
        let mut filtered = results;

        // Apply date range filter
        if let Some(date_range) = &query.date_range {
            filtered.retain(|result| {
                if let Some(timestamp) = result.message.message.timestamp {
                    timestamp >= date_range.start && timestamp <= date_range.end
                } else {
                    false
                }
            });
        }

        // Apply user filter
        if let Some(user_filter) = &query.user_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .role
                    .to_string()
                    .contains(user_filter.as_ref() as &str)
            });
        }

        // Apply session filter
        if let Some(session_filter) = &query.session_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .id
                    .as_ref()
                    .is_some_and(|id| id.contains(session_filter.as_ref() as &str))
            });
        }

        // Apply content type filter
        if let Some(content_type_filter) = &query.content_type_filter {
            filtered.retain(|result| {
                result
                    .message
                    .message
                    .content
                    .contains(content_type_filter.as_ref() as &str)
            });
        }

        // Apply tag filter
        if let Some(tag_filter) = &query.tag_filter
            && !tag_filter.is_empty()
        {
            filtered.retain(|result| {
                // Message must have at least one of the filter tags
                result.tags.iter().any(|tag| tag_filter.contains(tag))
            });
        }

        filtered
    }

    /// Apply multiple sorting options (Relevance, DateDesc/Asc, UserDesc/Asc)
    fn apply_sorting(mut results: Vec<SearchResult>, sort_order: &SortOrder) -> Vec<SearchResult> {
        match sort_order {
            SortOrder::Relevance => {
                // Sort by relevance score (highest first)
                results.sort_by(|a, b| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SortOrder::DateDescending => {
                // Sort by date (newest first)
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .timestamp
                        .unwrap_or(0)
                        .cmp(&a.message.message.timestamp.unwrap_or(0))
                });
            }
            SortOrder::DateAscending => {
                // Sort by date (oldest first)
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .timestamp
                        .unwrap_or(0)
                        .cmp(&b.message.message.timestamp.unwrap_or(0))
                });
            }
            SortOrder::UserDescending => {
                // Sort by user role (descending)
                results.sort_by(|a, b| {
                    b.message
                        .message
                        .role
                        .to_string()
                        .cmp(&a.message.message.role.to_string())
                });
            }
            SortOrder::UserAscending => {
                // Sort by user role (ascending)
                results.sort_by(|a, b| {
                    a.message
                        .message
                        .role
                        .to_string()
                        .cmp(&b.message.message.role.to_string())
                });
            }
        }
        results
    }

    /// Apply pagination support (`offset`, `max_results`)
    fn apply_pagination(
        results: Vec<SearchResult>,
        offset: usize,
        max_results: usize,
    ) -> Vec<SearchResult> {
        results.into_iter().skip(offset).take(max_results).collect()
    }

    /// Update query statistics with performance tracking
    fn update_query_statistics(&self, query_duration_ms: f64) {
        // Increment query counter atomically
        self.index.increment_query_counter();
        
        // Update statistics including average query time
        let mut stats = self.index.statistics.blocking_write();
        
        // Calculate running average: new_avg = old_avg + (new_value - old_avg) / count
        let query_count = self.index.query_counter.get();
        if query_count > 0 {
            let old_avg = stats.average_query_time;
            #[allow(clippy::cast_precision_loss)]
            let new_avg = old_avg + (query_duration_ms - old_avg) / (query_count as f64);
            stats.average_query_time = new_avg;
        }
        
        stats.total_queries = query_count;
    }

    /// Search messages (collects all results asynchronously)
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if search execution fails
    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, SearchError> {
        let stream = self.search_stream(query);
        tokio::pin!(stream);
        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }
        Ok(results)
    }

    /// Add message to search index
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if message cannot be added to the index
    pub async fn add_message(&self, message: SearchChatMessage) -> Result<(), SearchError> {
        self.index.add_message(message).await
    }

    /// Add message to search index (streaming)
    #[must_use]
    pub fn add_message_stream(
        &self,
        message: SearchChatMessage,
    ) -> Pin<Box<dyn Stream<Item = index::IndexResult> + Send>> {
        self.index.add_message_stream(message)
    }

    /// Export search results
    #[must_use]
    pub fn export_results(
        &self,
        results: Vec<SearchResult>,
        options: Option<ExportOptions>,
    ) -> Pin<Box<dyn Stream<Item = crate::domain::context::chunks::CandleJsonChunk> + Send>> {
        self.exporter.export_stream(results, options)
    }

    /// Get search statistics
    #[must_use]
    pub fn get_statistics(&self) -> SearchStatistics {
        self.index.get_statistics()
    }
}

impl Clone for ChatSearcher {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            query_processor: self.query_processor.clone(),
            ranker: self.ranker.clone(),
            exporter: self.exporter.clone(),
        }
    }
}

impl Default for ChatSearcher {
    fn default() -> Self {
        Self::new(Arc::new(ChatSearchIndex::new()))
    }
}
