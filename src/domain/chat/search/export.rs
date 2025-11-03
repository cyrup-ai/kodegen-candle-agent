//! Search result export functionality

use serde_json;
use std::pin::Pin;
use tokio_stream::Stream;

use super::types::{ExportFormat, ExportOptions, SearchError, SearchResult};
use crate::domain::chat::message::CandleSearchChatMessage;
use crate::domain::context::chunks::CandleJsonChunk;

/// Search result exporter with streaming capabilities
pub struct SearchExporter {
    /// Default export options
    default_options: ExportOptions,
}

/// History exporter for chat conversation history (domain version)
#[derive(Debug)]
pub struct HistoryExporter {
    /// Default export options
    default_options: ExportOptions,
}

impl SearchExporter {
    /// Create a new search exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_options: ExportOptions::default(),
        }
    }

    /// Export search results as a stream
    #[must_use]
    pub fn export_stream(
        &self,
        results: Vec<SearchResult>,
        options: Option<ExportOptions>,
    ) -> Pin<Box<dyn Stream<Item = CandleJsonChunk> + Send>> {
        let export_options = options.unwrap_or_else(|| self.default_options.clone());
        let limited_results = if let Some(max) = export_options.max_results {
            results.into_iter().take(max).collect()
        } else {
            results
        };

        // Clone self to avoid borrowing issues
        let _self_clone = self.clone();

        Box::pin(crate::async_stream::spawn_stream(move |tx| async move {
            if let ExportFormat::Json = export_options.format {
                if let Ok(json) =
                    SearchExporter::export_json_sync(&limited_results, &export_options)
                    && let Ok(value) = serde_json::from_str(&json)
                {
                    let _ = tx.send(CandleJsonChunk(value));
                }
            } else {
                // Other formats not implemented in simplified version
                let error_value = serde_json::json!({"error": "Export format not supported"});
                let _ = tx.send(CandleJsonChunk(error_value));
            }
        }))
    }
}

/// Filter search results based on export options
///
/// Conditionally removes metadata and context fields from search results
/// based on the export option flags. This ensures that the serialized output
/// respects user preferences for data inclusion.
///
/// # Arguments
///
/// * `results` - Search results to filter
/// * `options` - Export options containing `include_metadata` and `include_context` flags
///
/// # Returns
///
/// Filtered vector of search results with fields conditionally cleared
fn filter_search_results(
    results: &[SearchResult],
    options: &ExportOptions,
) -> Vec<SearchResult> {
    results
        .iter()
        .map(|result| {
            let mut filtered = result.clone();
            
            // Conditionally exclude metadata based on flag
            if !options.include_metadata {
                filtered.metadata = None;
            }
            
            // Conditionally exclude context messages based on flag
            if !options.include_context {
                filtered.context = Vec::new();
            }
            
            filtered
        })
        .collect()
}

impl SearchExporter {
    /// Export to JSON format
    fn export_json_sync(
        results: &[SearchResult],
        options: &ExportOptions,
    ) -> Result<String, SearchError> {
        // Apply max_results limiting
        let limited_results: Vec<_> = if let Some(max) = options.max_results {
            results.iter().take(max).cloned().collect()
        } else {
            results.to_vec()
        };

        // Filter fields based on export options (respects include_metadata and include_context)
        let filtered_results = filter_search_results(&limited_results, options);

        // Serialize filtered results
        serde_json::to_string_pretty(&filtered_results).map_err(|e| SearchError::ExportError {
            reason: format!("JSON serialization failed: {e}"),
        })
    }
}

impl Default for SearchExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SearchExporter {
    fn clone(&self) -> Self {
        Self {
            default_options: self.default_options.clone(),
        }
    }
}

impl HistoryExporter {
    /// Create a new history exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_options: ExportOptions::default(),
        }
    }

    /// Create exporter with custom default options
    #[must_use]
    pub fn with_options(options: ExportOptions) -> Self {
        Self {
            default_options: options,
        }
    }

    /// Export chat history as a stream
    #[must_use]
    pub fn export_history_stream(
        &self,
        messages: Vec<CandleSearchChatMessage>,
        options: Option<ExportOptions>,
    ) -> Pin<Box<dyn Stream<Item = CandleJsonChunk> + Send>> {
        let export_options = options.unwrap_or_else(|| self.default_options.clone());
        let limited_messages = if let Some(max) = export_options.max_results {
            messages.into_iter().take(max).collect()
        } else {
            messages
        };

        Box::pin(crate::async_stream::spawn_stream(move |tx| async move {
            if let ExportFormat::Json = export_options.format {
                if let Ok(json) = serde_json::to_string_pretty(&limited_messages)
                    && let Ok(value) = serde_json::from_str(&json)
                {
                    let _ = tx.send(CandleJsonChunk(value));
                }
            } else {
                let error_value = serde_json::json!({
                    "error": "Export format not supported"
                });
                let _ = tx.send(CandleJsonChunk(error_value));
            }
        }))
    }

    /// Export chat history to JSON format (synchronous method)
    ///
    /// This method provides synchronous JSON serialization of chat messages.
    /// Use this for simple, blocking workflows where you need the complete
    /// JSON string immediately. For large datasets or async contexts, prefer
    /// [`export_history_stream()`](Self::export_history_stream) instead.
    ///
    /// # Supported Options
    ///
    /// Only the following `ExportOptions` fields are applicable:
    /// - `max_results`: Limits the number of messages exported
    /// - `format`: Must be `ExportFormat::Json`
    ///
    /// The `include_metadata` and `include_context` fields are **not applicable**
    /// because `CandleSearchChatMessage` doesn't have these fields.
    ///
    /// # When to Use
    ///
    /// **Use `export_json` for:**
    /// - Quick exports in synchronous code
    /// - Small to medium datasets (< 1000 messages)
    /// - Simple logging or debugging
    /// - Non-async contexts
    ///
    /// **Use `export_history_stream` for:**
    /// - Large datasets that should stream
    /// - Async/await contexts
    /// - Memory-conscious operations
    /// - Progressive rendering scenarios
    ///
    /// # Examples
    ///
    /// Basic usage with default options:
    /// ```rust
    /// use kodegen_tools_candle_agent::domain::chat::search::export::HistoryExporter;
    ///
    /// let exporter = HistoryExporter::new();
    /// let messages = vec![/* your CandleSearchChatMessage instances */];
    /// 
    /// let json = exporter.export_json(&messages, None)?;
    /// println!("Exported {} bytes of JSON", json.len());
    /// ```
    ///
    /// With `max_results` limiting:
    /// ```rust
    /// use kodegen_tools_candle_agent::domain::chat::search::{
    ///     export::HistoryExporter,
    ///     types::ExportOptions,
    /// };
    ///
    /// let exporter = HistoryExporter::new();
    /// let messages = vec![/* your messages */];
    /// 
    /// let options = ExportOptions {
    ///     max_results: Some(100), // Export only first 100 messages
    ///     ..Default::default()
    /// };
    /// 
    /// let json = exporter.export_json(&messages, Some(options))?;
    /// ```
    ///
    /// # Output Format
    ///
    /// Returns a pretty-printed JSON array:
    /// ```json
    /// [
    ///   {
    ///     "message": { /* CandleMessage fields */ },
    ///     "relevance_score": 0.95,
    ///     "highlights": ["keyword1", "keyword2"]
    ///   },
    ///   ...
    /// ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ExportError` if JSON serialization fails.
    /// This is rare but can occur if messages contain invalid UTF-8 or
    /// if `serde_json` encounters an unexpected data structure.
    pub fn export_json(
        &self,
        messages: &[CandleSearchChatMessage],
        options: Option<ExportOptions>,
    ) -> Result<String, SearchError> {
        let opts = options.unwrap_or_else(|| self.default_options.clone());

        // Apply max_results limiting
        let limited_messages: Vec<_> = if let Some(max) = opts.max_results {
            messages.iter().take(max).cloned().collect()
        } else {
            messages.to_vec()
        };

        serde_json::to_string_pretty(&limited_messages).map_err(|e| SearchError::ExportError {
            reason: format!("JSON serialization failed: {e}"),
        })
    }
}

impl Default for HistoryExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HistoryExporter {
    fn clone(&self) -> Self {
        Self {
            default_options: self.default_options.clone(),
        }
    }
}
