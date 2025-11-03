//! Template storage implementation
//!
//! Provides persistent storage for templates with caching support.

use std::collections::HashMap;

use crate::domain::chat::templates::core::{ChatTemplate, TemplateResult};

/// Template storage interface
pub trait TemplateStore: Send + Sync {
    /// Store a template
    ///
    /// # Errors
    ///
    /// Returns `TemplateError` if template cannot be stored
    fn store(
        &self,
        template: &ChatTemplate,
    ) -> impl std::future::Future<Output = TemplateResult<()>> + Send;

    /// Retrieve a template by name
    ///
    /// # Errors
    ///
    /// Returns `TemplateError` if template retrieval fails
    fn get(
        &self,
        name: &str,
    ) -> impl std::future::Future<Output = TemplateResult<Option<ChatTemplate>>> + Send;

    /// Delete a template
    ///
    /// # Errors
    ///
    /// Returns `TemplateError` if template deletion fails
    fn delete(&self, name: &str) -> impl std::future::Future<Output = TemplateResult<bool>> + Send;

    /// List all template names
    ///
    /// # Errors
    ///
    /// Returns `TemplateError` if template listing fails
    fn list(&self) -> impl std::future::Future<Output = TemplateResult<Vec<String>>> + Send;

    /// Check if template exists
    ///
    /// # Errors
    ///
    /// Returns `TemplateError` if existence check fails
    fn exists(&self, name: &str) -> impl std::future::Future<Output = TemplateResult<bool>> + Send;
}

/// In-memory template store implementation
pub struct MemoryStore {
    templates: tokio::sync::RwLock<HashMap<String, ChatTemplate>>,
}

impl MemoryStore {
    /// Create a new memory store
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateStore for MemoryStore {
    async fn store(&self, template: &ChatTemplate) -> TemplateResult<()> {
        let mut store = self.templates.write().await;
        store.insert(template.name().clone(), template.clone());
        Ok(())
    }

    async fn get(&self, name: &str) -> TemplateResult<Option<ChatTemplate>> {
        let store = self.templates.read().await;
        Ok(store.get(name).cloned())
    }

    async fn delete(&self, name: &str) -> TemplateResult<bool> {
        let mut store = self.templates.write().await;
        Ok(store.remove(name).is_some())
    }

    async fn list(&self) -> TemplateResult<Vec<String>> {
        let store = self.templates.read().await;
        Ok(store.keys().cloned().collect())
    }

    async fn exists(&self, name: &str) -> TemplateResult<bool> {
        let store = self.templates.read().await;
        Ok(store.contains_key(name))
    }
}
