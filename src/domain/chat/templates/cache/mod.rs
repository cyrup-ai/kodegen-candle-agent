//! Template caching system
//!
//! Provides high-performance caching for compiled templates.

pub mod store;

pub use store::{MemoryStore, TemplateStore};
