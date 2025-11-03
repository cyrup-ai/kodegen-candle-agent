# KODEGEN.ᴀɪ Candle Agent

[![License](https://img.shields.io/badge/license-Apache%202.0%20OR%20MIT-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)

**Memory-efficient, blazing-fast MCP tools for code generation agents.**

A high-performance [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides cognitive memory capabilities for AI agents. Built with Rust and the Candle ML framework, it delivers semantic memory storage, retrieval, and quantum-inspired routing for intelligent code generation workflows.

## Features

- **🧠 Cognitive Memory System** - Store and retrieve code context with semantic understanding
- **⚡ High-Performance** - Rust + SIMD optimizations for blazing-fast embeddings and retrieval
- **🔄 Async Operations** - Non-blocking memory ingestion with progress tracking
- **🎯 Multiple Retrieval Strategies** - Semantic, temporal, and hybrid search modes
- **🌊 Quantum-Inspired Routing** - Advanced memory importance scoring with entanglement
- **📊 Vector Storage** - Support for FAISS, HNSW, and instant-distance backends
- **💾 Persistent Storage** - SurrealDB with embedded SurrealKV for ACID transactions
- **🚀 Hardware Acceleration** - CUDA, Metal, MKL, and Accelerate support
- **🔧 MCP Compatible** - Works with Claude Desktop, Cline, and other MCP clients

## Quick Start

### Prerequisites

- Rust nightly toolchain (automatically configured via `rust-toolchain.toml`)
- For GPU acceleration:
  - CUDA 12+ (NVIDIA GPUs)
  - Metal (Apple Silicon)
  - MKL (Intel CPUs)

### Installation

```bash
# Clone the repository
git clone https://github.com/cyrup-ai/kodegen-candle-agent.git
cd kodegen-candle-agent

# Build with default features
cargo build --release

# Or build with hardware acceleration
cargo build --release --features metal  # macOS
cargo build --release --features cuda   # NVIDIA GPU
```

### Running the Server

```bash
# Start the MCP server (HTTP transport)
cargo run --release

# The server will start on http://localhost:3000 by default
```

### Configuration for MCP Clients

Add to your MCP client configuration (e.g., Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "kodegen-candle-agent": {
      "command": "cargo",
      "args": ["run", "--release"],
      "cwd": "/path/to/kodegen-candle-agent"
    }
  }
}
```

## Usage

The server provides four MCP tools for memory operations:

### 1. Memorize Content

Ingest files or directories into a named memory library:

```json
{
  "tool": "memorize",
  "arguments": {
    "input": "/path/to/your/codebase",
    "library": "my-project"
  }
}
```

Returns a `session_id` for tracking the async operation.

### 2. Check Memorization Status

Poll the progress of a memorization task:

```json
{
  "tool": "check_memorize_status",
  "arguments": {
    "session_id": "your-session-id"
  }
}
```

Returns status (`IN_PROGRESS`, `COMPLETED`, `FAILED`) with progress details.

### 3. Recall Memories

Search for relevant memories using semantic similarity:

```json
{
  "tool": "recall",
  "arguments": {
    "query": "authentication logic",
    "library": "my-project",
    "top_k": 5
  }
}
```

Returns ranked memories with similarity scores and importance metrics.

### 4. List Memory Libraries

Enumerate all available memory libraries:

```json
{
  "tool": "list_memory_libraries",
  "arguments": {}
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Tools Layer                      │
│  (memorize, recall, check_status, list_libraries)      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              Memory Coordinator Pool                    │
│        (Per-library coordinator management)             │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼─────┐ ┌────▼────┐ ┌──────▼──────┐
│  Graph DB   │ │ Vector  │ │  Cognitive  │
│ (SurrealDB) │ │ Storage │ │   Workers   │
└─────────────┘ └─────────┘ └─────────────┘
```

### Key Components

- **Memory Coordinator** - Orchestrates operations across graph DB, vector store, and cognitive workers
- **Quantum Routing** - Uses quantum-inspired algorithms for intelligent memory importance scoring
- **Committee Evaluation** - Multiple evaluators vote on memory relevance and importance
- **Background Workers** - Async processing for embeddings, indexing, and memory decay
- **Transaction Manager** - ACID guarantees for memory operations

## Development

### Building with Features

```bash
# Full cognitive capabilities
cargo build --features full-cognitive

# Specific vector backends
cargo build --features faiss-vector
cargo build --features hnsw-vector

# API server (HTTP endpoint for memory operations)
cargo build --features api

# Development mode (debug + desktop features)
cargo build --features dev
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test module
cargo test --test memory

# Run with output
cargo test -- --nocapture

# Run a single test
cargo test test_quantum_mcts
```

### Running the Example

```bash
# Run the demo that exercises all tools
cargo run --example candle_agent_demo --release
```

## Performance

The system is optimized for production use:

- **Zero-allocation patterns** with `arrayvec` and `smallvec` for hot paths
- **SIMD optimizations** via `kodegen_simd` for vector operations
- **Lazy loading** of embedding models and coordinators
- **Connection pooling** for efficient database access
- **Async architecture** throughout for maximum concurrency

Typical performance on Apple M1 Pro:
- Embedding generation: ~500 tokens/sec (Stella 400M)
- Memory ingestion: ~1000 files/min (with chunking and indexing)
- Semantic search: <10ms for top-5 retrieval (1M+ memories)

## Configuration

Memory system behavior can be configured via environment variables or the `MemoryConfig` struct:

```rust
use kodegen_candle_agent::memory::utils::config::MemoryConfig;

let config = MemoryConfig {
    database: DatabaseConfig {
        connection_string: "surrealkv://memory.db".to_string(),
        namespace: "kodegen".to_string(),
        database: "memories".to_string(),
        username: None,
        password: None,
    },
    vector: VectorConfig {
        backend: VectorBackend::InstantDistance,
        dimension: 1024,
    },
    cognitive: CognitiveConfig::default(),
};
```

## Embedding Models

The system uses the Stella embedding model family by default:
- **stella_en_400M_v5** - 400M parameter English model (default)
- High quality semantic representations optimized for code and text

Models are automatically downloaded from HuggingFace Hub on first use.

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

This project is dual-licensed under:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

You may choose either license for your purposes.

## Links

- **Homepage**: [https://kodegen.ai](https://kodegen.ai)
- **Repository**: [https://github.com/cyrup-ai/kodegen-candle-agent](https://github.com/cyrup-ai/kodegen-candle-agent)
- **MCP Protocol**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)

---

Built with ❤️ by the KODEGEN.ᴀɪ team
