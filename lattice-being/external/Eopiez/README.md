<<<<<< cursor/vectorize-message-states-with-symbolic-representation-23cb
# Message Vectorizer

A Julia-based system for transforming motif tokens into higher-order narrative/message states using symbolic computation and vector embeddings.

## Overview

The Message Vectorizer converts motif configurations (e.g., isolation + time, decay + memory) into compressed symbolic states represented as vectors. It uses Symbolics.jl for symbolic computation and provides entropy scoring for message complexity analysis.

## Features

- **Symbolic State Representation**: Uses Symbolics.jl for symbolic manipulation of motif configurations
- **Vector Embeddings**: Creates high-dimensional vector representations of motif tokens
- **Entropy Scoring**: Computes information entropy for message complexity analysis
- **al-ULS Interface**: Provides formatted output for al-ULS module consumption
- **Compression**: Compresses motif configurations into efficient symbolic states

## Installation

1. Clone this repository
2. Install Julia dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

### Basic Usage

```julia
using MessageVectorizer

# Create motif tokens
isolation_motif = MotifToken(
    :isolation_time,
    Dict{Symbol, Any}(:intensity => 0.8, :duration => 24.0),
    0.7,
    [:temporal, :spatial, :emotional]
)

decay_motif = MotifToken(
    :decay_memory,
    Dict{Symbol, Any}(:decay_rate => 0.3, :memory_strength => 0.6),
    0.6,
    [:cognitive, :temporal, :neural]
)

# Initialize vectorizer
vectorizer = MessageVectorizer(64)

# Add motif embeddings
add_motif_embedding!(vectorizer, isolation_motif)
add_motif_embedding!(vectorizer, decay_motif)

# Vectorize message
motifs = [isolation_motif, decay_motif]
message_state = vectorize_message(motifs, vectorizer)

# Get al-ULS compatible output
uls_output = al_uls_interface(message_state)
```

### Advanced Configuration

```julia
# Custom vectorizer with specific parameters
vectorizer = MessageVectorizer(
    128,                    # embedding dimension
    entropy_threshold=0.7,  # entropy threshold
    compression_ratio=0.85  # compression ratio
)
```

## API Reference

### Core Types

#### `MotifToken`
Represents a basic motif token with symbolic properties.

```julia
struct MotifToken
    name::Symbol                    # Motif identifier
    properties::Dict{Symbol, Any}   # Motif properties
    weight::Float64                 # Motif weight
    context::Vector{Symbol}         # Contextual tags
end
```

#### `MessageState`
Represents a compressed symbolic state of a message.

```julia
struct MessageState
    symbolic_expression::Num                    # Symbolic representation
    vector_representation::Vector{Float64}     # Vector embedding
    entropy_score::Float64                      # Information entropy
    motif_configuration::Dict{Symbol, Float64} # Motif weights
    metadata::Dict{String, Any}                # Additional metadata
end
```

#### `MessageVectorizer`
Main vectorizer for transforming motif tokens.

```julia
struct MessageVectorizer
    motif_embeddings::Dict{Symbol, Vector{Float64}}  # Stored embeddings
    symbolic_variables::Dict{Symbol, Num}            # Symbolic variables
    embedding_dim::Int                               # Embedding dimension
    entropy_threshold::Float64                       # Entropy threshold
    compression_ratio::Float64                       # Compression ratio
end
```

### Core Functions

#### `vectorize_message(motifs, vectorizer)`
Transform motif tokens into a message state vector.

#### `compute_entropy(vector, motif_config)`
Compute entropy score for a message vector.

#### `create_motif_embedding(motif, dim)`
Create a vector embedding for a motif token.

#### `symbolic_state_compression(motifs, vectorizer)`
Compress motif tokens into a symbolic state representation.

#### `al_uls_interface(message_state)`
Format message state for al-ULS module consumption.

## Examples

### Running the Demo

```bash
julia examples/message_vectorizer_demo.jl
```

### Running Tests

```bash
julia test/runtests.jl
```

## Output Format

The al-ULS interface provides the following output structure:

```json
{
  "symbolic_expression": "0.7*s + 0.6*τ + ...",
  "vector_representation": [0.1, 0.2, 0.3, ...],
  "entropy_score": 2.45,
  "motif_configuration": {
    "isolation_time": 0.7,
    "decay_memory": 0.6
  },
  "metadata": {
    "num_motifs": 2,
    "compression_ratio": 0.8,
    "timestamp": 1234567890
  },
  "compressed_size": 64,
  "information_density": 0.038
}
```

## Dependencies

- **Symbolics.jl**: Symbolic computation and manipulation
- **SymbolicNumericIntegration.jl**: Symbolic-numeric integration
- **LinearAlgebra**: Vector operations and linear algebra
- **StatsBase**: Statistical functions for entropy computation
- **JSON3**: JSON serialization for output formatting
- **DataFrames**: Data manipulation (optional)

## Architecture

The Message Vectorizer follows a three-stage pipeline:

1. **Motif Embedding**: Convert motif tokens into vector representations
2. **Symbolic Compression**: Combine motifs into symbolic expressions
3. **State Vectorization**: Convert symbolic states into consumable vectors

### Symbolic Variables

The system uses four primary symbolic variables:
- `s`: State variable
- `τ`: Temporal variable  
- `μ`: Memory variable
- `σ`: Spatial variable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# Eopiez

A comprehensive AI-powered content analysis and processing system organized in phases.

## Project Overview

Eopiez is a multi-phase project that implements advanced content analysis, vectorization, theming, compression, and integration capabilities. Each phase builds upon the previous one to create a complete content processing pipeline.

## Repository Structure

```
/Eopiez
├── README.md                 # This file - project overview
├── ProjectPlan.md            # Detailed project planning and specifications
├── /motif_detector          # Phase 1: Pattern and motif detection
├── /message_vectorizer      # Phase 2: Content vectorization
├── /sheaf_theme_engine      # Phase 3: Thematic analysis engine
├── /dirac_compressor        # Phase 4: Advanced compression algorithms
├── /integration_suite       # Phase 5: System integration and orchestration
└── /tests                   # Comprehensive test suite
```

## Phase Descriptions

### Phase 1: Motif Detector
Advanced pattern recognition and motif detection algorithms for content analysis.

### Phase 2: Message Vectorizer
Content vectorization and embedding generation for semantic analysis.

### Phase 3: Sheaf Theme Engine
Thematic analysis and content categorization using sheaf theory principles.

### Phase 4: Dirac Compressor
High-efficiency compression algorithms for optimized data storage and transmission.

### Phase 5: Integration Suite
System integration, orchestration, and end-to-end workflow management.

## Getting Started

1. Review `ProjectPlan.md` for detailed specifications
2. Navigate to individual phase directories for implementation details
3. Run tests from the `/tests` directory

## Development Workflow

Each phase should be developed sequentially, with thorough testing at each stage. The integration suite will combine all phases into a cohesive system.

## License

See LICENSE file for details.
>>>>>> main
