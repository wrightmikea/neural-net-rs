# Neural Network Demonstration Platform

A comprehensive educational neural network framework implemented in Rust, featuring a full-featured CLI, checkpoint/resume functionality, and visual training progress bars.

## Overview

This project demonstrates fundamental neural network concepts through a clean, well-tested Rust implementation. It includes everything needed to train, evaluate, and experiment with neural networks on classic logic gate problems (AND, OR, XOR).

## Features

- **Feed-forward Neural Networks**: Configurable architecture with backpropagation training
- **Interactive CLI**: Full-featured command-line interface for training and evaluation
- **Checkpoint System**: Save and resume training at any point
- **Visual Progress Bars**: Real-time training progress with ETA and loss metrics
- **Training Controller**: Advanced training orchestration with callback support
- **Example Problems**: Built-in AND, OR, and XOR logic gate training examples
- **Comprehensive Testing**: 119+ tests with 100% passing rate
- **Zero Clippy Warnings**: Clean, idiomatic Rust code throughout

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd neural-net-rs

# Build the project
cargo build --release

# Run the CLI
cargo run --bin neural-net-cli -- --help
```

### Training a Network

```bash
# Train an XOR network with visual progress bar
cargo run --bin neural-net-cli -- train --example xor --epochs 10000

# Save the trained model
cargo run --bin neural-net-cli -- train --example xor --epochs 10000 --output xor_model.json

# Customize learning rate
cargo run --bin neural-net-cli -- train --example and --epochs 5000 --learning-rate 0.3 --output and_model.json
```

### Evaluating a Trained Model

```bash
# Load and evaluate a trained model
cargo run --bin neural-net-cli -- eval --model xor_model.json --input 1.0,0.0

# View model information
cargo run --bin neural-net-cli -- info --model xor_model.json
```

### Resuming Training

```bash
# Resume training from a checkpoint
cargo run --bin neural-net-cli -- resume --checkpoint xor_model.json --epochs 5000 --output xor_continued.json
```

## Project Structure

This is a Cargo workspace with multiple crates:

```
neural-net-rs/
├── matrix/                 # Core linear algebra library
├── neural-network/         # Neural network implementation
│   ├── src/
│   │   ├── network.rs     # Network architecture
│   │   ├── activations.rs # Activation functions
│   │   ├── checkpoint.rs  # Save/load functionality
│   │   ├── training.rs    # Training controller
│   │   └── examples.rs    # Built-in examples
│   └── tests/             # Integration tests
├── neural-net-cli/         # Command-line interface
│   ├── src/main.rs        # CLI implementation
│   └── tests/             # CLI integration tests
└── consumer_binary/        # Example usage binary
```

## CLI Commands

### `list` - List Available Examples

```bash
cargo run --bin neural-net-cli -- list
```

Shows all built-in training examples with descriptions.

### `train` - Train a New Network

```bash
cargo run --bin neural-net-cli -- train [OPTIONS]

Options:
  -e, --example <EXAMPLE>          Example to train on (and, or, xor)
  -n, --epochs <EPOCHS>            Number of training epochs [default: 10000]
  -l, --learning-rate <RATE>       Learning rate [default: 0.5]
  -o, --output <FILE>              Output file path for trained model
```

Features:
- Visual progress bar with ETA
- Real-time loss tracking
- Automatic checkpoint saving

### `resume` - Resume Training from Checkpoint

```bash
cargo run --bin neural-net-cli -- resume [OPTIONS]

Options:
  -c, --checkpoint <FILE>          Path to checkpoint file
  -n, --epochs <EPOCHS>            Number of additional training epochs
  -o, --output <FILE>              Output file path for updated model
```

### `eval` - Evaluate a Trained Model

```bash
cargo run --bin neural-net-cli -- eval [OPTIONS]

Options:
  -m, --model <FILE>               Path to trained model file
  -i, --input <VALUES>             Input values (comma-separated)
```

Example:
```bash
cargo run --bin neural-net-cli -- eval --model xor_model.json --input 1.0,0.0
```

### `info` - Display Model Information

```bash
cargo run --bin neural-net-cli -- info [OPTIONS]

Options:
  -m, --model <FILE>               Path to model file
```

Displays:
- Model metadata (version, example, epochs, learning rate, timestamp)
- Network architecture (layers, neurons)
- Weight matrix dimensions
- Bias vector dimensions
- Total parameter count

## Architecture Details

### Matrix Library

The `matrix` crate provides the foundation for all neural network operations:

- **Matrix struct**: Efficient row-major storage with `Vec<f64>`
- **Operations**: Element-wise multiply, dot product, transpose, add, subtract
- **Functional programming**: Generic `map` function for transformations
- **Construction helpers**: `new()`, `zeros()`, `random()`, and `matrix!` macro
- **Well-tested**: Comprehensive test suite with edge cases

### Neural Network

The `neural-network` crate implements the core learning algorithms:

- **Configurable architecture**: Specify layer sizes as `Vec<usize>`
- **Activation functions**: Pluggable activation (currently SIGMOID)
- **Forward propagation**: Efficient matrix operations with activation caching
- **Backpropagation**: Gradient computation and weight updates
- **Serialization**: Full network state save/load with `serde`

### Checkpoint System

Robust checkpoint functionality for long-running training:

```rust
pub struct Checkpoint {
    pub metadata: CheckpointMetadata,
    pub network: Network,
}

pub struct CheckpointMetadata {
    pub version: String,
    pub example: String,
    pub epoch: u32,
    pub total_epochs: u32,
    pub learning_rate: f64,
    pub timestamp: String,
}
```

Features:
- Version checking on load
- Human-readable JSON format
- Automatic timestamp tracking
- Training continuity metadata

### Training Controller

Advanced training orchestration with callback support:

```rust
pub struct TrainingController {
    network: Network,
    config: TrainingConfig,
    callbacks: Vec<TrainingCallback>,
}
```

Features:
- **Callbacks**: Execute custom code after each epoch
- **Auto-checkpointing**: Periodic checkpoint saving
- **Progress tracking**: Loss calculation and monitoring
- **Verbose mode**: Optional detailed logging
- **Resumable training**: Load and continue from checkpoints

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p matrix
cargo test -p neural-network
cargo test -p neural-net-cli

# Run specific test file
cargo test --test checkpoint_tests
cargo test --test eval_tests

# Run with output
cargo test -- --nocapture
```

### Code Quality

```bash
# Run clippy (linter)
cargo clippy --all-targets --all-features

# Format code
cargo fmt

# Build documentation
cargo doc --no-deps --open
```

### Test Coverage

- **Total tests**: 119+
- **Matrix tests**: 12 unit tests
- **Neural network tests**: 50+ integration tests
- **CLI tests**: 57+ integration tests
- **Test isolation**: Uses `tempfile` crate for parallel test safety

## Examples

### Training XOR (Classic Non-Linear Problem)

```bash
# Train XOR with 10,000 epochs
cargo run --bin neural-net-cli -- train \
  --example xor \
  --epochs 10000 \
  --learning-rate 0.5 \
  --output models/xor.json

# Evaluate all combinations
cargo run --bin neural-net-cli -- eval --model models/xor.json --input 0.0,0.0  # ~0.0
cargo run --bin neural-net-cli -- eval --model models/xor.json --input 0.0,1.0  # ~1.0
cargo run --bin neural-net-cli -- eval --model models/xor.json --input 1.0,0.0  # ~1.0
cargo run --bin neural-net-cli -- eval --model models/xor.json --input 1.0,1.0  # ~0.0
```

### Long Training with Resume

```bash
# Initial training (5000 epochs)
cargo run --bin neural-net-cli -- train \
  --example xor \
  --epochs 5000 \
  --output models/xor_partial.json

# Check progress
cargo run --bin neural-net-cli -- info --model models/xor_partial.json

# Resume for another 5000 epochs
cargo run --bin neural-net-cli -- resume \
  --checkpoint models/xor_partial.json \
  --epochs 5000 \
  --output models/xor_full.json
```

## Built-in Examples

### AND Gate

- **Architecture**: [2, 2, 1]
- **Description**: Logical AND operation
- **Difficulty**: Easy (linearly separable)

### OR Gate

- **Architecture**: [2, 2, 1]
- **Description**: Logical OR operation
- **Difficulty**: Easy (linearly separable)

### XOR Gate

- **Architecture**: [2, 3, 1]
- **Description**: Logical XOR operation (classic non-linear problem)
- **Difficulty**: Moderate (requires hidden layer)

## Technical Stack

- **Language**: Rust 2024 Edition
- **Build System**: Cargo with workspace support
- **Dependencies**:
  - `serde` / `serde_json`: Serialization
  - `clap`: CLI argument parsing
  - `chrono`: Timestamp handling
  - `anyhow`: Error handling
  - `indicatif`: Progress bars
  - `tempfile`: Test isolation

## Testing Philosophy

This project follows strict Test-Driven Development (TDD):

1. **RED**: Write failing tests first
2. **GREEN**: Implement minimal code to pass
3. **REFACTOR**: Improve code quality while maintaining tests

All features are fully tested with both unit and integration tests. The test suite runs in parallel safely using the `tempfile` crate for test isolation.

## Performance Considerations

- **Matrix operations**: Optimized for small matrices (typical NN sizes)
- **Memory efficiency**: Row-major storage, minimal allocations
- **Training speed**: Suitable for small networks and datasets
- **Not production-ready**: Educational focus, not optimized for large-scale ML

## Contributing

This is an educational project demonstrating neural network concepts. When contributing:

1. Follow TDD methodology
2. Maintain 100% test pass rate
3. Keep clippy warnings at zero
4. Add tests for all new features
5. Update documentation

## License

See LICENSE file for details.

## Resources

- **Original README**: See `ORIG-README.md` for the initial project description
- **Development Guide**: See `CLAUDE.md` for development guidelines
- **Lessons Learned**: See `docs/learnings.md` for documented patterns and solutions

## Future Enhancements

Potential areas for expansion:

- [ ] Web server with REST API (Axum)
- [ ] Real-time training visualization (Server-Sent Events)
- [ ] WASM compilation for browser execution
- [ ] Web UI for interactive training
- [ ] Additional activation functions (ReLU, Tanh, etc.)
- [ ] More complex datasets (MNIST, etc.)
- [ ] Convolutional and recurrent architectures
- [ ] GPU acceleration

---

Built with ❤️ as an educational demonstration of neural networks in Rust.
