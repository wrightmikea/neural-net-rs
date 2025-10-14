/// Neural Network Demonstration Platform CLI
///
/// Command-line interface for training and evaluating neural networks
/// on classic logic gate problems (AND, OR, XOR).
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "neural-net")]
#[command(about = "Neural Network Demonstration Platform", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available training examples
    List,

    /// Train a neural network on an example
    Train {
        /// Example to train on (and, or, xor)
        #[arg(short, long)]
        example: String,

        /// Number of training epochs
        #[arg(short = 'n', long, default_value = "10000")]
        epochs: u32,

        /// Learning rate
        #[arg(short, long, default_value = "0.5")]
        learning_rate: f64,

        /// Output file path for trained model
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Evaluate a trained model
    Eval {
        /// Path to trained model file
        #[arg(short, long)]
        model: String,

        /// Input values (comma-separated)
        #[arg(short, long)]
        input: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::List => {
            cmd_list()?;
        }
        Commands::Train {
            example,
            epochs,
            learning_rate,
            output,
        } => {
            cmd_train(&example, epochs, learning_rate, output)?;
        }
        Commands::Eval { model, input } => {
            cmd_eval(&model, input)?;
        }
    }

    Ok(())
}

/// List available training examples
fn cmd_list() -> anyhow::Result<()> {
    use neural_network::examples;

    println!("Available Examples:");
    println!();

    for name in examples::list_examples() {
        let example = examples::get_example(name).unwrap();
        println!("  {} - {}", name, example.description);
    }

    Ok(())
}

/// Train a neural network (scaffold implementation)
fn cmd_train(
    example: &str,
    _epochs: u32,
    _learning_rate: f64,
    _output: Option<String>,
) -> anyhow::Result<()> {
    use neural_network::examples;

    // Validate example exists
    if examples::get_example(example).is_none() {
        anyhow::bail!("Unknown example: {}. Use 'list' to see available examples.", example);
    }

    println!("Training on example: {}", example);
    println!("(Full training implementation coming in Task 1.5)");

    Ok(())
}

/// Evaluate a model (scaffold implementation)
fn cmd_eval(_model: &str, _input: Option<String>) -> anyhow::Result<()> {
    println!("Eval command scaffold");
    println!("(Full eval implementation coming in Phase 3)");

    Ok(())
}

