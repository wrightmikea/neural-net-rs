// Integration tests for eval command
use std::process::Command;
use tempfile::TempDir;

fn create_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp directory")
}

#[test]
fn test_eval_trained_and_gate() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("and_model.json");

    // Train an AND gate model
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "and",
            "--epochs",
            "5000",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Evaluate: 0, 0 -> should be close to 0
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--model",
            model_path.to_str().unwrap(),
            "--input",
            "0.0,0.0",
        ])
        .output()
        .expect("Failed to run eval");

    assert!(output.status.success(), "Eval should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Output") || stdout.contains("0."), "Should show output value");

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_eval_multiple_inputs() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("or_model.json");

    // Train an OR gate model
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "or",
            "--epochs",
            "5000",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Test multiple inputs
    let test_cases = vec!["0.0,0.0", "0.0,1.0", "1.0,0.0", "1.0,1.0"];

    for input in test_cases {
        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "neural-net-cli",
                "--",
                "eval",
                "--model",
                model_path.to_str().unwrap(),
                "--input",
                input,
            ])
            .output()
            .expect("Failed to run eval");

        assert!(
            output.status.success(),
            "Eval with input {} should succeed",
            input
        );
    }

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_eval_requires_model() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--input",
            "0.0,0.0",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        !output.status.success(),
        "Eval without model should fail"
    );
}

#[test]
fn test_eval_nonexistent_model() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--model",
            "/nonexistent/model.json",
            "--input",
            "0.0,0.0",
        ])
        .output()
        .expect("Failed to run eval");

    assert!(
        !output.status.success(),
        "Eval with nonexistent model should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No such file") || stderr.contains("not found") || stderr.contains("failed"),
        "Error should mention file not found"
    );
}

#[test]
fn test_eval_invalid_input_format() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("model.json");

    // Train a model
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "and",
            "--epochs",
            "100",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Test with invalid input format
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--model",
            model_path.to_str().unwrap(),
            "--input",
            "invalid",
        ])
        .output()
        .expect("Failed to run eval");

    assert!(
        !output.status.success(),
        "Eval with invalid input format should fail"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_eval_wrong_input_dimensions() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("model.json");

    // Train a 2-input model
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "and",
            "--epochs",
            "100",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Test with wrong number of inputs (3 instead of 2)
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--model",
            model_path.to_str().unwrap(),
            "--input",
            "0.0,0.0,0.0",
        ])
        .output()
        .expect("Failed to run eval");

    assert!(
        !output.status.success(),
        "Eval with wrong input dimensions should fail"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_eval_all_examples() {
    let temp_dir = create_temp_dir();

    // Test all three examples
    let examples = vec!["and", "or", "xor"];

    for example in examples {
        let model_path = temp_dir.path().join(format!("{}_model.json", example));

        // Train model
        Command::new("cargo")
            .args([
                "run",
                "--bin",
                "neural-net-cli",
                "--",
                "train",
                "--example",
                example,
                "--epochs",
                "1000",
                "--output",
                model_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to train");

        // Eval model
        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "neural-net-cli",
                "--",
                "eval",
                "--model",
                model_path.to_str().unwrap(),
                "--input",
                "1.0,1.0",
            ])
            .output()
            .expect("Failed to eval");

        assert!(
            output.status.success(),
            "Eval of {} model should succeed",
            example
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Output") || stdout.contains("0.") || stdout.contains("1."),
            "Should show prediction output for {}",
            example
        );
    }

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_eval_shows_model_info() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("model.json");

    // Train a model
    Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "xor",
            "--epochs",
            "1000",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Eval and check that model info is displayed
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "eval",
            "--model",
            model_path.to_str().unwrap(),
            "--input",
            "0.0,1.0",
        ])
        .output()
        .expect("Failed to eval");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show some model information
    assert!(
        stdout.contains("Architecture") || stdout.contains("Model") || stdout.contains("xor"),
        "Should display model information"
    );

    // TempDir automatically cleans up when dropped
}
