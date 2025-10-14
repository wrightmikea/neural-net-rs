// Integration tests for info command
use std::process::Command;
use tempfile::TempDir;

fn create_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp directory")
}

#[test]
fn test_info_displays_model_metadata() {
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

    // Run info command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run info");

    assert!(output.status.success(), "Info should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should display metadata
    assert!(stdout.contains("xor"), "Should show example name");
    assert!(stdout.contains("1000"), "Should show epoch count");
    assert!(stdout.contains("0.5"), "Should show learning rate");
    assert!(stdout.contains("1.0"), "Should show version");

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_info_displays_architecture() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("and_model.json");

    // Train an AND model (2-2-1 architecture)
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
            "500",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Run info command
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run info");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should display architecture info
    assert!(
        stdout.contains("Architecture") || stdout.contains("Layers"),
        "Should show architecture label"
    );
    assert!(stdout.contains("2") && stdout.contains("1"), "Should show layer sizes");

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_info_requires_model() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        !output.status.success(),
        "Info without model should fail"
    );
}

#[test]
fn test_info_nonexistent_model() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            "/nonexistent/model.json",
        ])
        .output()
        .expect("Failed to run info");

    assert!(
        !output.status.success(),
        "Info with nonexistent model should fail"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No such file") || stderr.contains("not found") || stderr.contains("failed"),
        "Error should mention file not found"
    );
}

#[test]
fn test_info_all_examples() {
    let temp_dir = create_temp_dir();
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
                "500",
                "--output",
                model_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to train");

        // Run info
        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "neural-net-cli",
                "--",
                "info",
                "--model",
                model_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run info");

        assert!(
            output.status.success(),
            "Info for {} model should succeed",
            example
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains(example),
            "Should show example name: {}",
            example
        );
    }

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_info_displays_timestamp() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("model.json");

    // Train model
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
            "100",
            "--output",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Run info
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run info");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should display timestamp
    assert!(
        stdout.contains("Timestamp") || stdout.contains("Created") || stdout.contains("2025"),
        "Should show timestamp information"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_info_displays_weights_info() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("model.json");

    // Train XOR model (has 2 weight matrices)
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

    // Run info
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run info");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should display weights information
    assert!(
        stdout.contains("Weight") || stdout.contains("weight") || stdout.contains("parameters"),
        "Should show weight information"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_info_with_corrupted_file() {
    let temp_dir = create_temp_dir();
    let model_path = temp_dir.path().join("corrupted.json");

    std::fs::write(&model_path, "not valid json {{{").expect("Failed to write file");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "info",
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run info");

    assert!(
        !output.status.success(),
        "Info with corrupted file should fail"
    );

    // TempDir automatically cleans up when dropped
}
