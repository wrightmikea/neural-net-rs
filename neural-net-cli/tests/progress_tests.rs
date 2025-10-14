// Integration tests for progress bar functionality during training
use std::process::Command;
use tempfile::TempDir;

fn create_temp_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp directory")
}

#[test]
fn test_train_with_progress_completes() {
    let temp_dir = create_temp_dir();
    let output_path = temp_dir.path().join("model.json");

    let output = Command::new("cargo")
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
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run train");

    assert!(
        output.status.success(),
        "Training with progress should complete successfully"
    );
    assert!(output_path.exists(), "Model should be saved");

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_train_short_duration_shows_progress() {
    let temp_dir = create_temp_dir();
    let output_path = temp_dir.path().join("model.json");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "or",
            "--epochs",
            "50",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run train");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show some indication of progress or completion
    assert!(
        stdout.contains("Training") || stdout.contains("complete") || stdout.contains("Epoch"),
        "Should show training progress information"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_train_long_duration_shows_progress() {
    let temp_dir = create_temp_dir();
    let output_path = temp_dir.path().join("model.json");

    let output = Command::new("cargo")
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
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run train");

    assert!(
        output.status.success(),
        "Long training with progress should complete"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_resume_with_progress_completes() {
    let temp_dir = create_temp_dir();
    let checkpoint1 = temp_dir.path().join("checkpoint1.json");
    let checkpoint2 = temp_dir.path().join("checkpoint2.json");

    // Initial training
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
            "50",
            "--output",
            checkpoint1.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    // Resume training
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "resume",
            "--checkpoint",
            checkpoint1.to_str().unwrap(),
            "--epochs",
            "50",
            "--output",
            checkpoint2.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to resume");

    assert!(
        output.status.success(),
        "Resume with progress should complete successfully"
    );
    assert!(checkpoint2.exists(), "Resumed model should be saved");

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_train_all_examples_with_progress() {
    let temp_dir = create_temp_dir();
    let examples = vec!["and", "or", "xor"];

    for example in examples {
        let output_path = temp_dir.path().join(format!("{}_model.json", example));

        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "neural-net-cli",
                "--",
                "train",
                "--example",
                example,
                "--epochs",
                "100",
                "--output",
                output_path.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to train");

        assert!(
            output.status.success(),
            "Training {} with progress should succeed",
            example
        );
        assert!(
            output_path.exists(),
            "Model file for {} should be created",
            example
        );
    }

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_progress_with_custom_learning_rate() {
    let temp_dir = create_temp_dir();
    let output_path = temp_dir.path().join("model.json");

    let output = Command::new("cargo")
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
            "--learning-rate",
            "0.3",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to train");

    assert!(
        output.status.success(),
        "Training with custom learning rate should show progress and complete"
    );

    // TempDir automatically cleans up when dropped
}

#[test]
fn test_progress_without_output_file() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "neural-net-cli",
            "--",
            "train",
            "--example",
            "or",
            "--epochs",
            "50",
        ])
        .output()
        .expect("Failed to train");

    assert!(
        output.status.success(),
        "Training without output should still show progress and complete"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Training") || stdout.contains("complete"),
        "Should show training status"
    );
}
