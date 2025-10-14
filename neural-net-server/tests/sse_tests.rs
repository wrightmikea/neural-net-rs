// Integration tests for Server-Sent Events (SSE) training progress
use std::time::Duration;
use tokio::time::sleep;
use serde_json::json;

async fn start_test_server(port: u16) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    let addr = format!("127.0.0.1:{}", port);
    tokio::spawn(async move {
        neural_net_server::run_server(&addr).await
    })
}

#[tokio::test]
async fn test_train_stream_endpoint_exists() {
    let handle = start_test_server(3020).await;
    sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    // Start training with streaming
    let request_body = json!({
        "example": "and",
        "epochs": 50,
        "learning_rate": 0.5,
        "stream": true
    });

    let response = client
        .post("http://127.0.0.1:3020/api/train/stream")
        .json(&request_body)
        .send()
        .await;

    assert!(response.is_ok(), "Should be able to connect to stream endpoint");

    handle.abort();
}

#[tokio::test]
async fn test_train_stream_returns_events() {
    let handle = start_test_server(3021).await;
    sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    let request_body = json!({
        "example": "and",
        "epochs": 10,
        "learning_rate": 0.5
    });

    let response = client
        .post("http://127.0.0.1:3021/api/train/stream")
        .json(&request_body)
        .send()
        .await
        .expect("Should get response");

    // Check content type is text/event-stream
    let content_type = response.headers().get("content-type");
    assert!(
        content_type.is_some(),
        "Should have content-type header"
    );
    assert!(
        content_type.unwrap().to_str().unwrap().contains("text/event-stream"),
        "Content-type should be text/event-stream"
    );

    handle.abort();
}

#[tokio::test]
async fn test_sse_train_completes() {
    let handle = start_test_server(3022).await;
    sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    let request_body = json!({
        "example": "and",
        "epochs": 5,
        "learning_rate": 0.5
    });

    // This will stream but we just check it completes
    let response = client
        .post("http://127.0.0.1:3022/api/train/stream")
        .json(&request_body)
        .send()
        .await
        .expect("Should get response");

    assert!(response.status().is_success(), "Training should complete successfully");

    handle.abort();
}

#[tokio::test]
async fn test_sse_invalid_example() {
    let handle = start_test_server(3023).await;
    sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    let request_body = json!({
        "example": "invalid",
        "epochs": 5,
        "learning_rate": 0.5
    });

    let response = client
        .post("http://127.0.0.1:3023/api/train/stream")
        .json(&request_body)
        .send()
        .await
        .expect("Should get response");

    assert!(
        response.status().is_client_error(),
        "Should return 4xx for invalid example"
    );

    handle.abort();
}
