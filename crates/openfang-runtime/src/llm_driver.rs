//! LLM driver trait and types.
//!
//! Abstracts over multiple LLM providers (Anthropic, OpenAI, Ollama, etc.).

use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Message, StopReason, TokenUsage};
use openfang_types::tool::{ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for LLM driver operations.
#[derive(Error, Debug)]
pub enum LlmError {
    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    Http(String),
    /// API returned an error.
    #[error("API error ({status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },
    /// Rate limited — should retry after delay.
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        /// How long to wait before retrying.
        retry_after_ms: u64,
    },
    /// Response parsing failed.
    #[error("Parse error: {0}")]
    Parse(String),
    /// No API key configured.
    #[error("Missing API key: {0}")]
    MissingApiKey(String),
    /// Model overloaded.
    #[error("Model overloaded, retry after {retry_after_ms}ms")]
    Overloaded {
        /// How long to wait before retrying.
        retry_after_ms: u64,
    },
    /// Authentication failed (invalid/missing API key).
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    /// Model not found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// A request to an LLM for completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Available tools the model can use.
    pub tools: Vec<ToolDefinition>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// System prompt (extracted from messages for APIs that need it separately).
    pub system: Option<String>,
    /// Extended thinking configuration (if supported by the model).
    pub thinking: Option<openfang_types::config::ThinkingConfig>,
}

/// A response from an LLM completion.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The content blocks in the response.
    pub content: Vec<ContentBlock>,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Tool calls extracted from the response.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage statistics.
    pub usage: TokenUsage,
}

impl CompletionResponse {
    /// Extract text content from the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                ContentBlock::Thinking { .. } => None,
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Check if the response has any meaningful content (including Thinking blocks).
    /// Used to distinguish true empty responses from thinking-only responses.
    pub fn has_any_content(&self) -> bool {
        self.content.iter().any(|block| match block {
            ContentBlock::Text { text, .. } => !text.is_empty(),
            ContentBlock::Thinking { thinking, .. } => !thinking.is_empty(),
            ContentBlock::RedactedThinking { data } => !data.is_empty(),
            ContentBlock::ToolUse { .. } | ContentBlock::Image { .. } => true,
            _ => false,
        })
    }
}

/// Events emitted during streaming LLM completion.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text content.
    TextDelta { text: String },
    /// A tool use block has started.
    ToolUseStart { id: String, name: String },
    /// Incremental JSON input for an in-progress tool use.
    ToolInputDelta { text: String },
    /// A tool use block is complete with parsed input.
    ToolUseEnd {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Incremental thinking/reasoning text.
    ThinkingDelta { text: String },
    /// The entire response is complete.
    ContentComplete {
        stop_reason: StopReason,
        usage: TokenUsage,
    },
    /// Agent lifecycle phase change (for UX indicators).
    PhaseChange {
        phase: String,
        detail: Option<String>,
    },
    /// Tool execution completed with result (emitted by agent loop, not LLM driver).
    ToolExecutionResult {
        id: String,
        name: String,
        result_preview: String,
        is_error: bool,
    },
}

/// Trait for LLM drivers.
#[async_trait]
pub trait LlmDriver: Send + Sync {
    /// Send a completion request and get a response.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Stream a completion request, sending incremental events to the channel.
    /// Returns the full response when complete. Default wraps `complete()`.
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let response = self.complete(request).await?;
        let text = response.text();
        if !text.is_empty() {
            let _ = tx.send(StreamEvent::TextDelta { text }).await;
        }
        let _ = tx
            .send(StreamEvent::ContentComplete {
                stop_reason: response.stop_reason,
                usage: response.usage,
            })
            .await;
        Ok(response)
    }
}

/// Configuration for creating an LLM driver.
#[derive(Clone, Serialize, Deserialize)]
pub struct DriverConfig {
    /// Provider name.
    pub provider: String,
    /// API key.
    pub api_key: Option<String>,
    /// Base URL override.
    pub base_url: Option<String>,
    /// Skip interactive permission prompts (Claude Code provider only).
    ///
    /// When `true`, adds `--dangerously-skip-permissions` to the spawned
    /// `claude` CLI.  Defaults to `true` because OpenFang runs as a daemon
    /// with no interactive terminal, so permission prompts would block
    /// indefinitely.  OpenFang's own capability / RBAC layer already
    /// restricts what agents can do, making this safe.
    #[serde(default = "default_skip_permissions")]
    pub skip_permissions: bool,

    /// Per-message subprocess turn timeout in seconds.
    ///
    /// Caps how long the runtime will wait for a single CLI subprocess turn
    /// (one message round-trip) before killing the process and reporting a
    /// timeout failure. When unset, the driver's own default is used
    /// (currently 300s). Long-context Opus calls with heavy tool surfaces
    /// routinely take >4 minutes, so users running large prompts may want
    /// to bump this to 480–600s.
    ///
    /// Can also be overridden at runtime via the
    /// `OPENFANG_SUBPROCESS_TIMEOUT_SECS` env var, which wins over both
    /// this field and the driver default.
    ///
    /// **Scope:** Currently only honored by `provider = "claude-code"`.
    /// Other providers (`default`, `qwen-code`, `openai`, `bedrock`, etc.)
    /// accept the field for forward-compatibility but silently ignore it
    /// today. As additional subprocess-based drivers are added, they will
    /// opt in to this field individually.
    #[serde(default)]
    pub subprocess_timeout_secs: Option<u64>,
}

fn default_skip_permissions() -> bool {
    true
}

/// SECURITY: Custom Debug impl redacts the API key.
impl std::fmt::Debug for DriverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("skip_permissions", &self.skip_permissions)
            .field("subprocess_timeout_secs", &self.subprocess_timeout_secs)
            .finish()
    }
}

/// A wrapper driver that intercepts completion requests and pauses/sleeps
/// until the allowed hours window starts if currently outside.
pub struct TimeWindowedDriver {
    pub inner: std::sync::Arc<dyn LlmDriver>,
    pub window_config: std::sync::Arc<std::sync::RwLock<openfang_types::config::InferenceWindowConfig>>,
}

impl TimeWindowedDriver {
    pub fn new(
        inner: std::sync::Arc<dyn LlmDriver>,
        window_config: std::sync::Arc<std::sync::RwLock<openfang_types::config::InferenceWindowConfig>>,
    ) -> Self {
        Self { inner, window_config }
    }

    async fn wait_for_allowed_hours(&self, tx: Option<&tokio::sync::mpsc::Sender<StreamEvent>>) {
        let mut first_pause = true;
        loop {
            let config = {
                let guard = self.window_config.read().unwrap();
                guard.clone()
            };

            if !config.enabled {
                break;
            }

            // Get current time in specified timezone (or local time)
            let now = chrono::Utc::now();
            use chrono::Timelike;
            let hour = if let Some(ref tz_str) = config.timezone {
                match tz_str.parse::<chrono_tz::Tz>() {
                    Ok(tz) => now.with_timezone(&tz).hour(),
                    Err(_) => now.with_timezone(&chrono::Local).hour(),
                }
            } else {
                now.with_timezone(&chrono::Local).hour()
            };

            if config.is_allowed_hour(hour) {
                if !first_pause {
                    if let Some(tx_channel) = tx {
                        let _ = tx_channel.send(StreamEvent::PhaseChange {
                            phase: "running".to_string(),
                            detail: Some("Inference window opened. Resuming request...".to_string()),
                        }).await;
                    }
                }
                break;
            }

            if first_pause {
                first_pause = false;
                if let Some(tx_channel) = tx {
                    let detail = format!(
                        "Inference paused: outside allowed hours ({} to {}). Re-opens at {}:00.",
                        config.start_hour, config.end_hour, config.start_hour
                    );
                    let _ = tx_channel.send(StreamEvent::PhaseChange {
                        phase: "paused".to_string(),
                        detail: Some(detail),
                    }).await;
                }
            }

            tracing::info!(
                "Inference paused: outside allowed hours ({} to {}). Current hour is {}. Sleeping 30s...",
                config.start_hour, config.end_hour, hour
            );
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    }
}

#[async_trait]
impl LlmDriver for TimeWindowedDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.wait_for_allowed_hours(None).await;
        self.inner.complete(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        self.wait_for_allowed_hours(Some(&tx)).await;
        self.inner.stream(request, tx).await
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_response_text() {
        let response = CompletionResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Hello ".to_string(),
                    provider_metadata: None,
                },
                ContentBlock::Text {
                    text: "world!".to_string(),
                    provider_metadata: None,
                },
            ],
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage::default(),
        };
        assert_eq!(response.text(), "Hello world!");
    }

    #[test]
    fn test_stream_event_clone() {
        let event = StreamEvent::TextDelta {
            text: "hello".to_string(),
        };
        let cloned = event.clone();
        assert!(matches!(cloned, StreamEvent::TextDelta { text } if text == "hello"));
    }

    #[test]
    fn test_stream_event_variants() {
        let events: Vec<StreamEvent> = vec![
            StreamEvent::TextDelta {
                text: "hi".to_string(),
            },
            StreamEvent::ToolUseStart {
                id: "t1".to_string(),
                name: "web_search".to_string(),
            },
            StreamEvent::ToolInputDelta {
                text: "{\"q".to_string(),
            },
            StreamEvent::ToolUseEnd {
                id: "t1".to_string(),
                name: "web_search".to_string(),
                input: serde_json::json!({"query": "rust"}),
            },
            StreamEvent::ContentComplete {
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            },
        ];
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn test_default_stream_sends_events() {
        use tokio::sync::mpsc;

        struct FakeDriver;

        #[async_trait]
        impl LlmDriver for FakeDriver {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Hello!".to_string(),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage {
                        input_tokens: 5,
                        output_tokens: 3,
                    },
                })
            }
        }

        let driver = FakeDriver;
        let (tx, mut rx) = mpsc::channel(16);
        let request = CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        };

        let response = driver.stream(request, tx).await.unwrap();
        assert_eq!(response.text(), "Hello!");

        // Should receive TextDelta then ContentComplete
        let ev1 = rx.recv().await.unwrap();
        assert!(matches!(ev1, StreamEvent::TextDelta { text } if text == "Hello!"));

        let ev2 = rx.recv().await.unwrap();
        assert!(matches!(
            ev2,
            StreamEvent::ContentComplete {
                stop_reason: StopReason::EndTurn,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_time_windowed_driver_disabled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TestDriver {
            call_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl LlmDriver for TestDriver {
            async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "done".to_string(),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(TestDriver {
            call_count: call_count.clone(),
        });

        // Config disabled
        let window_config = Arc::new(std::sync::RwLock::new(
            openfang_types::config::InferenceWindowConfig {
                enabled: false,
                start_hour: 9,
                end_hour: 17,
                timezone: None,
            },
        ));

        let driver = TimeWindowedDriver::new(inner, window_config);
        let request = CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        };

        let response = driver.complete(request).await.unwrap();
        assert_eq!(response.text(), "done");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_time_windowed_driver_enabled_in_window() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TestDriver {
            call_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl LlmDriver for TestDriver {
            async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "done".to_string(),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(TestDriver {
            call_count: call_count.clone(),
        });

        // Get current system local hour to dynamically configure the window to be open.
        let now = chrono::Utc::now().with_timezone(&chrono::Local);
        use chrono::Timelike;
        let current_hour = now.hour();
        let start_hour = current_hour;
        let end_hour = (current_hour + 1) % 24;

        // Config enabled, window covering current hour
        let window_config = Arc::new(std::sync::RwLock::new(
            openfang_types::config::InferenceWindowConfig {
                enabled: true,
                start_hour,
                end_hour,
                timezone: None,
            },
        ));

        let driver = TimeWindowedDriver::new(inner, window_config);
        let request = CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        };

        let response = driver.complete(request).await.unwrap();
        assert_eq!(response.text(), "done");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_time_windowed_driver_feedback_gated() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TestDriver {
            call_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl LlmDriver for TestDriver {
            async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "done".to_string(),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage::default(),
                })
            }
        }

        let call_count = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(TestDriver {
            call_count: call_count.clone(),
        });

        // Get current system local hour.
        let now = chrono::Utc::now().with_timezone(&chrono::Local);
        use chrono::Timelike;
        let current_hour = now.hour();

        // Create a closed window (starts in 2 hours, lasts 1 hour).
        let start_hour = (current_hour + 2) % 24;
        let end_hour = (current_hour + 3) % 24;

        let window_config = Arc::new(std::sync::RwLock::new(
            openfang_types::config::InferenceWindowConfig {
                enabled: true,
                start_hour,
                end_hour,
                timezone: None,
            },
        ));

        let driver = Arc::new(TimeWindowedDriver::new(inner, window_config.clone()));
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);

        let request = CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        };

        // Spawn stream call in a background thread since it sleeps
        let driver_clone = driver.clone();
        let request_clone = request.clone();
        let handle = tokio::spawn(async move {
            driver_clone.stream(request_clone, tx).await
        });

        // 1. Should immediately receive a paused phase change event
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("Timeout waiting for paused event")
            .expect("Channel closed");

        if let StreamEvent::PhaseChange { phase, detail } = event {
            assert_eq!(phase, "paused");
            assert!(detail.unwrap().contains("Inference paused"));
        } else {
            panic!("Expected PhaseChange event");
        }

        // Confirm inner driver has not been invoked
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        // 2. Hot-reload/dynamically open the window
        {
            let mut guard = window_config.write().unwrap();
            guard.start_hour = current_hour;
            guard.end_hour = (current_hour + 1) % 24;
        }

        // 3. Should receive a running phase change event when it wakes up
        let event2 = tokio::time::timeout(std::time::Duration::from_secs(35), rx.recv())
            .await
            .expect("Timeout waiting for running event")
            .expect("Channel closed");

        if let StreamEvent::PhaseChange { phase, detail } = event2 {
            assert_eq!(phase, "running");
            assert!(detail.unwrap().contains("Resuming request"));
        } else {
            panic!("Expected PhaseChange event");
        }

        // 4. Then we should receive text delta and content complete from inner
        let event3 = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("Timeout waiting for text delta")
            .expect("Channel closed");

        assert!(matches!(event3, StreamEvent::TextDelta { text } if text == "done"));

        let event4 = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("Timeout waiting for content complete")
            .expect("Channel closed");

        assert!(matches!(event4, StreamEvent::ContentComplete { .. }));

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result.text(), "done");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
