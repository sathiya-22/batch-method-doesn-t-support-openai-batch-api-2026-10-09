## Problem: Inefficient Batch Processing for Agentic Workloads

The `batch` method in popular agentic AI frameworks (like LangChain's ChatModels and Runnables) often does not fully leverage the native batch API capabilities offered by LLM providers, specifically OpenAI's batch API. This means that when developers want to send multiple independent requests to an LLM, they are typically processed sequentially or via simple concurrency, rather than being bundled into a single, optimized batch request by the provider.

This leads to:
1.  **Higher Latency:** Each request incurs its own network round trip and processing overhead.
2.  **Increased Cost:** Some providers, including OpenAI, offer cost savings for batch processing.
3.  **Suboptimal Resource Utilization:** The provider's infrastructure isn't utilized as efficiently as it could be.

This project addresses this by providing a dedicated `OpenAIBatchSubmitter` utility that explicitly utilizes the OpenAI Batch API for a list of `ChatCompletionRequest` objects. It allows agentic AI developers to submit multiple prompts as a single batch, retrieve the batch ID, and then poll for the results, ensuring native batch processing benefits are leveraged.

## Project Shape and Stack: Python Utility

A Python utility was chosen because:
*   Python is the de facto language for most agentic AI development and frameworks.
*   The OpenAI Python SDK is mature and widely used, making integration straightforward.
*   The problem is about optimizing an existing interaction pattern, fitting well within a library-style utility.

## Setup and Usage (Zero API Keys)

This prototype works with *zero* API keys for its core functionality of demonstrating the batch submission and polling mechanism. It uses a mock OpenAI client to simulate the API interactions.

1.  **Clone the repository:**
    ```bash
    git clone <this-repo-url>
    cd <this-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the example:**
    ```bash
    python main.py
    ```

You will see output demonstrating:
*   The creation of mock batch requests.
*   The submission of a batch (simulated).
*   The polling for batch completion (simulated with a delay).
*   The retrieval of mock batch results.

## Optional Real-LLM Adapter

This prototype *does not* include an optional real-LLM adapter for OpenAI, as the core problem is about the *mechanism* of using the OpenAI Batch API, not about generic LLM interaction. The `OpenAIBatchSubmitter` is designed to work directly with the real OpenAI Python SDK when provided with a genuine `OPENAI_API_KEY`.

To use it with a *real* OpenAI API key, you would:
1.  Set your `OPENAI_API_KEY` environment variable.
2.  Uncomment the relevant lines in `main.py` (marked with `[OPTIONAL]`) to use a real `OpenAI` client.
3.  Ensure your OpenAI account has access to the batch API.

The current setup with the mock client effectively demonstrates the API interaction flow without requiring any credentials.

---

### Key Files:
*   `main.py`: Entry point demonstrating the `OpenAIBatchSubmitter`.
*   `openai_batch_submitter.py`: The core utility class.
*   `mock_openai_client.py`: A mock OpenAI client for local testing without API keys.
*   `requirements.txt`: Python dependencies.
*   `fixtures/batch_responses.json`: Sample mock responses for batch operations.
