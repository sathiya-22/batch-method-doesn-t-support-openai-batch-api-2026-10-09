import os
import time
from typing import List, Dict, Any

# Conditional import for OpenAI or MockOpenAIClient
try:
    from openai import OpenAI
    # Use actual OpenAI types if available
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    # If openai is not installed, use dummy types and ensure MockOpenAIClient is available
    print("OpenAI SDK not found. Running with MockOpenAIClient.")
    from mock_openai_client import MockOpenAIClient as OpenAI # Alias for consistent usage
    from mock_openai_client import MockChatCompletionRequest as ChatCompletionRequest # Alias for consistent usage
    ChatCompletionMessageParam = Dict[str, str] # Dummy type

from openai_batch_submitter import OpenAIBatchSubmitter, ChatCompletionRequest


def create_sample_requests(num_requests: int) -> List[ChatCompletionRequest]:
    """Generates a list of sample ChatCompletionRequest objects."""
    requests = []
    for i in range(num_requests):
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": f"What is the capital of country {i}?"}
        ]
        requests.append(
            ChatCompletionRequest(
                messages=messages,
                model="gpt-3.5-turbo",
                custom_id=f"prompt-{i}"
            )
        )
    return requests

def main():
    # --- Client Initialization ---
    # [OPTIONAL] To use a real OpenAI client, uncomment the lines below and set OPENAI_API_KEY
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if openai_api_key:
    #     client = OpenAI(api_key=openai_api_key)
    # else:
    #     print("OPENAI_API_KEY not found. Using MockOpenAIClient.")
    #     client = OpenAI(fixture_path="fixtures/batch_responses.json")

    # Default to MockOpenAIClient for zero-API-key demonstration
    client = OpenAI(fixture_path="fixtures/batch_responses.json")
    print(f"Using client type: {type(client).__name__}")

    batch_submitter = OpenAIBatchSubmitter(client=client)

    # --- Create Sample Requests ---
    num_requests = 2
    sample_requests = create_sample_requests(num_requests)
    print(f"\nGenerated {len(sample_requests)} sample chat completion requests.")

    # --- Submit Batch ---
    try:
        batch_id = batch_submitter.submit_batch(sample_requests)
        print(f"Batch submitted successfully with ID: {batch_id}")

        # Simulate some delay before polling for the first time if using mock
        if isinstance(client, OpenAI) and client.__class__.__name__ == 'MockOpenAIClient':
            print("Simulating batch processing delay...")
            time.sleep(3) # Give mock time to transition status

        # --- Get Batch Results ---
        print("\nAttempting to retrieve batch results...")
        results = batch_submitter.get_batch_results(batch_id)

        print(f"\nSuccessfully retrieved {len(results)} results from batch '{batch_id}':")
        for res in results:
            custom_id = res.get('custom_id', 'N/A')
            response_content = res.get('response', {}).get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"  Custom ID: {custom_id}, Response: {response_content[:80]}...")

    except Exception as e:
        print(f"\nAn error occurred during batch processing: {e}")

if __name__ == "__main__":
    main()
