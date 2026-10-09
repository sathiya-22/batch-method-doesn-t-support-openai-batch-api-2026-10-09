import io
import json
import time
from typing import Dict, Any, List, Union
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat_completion_request_message_param import ChatCompletionRequestMessageParam
except ImportError:
    # Define dummy types for standalone use without openai installed
    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass
        @property
        def files(self): return self
        @property
        def batches_api(self): return self
        def create(self, *args, **kwargs): pass
        def retrieve_content(self, *args, **kwargs): pass
        def retrieve(self, *args, **kwargs): pass

    ChatCompletionMessageParam = Dict[str, str]
    ChatCompletionRequestMessageParam = Dict[str, str]

class ChatCompletionRequest:
    """
    Represents a single chat completion request for the OpenAI Batch API.
    This structure aligns with what OpenAI expects for batch input.
    """
    def __init__(self,
                 messages: List[ChatCompletionRequestMessageParam],
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 custom_id: str = None):
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.custom_id = custom_id

    def to_dict(self) -> Dict[str, Any]:
        """Converts the request to the dictionary format expected by OpenAI Batch API."""
        body = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        request_data = {
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        if self.custom_id:
            request_data["custom_id"] = self.custom_id
        return request_data

class OpenAIBatchSubmitter:
    """
    A utility to submit multiple ChatCompletionRequests to OpenAI's Batch API
    and retrieve their results.
    """
    def __init__(self, client: OpenAI):
        self.client = client

    def _prepare_input_file(self, requests: List[ChatCompletionRequest]) -> io.BytesIO:
        """Prepares a JSONL file in memory from a list of ChatCompletionRequest objects."""
        jsonl_content = ""
        for i, req in enumerate(requests):
            req_data = req.to_dict()
            if not req_data.get("custom_id"):
                req_data["custom_id"] = f"prompt-{i}" # Ensure a custom_id for tracking
            jsonl_content += json.dumps(req_data) + "\n"
        return io.BytesIO(jsonl_content.encode('utf-8'))

    def submit_batch(self, requests: List[ChatCompletionRequest]) -> str:
        """
        Submits a list of ChatCompletionRequests as a single batch.

        Args:
            requests: A list of ChatCompletionRequest objects.

        Returns:
            The ID of the created batch.
        """
        if not requests:
            raise ValueError("No requests provided for batch submission.")

        print(f"Preparing batch input file for {len(requests)} requests...")
        input_file_io = self._prepare_input_file(requests)

        print("Uploading input file...")
        uploaded_file = self.client.files.create(file=input_file_io, purpose="batch")
        input_file_id = uploaded_file.id
        print(f"Input file uploaded: {input_file_id}")

        print("Creating batch job...")
        batch_job = self.client.batches_api.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch_job.id
        print(f"Batch job created: {batch_id}")
        return batch_id

    @retry(
        stop=stop_after_attempt(20), # Retry up to 20 times
        wait=wait_fixed(10),       # Wait 10 seconds between retries
        retry=retry_if_exception_type(Exception) # Retry on any exception, could be more specific
    )
    def _poll_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Polls the status of a batch job."""
        batch_status = self.client.batches_api.retrieve(batch_id)
        print(f"Polling batch '{batch_id}' status: {batch_status.status}")
        if batch_status.status not in ["completed", "failed", "cancelled"]:
            raise Exception(f"Batch {batch_id} not yet finished. Current status: {batch_status.status}")
        return batch_status

    def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Polls for batch completion and retrieves the results.

        Args:
            batch_id: The ID of the batch job.

        Returns:
            A list of dictionaries, where each dictionary contains the response
            for a single request in the batch, including its custom_id.
        """
        print(f"Waiting for batch '{batch_id}' to complete...")
        final_batch_status = self._poll_batch_status(batch_id)

        if final_batch_status.status != "completed":
            raise RuntimeError(f"Batch {batch_id} failed or was cancelled. Status: {final_batch_status.status}")

        output_file_id = final_batch_status.output_file_id
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} completed but no output file ID found.")

        print(f"Retrieving results from output file: {output_file_id}")
        output_content = self.client.files.retrieve_content(output_file_id)

        results = []
        for line in output_content.splitlines():
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not decode JSON line from batch output: {line}. Error: {e}")
        return results
