import json
import time
import io
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock

class MockOpenAIClient:
    """
    A mock OpenAI client to simulate batch API interactions for testing without
    requiring an actual API key or network calls.
    """
    def __init__(self, fixture_path: str = "fixtures/batch_responses.json"):
        with open(fixture_path, 'r') as f:
            self.fixtures = json.load(f)
        self.files_uploaded: Dict[str, Any] = {}
        self.batches: Dict[str, Any] = {}
        self.file_counter = 0
        self.batch_counter = 0

    @property
    def files(self):
        mock_files = MagicMock()
        mock_files.create.side_effect = self._mock_files_create
        mock_files.retrieve_content.side_effect = self._mock_files_retrieve_content
        return mock_files

    @property
    def batches_api(self):
        mock_batches = MagicMock()
        mock_batches.create.side_effect = self._mock_batches_create
        mock_batches.retrieve.side_effect = self._mock_batches_retrieve
        return mock_batches

    def _mock_files_create(self, file: io.BytesIO, purpose: str = "batch"):
        self.file_counter += 1
        file_id = f"file-mock_{self.file_counter}"
        self.files_uploaded[file_id] = {
            "id": file_id,
            "object": "file",
            "bytes": len(file.getvalue()),
            "created_at": int(time.time()),
            "filename": f"batch_input_{self.file_counter}.jsonl",
            "purpose": purpose,
            "content": file.getvalue().decode('utf-8') # Store content for verification
        }
        print(f"[Mock] File uploaded: {file_id}")
        return self.files_uploaded[file_id]

    def _mock_files_retrieve_content(self, file_id: str):
        if file_id == self.fixtures["batch_retrieve_completed"]["output_file_id"]:
            print(f"[Mock] Retrieving content for output file: {file_id}")
            # Simulate the content of the output file
            return "\n".join([json.dumps(item) for item in self.fixtures["output_file_content"]])
        else:
            raise ValueError(f"Mock file content not found for {file_id}")

    def _mock_batches_create(self, input_file_id: str, endpoint: str = "/v1/chat/completions", completion_window: str = "24h"):
        self.batch_counter += 1
        batch_id = f"batch_mock_{self.batch_counter}"
        batch_data = {
            **self.fixtures["batch_create_response"],
            "id": batch_id,
            "input_file_id": input_file_id,
            "created_at": int(time.time()),
            "status": "validating"
        }
        self.batches[batch_id] = batch_data
        print(f"[Mock] Batch created: {batch_id}")
        return batch_data

    def _mock_batches_retrieve(self, batch_id: str):
        if batch_id not in self.batches:
            raise ValueError(f"Mock batch {batch_id} not found.")

        current_batch = self.batches[batch_id]

        if current_batch["status"] == "validating":
            print(f"[Mock] Batch {batch_id} status: validating -> in_progress")
            # Simulate transition to in_progress
            self.batches[batch_id] = {**current_batch, **self.fixtures["batch_retrieve_in_progress"], "status": "in_progress"}
        elif current_batch["status"] == "in_progress":
            print(f"[Mock] Batch {batch_id} status: in_progress -> completed")
            # Simulate transition to completed
            self.batches[batch_id] = {**current_batch, **self.fixtures["batch_retrieve_completed"], "status": "completed"}
        
        return self.batches[batch_id]

class MockChatCompletionRequest:
    """A simple mock for ChatCompletionRequest structure."""
    def __init__(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", custom_id: str = None):
        self.messages = messages
        self.model = model
        self.custom_id = custom_id

    def to_dict(self):
        data = {
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages
            }
        }
        if self.custom_id:
            data["custom_id"] = self.custom_id
        return data
