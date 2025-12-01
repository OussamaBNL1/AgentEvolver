# Env Service API Documentation

## Overview

This document describes every Env Service API endpoint: purpose, parameters, return schema, and multiple access examples. The service supports querying available tasks, creating and managing instances, executing steps, evaluating results, and releasing resources.

## Basic Information

- Default base URL: `http://localhost:8000`
- All endpoints use HTTP `POST`
- Data format: `JSON`
- Recommended client timeout: 150–350 seconds (randomized)

## Endpoints

### 1. Get Environment Profile

#### Description
Return configuration info for a given environment type, primarily a list of available task IDs.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| env_type | string | Yes | Environment type, e.g. `"appworld"` |
| split | string | No | Data split, defaults to `"train"` |
| params | dict | No | Extra parameters (environment-specific) |

#### Return
- Type: `List[str]`
- Meaning: list of task IDs

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient(base_url="http://localhost:8000")
task_ids = client.get_env_profile(env_type="appworld", split="train")
print(f"Available tasks: {task_ids}")
```

##### curl
```bash
curl -X POST http://localhost:8000/get_env_profile \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "params": {"split": "train"}
  }'
```

##### Raw HTTP
```
POST /get_env_profile HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "params": {"split": "train"}
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/get_env_profile"
data = {
    "env_type": "appworld",
    "params": {"split": "train"}
}

response = requests.post(url, json=data, timeout=350)
task_ids = response.json().get("data", [])
print(f"Available tasks: {task_ids}")
```

### 2. Get Tools Info

#### Description
Retrieve tool metadata for a given instance. Structure may vary by environment implementation.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| messages | dict | No | Optional message/context payload |
| params | dict | No | Extra parameters |

#### Return
- Type: `Any`
- Meaning: environment-specific tool information

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
tools_info = client.get_tools_info(instance_id=instance_id)
print(f"Tools info: {tools_info}")
```

##### curl
```bash
curl -X POST http://localhost:8000/get_info \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### Raw HTTP
```
POST /get_info HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/get_info"
data = {
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
tools_info = response.json().get("data", None)
print(f"Tools info: {tools_info}")
```

### 3. Create Instance

#### Description
Create a new instance using a specified environment type and task ID.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| env_type | string | Yes | Environment type |
| task_id | string | Yes | Task ID |
| instance_id | string | No | Custom instance ID (auto-generated if omitted) |
| params | dict | No | Extra parameters |

#### Return
- Type: `dict`
- Schema:
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # state messages
      "reward": float,               # current reward value
      "is_terminated": bool,         # termination flag
      "info": {"instance_id": str, "task_id": str}  # basic metadata
  }
  ```

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
env_type = "appworld"
task_id = "task-123"
init_response = client.create_instance(env_type, task_id)
print(f"Created instance: {init_response['info']['instance_id']}")
```

##### curl
```bash
curl -X POST http://localhost:8000/create \
  -H "Content-Type: application/json" \
  -d '{
    "env_type": "appworld",
    "task_id": "task-123",
    "instance_id": "optional-instance-id",
    "params": {}
  }'
```

##### Raw HTTP
```
POST /create HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "env_type": "appworld",
  "task_id": "task-123",
  "instance_id": "optional-instance-id",
  "params": {}
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/create"
data = {
    "env_type": "appworld",
    "task_id": "task-123",
    "instance_id": "optional-instance-id",
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
init_response = response.json().get("data", {})
print(f"Created instance: {init_response.get('info', {}).get('instance_id')}")
```

### 4. Step (Execute Action)

#### Description
Execute one action step against a running instance.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| action | dict | No | Action message `{ "role": str, "content": str }` |
| params | dict | No | Extra parameters |

#### Return
- Type: `dict`
- Schema:
  ```python
  {
      "state": [{"role": str, "content": str}, ...],  # new state after action
      "reward": float,               # updated reward
      "is_terminated": bool,         # termination flag
      "info": {"instance_id": str, "task_id": str}  # metadata
  }
  ```

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
action = {"role": "assistant", "content": "print('hello world')"}
result = client.step(instance_id, action)
print(f"Step result: {result}")
```

##### curl
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('\''hello world'\'')"},
    "params": {}
  }'
```

##### Raw HTTP
```
POST /step HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {"role": "assistant", "content": "print('hello world')"},
  "params": {}
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/step"
data = {
    "instance_id": "your-instance-id",
    "messages": {"role": "assistant", "content": "print('hello world')"},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
result = response.json().get("data", {})
print(f"Step result: {result}")
```

### 5. Evaluate Instance

#### Description
Evaluate a given instance and return a numeric score.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |
| messages | dict | No | Optional evaluation-related messages |
| params | dict | No | Extra parameters |

#### Return
- Type: `float`
- Meaning: evaluation score

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
score = client.evaluate(instance_id)
print(f"Evaluation score: {score}")
```

##### curl
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
  }'
```

##### Raw HTTP
```
POST /evaluate HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id",
  "messages": {},
  "params": {}
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/evaluate"
data = {
    "instance_id": "your-instance-id",
    "messages": {},
    "params": {}
}

response = requests.post(url, json=data, timeout=350)
score = response.json().get("data", 0.0)
print(f"Evaluation score: {score}")
```

### 6. Release Instance

#### Description
Release resources tied to an instance.

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| instance_id | string | Yes | Instance ID |

#### Return
- Type: `bool`
- Meaning: `True` if release succeeded, otherwise `False`

#### Access Examples

##### EnvClient
```python
from env_client import EnvClient

client = EnvClient()
instance_id = "your-instance-id"
success = client.release_instance(instance_id)
print(f"Instance released: {success}")
```

##### curl
```bash
curl -X POST http://localhost:8000/release \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "your-instance-id"
  }'
```

##### Raw HTTP
```
POST /release HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "instance_id": "your-instance-id"
}
```

##### Python (requests)
```python
import requests

url = "http://localhost:8000/release"
data = {
    "instance_id": "your-instance-id"
}

response = requests.post(url, json=data, timeout=350)
success = response.json().get("success", False)
print(f"Instance released: {success}")
```

## Full Workflow Example

Below is a complete workflow using the Env Service, from retrieving tasks to releasing an instance.

### EnvClient Workflow
```python
from env_client import EnvClient

def full_workflow_demo():
  # Initialize client
    client = EnvClient(base_url="http://localhost:8000")
    env_type = "appworld"
    
  # 1. Fetch environment profile and task list
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")
    
    if not task_ids:
        print("No tasks available, exiting.")
        return
    
  # 2. Create instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id)
    print("Initial state:", init_response)
    
    instance_id = init_response["info"]["instance_id"]
    print(f"Created instance: {instance_id}")
    
  # 3. Get tools info
    tools_info = client.get_tools_info(instance_id)
    print(f"Tools available: {tools_info}")
    
  # 4. Execute step
    action = {"role": "assistant", "content": "print('Hello from workflow!')"}
    step_result = client.step(instance_id, action)
    print(f"Step result: {step_result}")
    
  # 5. Evaluate instance
    score = client.evaluate(instance_id)
    print(f"Final score: {score}")
    
  # 6. Release instance
    release_success = client.release_instance(instance_id)
    print(f"Instance released successfully: {release_success}")

if __name__ == "__main__":
    full_workflow_demo()
```

## Error Handling

1. All client helper methods implement up to 3 automatic retries.
2. On repeated failure a safe fallback value (empty list / None / 0.0 / False) is returned.
3. Error details are appended to a log file; sanitized default path: `./logs/error.out` (override via `CLIENT_LOG_PATH`).
4. Set environment variable `CLIENT_LOG_PATH` to customize log location.

## Notes

1. Always release instances you no longer need to prevent resource leakage.
2. Allow sufficiently long timeouts; some environments perform heavy processing.
3. Capabilities and optional parameters differ by `env_type`; consult the specific environment docs.
4. If `is_terminated` is `True`, the episode has ended; do not issue further `step` calls.