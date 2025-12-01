from env_service.env_client import EnvClient


# Usage example
def main():
    client = EnvClient(base_url="http://localhost:8080")

    # Get task list
    env_type = "appworld"
    task_ids = client.get_env_profile(env_type)
    print(f"Available tasks: {task_ids}")

    # Create instance
    task_id = task_ids[0]
    init_response = client.create_instance(env_type, task_id, params={"prompt": True})

    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # Get environment information

    tool_info = client.get_tools_info(instance_id, params={"prompt": False})
    print(f"env information is  {instance_id} : {tool_info}")

    # Execute action
    # For the appworld dataset, the action must be returned as a string containing a ```python``` code block

    action = {
        "role": "assistant",
        "content": "```python\nprint('hello appworld!!')\n```",
    }
    # action = {
    #     "role": "assistant",
    #     "content": "```python\nprint(apis.api_docs.show_api_doc(app_name='''phone'''))\n```",
    # }
    result = client.step(instance_id, action)
    print(f"Step result: {result}")



    action = {
        "role": "assistant",
        "content": "print('hello appworld!!')",
    }
    # action = {
    #     "role": "assistant",
    #     "content": "```python\nprint(apis.api_docs.show_api_doc(app_name='''phone'''))\n```",
    # }
    result = client.step(instance_id, action)
    print(f"Step result: {result}")


    # action = {
    #     "role": "assistant",
    #     "content": "123",
    #     "tool_calls": [
    #         {
    #             "id": "",
    #             "arguments": "{\"code\": \"print(apis.api_docs.show_api_descriptions(app_name='''phone'''))\"}",
    #             "name": "appworld",
    #             "type": "tool",
    #             "index": 0,
    #         }
    #     ],
    # }
    # action = {
    #     "content": 'It appears that the workout note has been successfully identified and the playlist songs have been extracted from the note. \n\nNow, we have the playlist name and songs. Let\'s log into Spotify, find the playlist, and start it.\n\nCode:\n```python\naccess_token = ...\nplaylist_id = ...\n\n# Find the playlist\nplaylist_response = apis.spotify.show_playlist_library(access_token=access_token, page_index=0)\nplaylists = playlist_response["playlists"]\n\nfound_playlist = next((playlist for playlist in playlists if playlist["name"] == playlist_name), None)\nif found_playlist:\n    playlist_id = found_playlist["id"]\n    print(f"Found the playlist with id: {playlist_id}")\nelse:\n    print("Playlist not found.")\n    apis.supervisor.complete_task()  # Mark the task incomplete as we couldn\'t find the playlist\n\nif playlist_id:\n    # Start playing the playlist\n    start_playlist_response = apis.spotify.start_playlist(playlist_id=playlist_id, access_token=access_token)\n    print(start_playlist_response)\n    api.supervisor.complete_task()  # Mark the task complete since we\'ve started the playlist\n```',
    #     "role": "assistant",
    #     "tool_calls": [],
    # }

    # result = client.step(instance_id, action, params={"extra_info": {"name": "123"}})
    # print(f"Step result: {result}")

    # Evaluate
    score = client.evaluate(instance_id, messages={}, params={"sparse": True})
    print(f"Evaluation score: {score}")

    # Release instance
    success = client.release_instance(instance_id)
    print(f"Instance released: {success}")


if __name__ == "__main__":
    main()
