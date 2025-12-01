"""
Stage 2: Abstract task queries from triplet sequences
Build Output Format examples according to env_type: webshop | bfcl | appworld
"""

def _output_format_block(env_type: str) -> str:
        env = (env_type or "").lower()
        if env == "bfcl":
                return (
                        "<task>\n"
                        "Description: Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.\n"
                        "Query: Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.\n"
                        "Confidence: 1.0\n"
                        "ActionSequence: \n"
                        "# step0\ncd(folder='document')\n"
                        "# step1\nmkdir(dir_name='temp')\n"
                        "# step2\nmv(source='final_report.pdf', destination='temp')\n"
                        "</task>\n"
                )
        if env == "appworld":
                return (
                        "<task>\n"
                        "Description: Get the most-liked song in my Spotify playlists.\n"
                        "Query: What is the title of the most-liked song in my Spotify playlists.\n"
                        "Confidence: 1.0\n"
                        "ActionSequence: \n"
                        "# step0\n```python\\nprint(apis.api_docs.show_app_descriptions())\\n```\n"
                        "# step1\n```python\\nprint(apis.api_docs.show_api_descriptions(app_name='supervisor'))\\n```\n"
                        "# step2\n```python\\nprint(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))\\n```\n"
                        "# step3\n```python\\nprint(apis.supervisor.show_account_passwords())\npasswords = apis.supervisor.show_account_passwords()\\n```\n"
                        "# step4\n```python\\nspotify_password = [account_password for account_password in passwords if account_password[\"account_name\"] == \"spotify\"][0][\"password\"]\nprint(spotify_password)\\n```\n"
                        "# step5\n```python\\nprint(apis.api_docs.show_api_descriptions(app_name='spotify'))\\n```\n"
                        "# step6\n```python\\nprint(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))\\n```\n"
                        "# step7\n```python\\nprint(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))\nprint(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_profile'))\\n```\n"
                        "# step8\n```python\\nemail = apis.supervisor.show_profile()['email']\naccess_token = apis.spotify.login(username=email, password=spotify_password)['access_token']\nplaylist_0 = apis.spotify.show_playlist_library(page_index=0, access_token=access_token)\nprint(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_song'))\nlike_count = apis.spotify.show_song(song_id=136)['like_count']\\n```\n"
                        "# step9\n```python\\npage_index = 0\nsong_ids_all = []\nwhile True:\n    playlists = apis.spotify.show_playlist_library(page_index=page_index, access_token=access_token)\n    if not playlists:\n        break\n    for _ in playlists:\n        song_ids_all.extend(_['song_ids'])\n    page_index += 1\nprint(song_ids_all)\n\nmax_id = -1\nmax_like_count = 0\nfor _ in song_ids_all:\n    like_count = apis.spotify.show_song(song_id=_)['like_count']\n    max_like_count = max(max_like_count, like_count)\n    if max_like_count == like_count:\n        max_id = _\nanswer = apis.spotify.show_song(song_id=max_id)['title']\nprint(answer)\napis.supervisor.complete_task(answer=answer)\\n```\n"
                        "</task>\n"
                )
        # default: webshop
        return (
                "<task>\n"
                "Description: something\n"
                "Query: something\n"
                "Confidence: 1.0\n"
                "ActionSequence: \n"
                "<action>\n"
                "\\\boxed{click[something]} \n"
                "\\\boxed{search[something]}\n"
                "</action>\n"
                "</task>\n"
        )


def build_task_extraction_system_prompt(env_type: str) -> str:
        return f"""
You are a *Task Abstraction Expert*. Your specialty is to inspect an agent’s
interaction history and distill concrete, goal-oriented tasks from it.

========================  YOUR JOB  ========================
1. Inspect the interaction tuples (history, action, observation).
2. Identify the specific goal or task the agent is attempting to achieve.
3. Abstract each goal into a clear, concise **task description**, a **query**
     (suitable for search or training), and the **minimal action sequence**
     that successfully completes the task.

=====================  ABSTRACTION RULES  ==================
• Focus on clear, goal-directed behaviour; ignore purely random exploration. 
• Please include as many steps as possible in ActionSequence. 
• Group similar behaviour patterns into the same task.  
• Every task must have **at least one** action sequence that was executed
    successfully.  
• Each task needs an explicit completion criterion.  
• All actions listed in an action sequence must be valid and directly
    executable by the agent.
• All actions listed in an action sequence must be included in the available APIs of the current environment state.
• Ensure that all actions listed in an action sequence are combined into a minimum sequence from the initial state of the environment to the completion of the task. No additional information or skipped steps are allowed.
• The ActionSequence of the query should have at least 4 steps.

========================  OUTPUT FORMAT  ===================
For every task you identify, output exactly one block in the form below:

{_output_format_block(env_type)}
"""
# The following are example tasks kept as reference examples.


##bfcl
# <task>
# Description: Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.
# Query: Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.
# Confidence: 1.0
# ActionSequence: 
# # step0
# cd(folder='document')
# # step1
# mkdir(dir_name='temp')
# # step2
# mv(source='final_report.pdf', destination='temp')
# </task>

##webshop
# <task>
# Description: something
# Query: something
# Confidence: 1.0
# ActionSequence: 
# <action>
# \\boxed{click[something]} 
# \\boxed{search[something]}
# </action>
# </task>

##appworld
# <task>
# Description: Get the most-liked song in my Spotify playlists.
# Query: What is the title of the most-liked song in my Spotify playlists.
# Confidence: 1.0
# ActionSequence: 
# # step0
# ```python\nprint(apis.api_docs.show_app_descriptions())\n```
# # step1
# ```python\nprint(apis.api_docs.show_api_descriptions(app_name='supervisor'))\n```
# # step2
# ```python\nprint(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))\n```
# # step3
# ```python\nprint(apis.supervisor.show_account_passwords())
# passwords = apis.supervisor.show_account_passwords()\n```
# # step4
# ```python\nspotify_password = [account_password for account_password in passwords if account_password["account_name"] == "spotify"][0]["password"]
# print(spotify_password)\n```
# # step5
# ```python\nprint(apis.api_docs.show_api_descriptions(app_name='spotify'))\n```
# # step6
# ```python\nprint(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))\n```
# # step7
# ```python\nprint(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
# print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_profile'))\n```
# # step8
# ```python\nemail = apis.supervisor.show_profile()['email']
# access_token = apis.spotify.login(username=email, password=spotify_password)['access_token']
# playlist_0 = apis.spotify.show_playlist_library(page_index=0, access_token=access_token)
# print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_song'))
# like_count = apis.spotify.show_song(song_id=136)['like_count']\n```
# # step9
# ```python\npage_index = 0
# song_ids_all = []
# while True:
#     playlists = apis.spotify.show_playlist_library(page_index=page_index, access_token=access_token)
#     if not playlists:
#         break
#     for _ in playlists:
#         song_ids_all.extend(_['song_ids'])
#     page_index += 1
# print(song_ids_all)

# max_id = -1
# max_like_count = 0
# for _ in song_ids_all:
#     like_count = apis.spotify.show_song(song_id=_)['like_count']
#     max_like_count = max(max_like_count, like_count)
#     if max_like_count == like_count:
#         max_id = _
# answer = apis.spotify.show_song(song_id=max_id)['title']
# print(answer)
# apis.supervisor.complete_task(answer=answer)\n```
# </task>

def get_task_extraction_prompt(triplets: list, exploration_memory=None, env_discription=None, env_type: str = "webshop") -> tuple:
    """Get the full prompt for task abstraction"""
    
    # Build triplet sequence description
    triplets_str = ""
    for i, triplet in enumerate(triplets, 1):
        history = triplet.get('history', [])
        action = triplet.get('action', '')
        observation = triplet.get('observation', '')[:1000]  # Truncate to 1000 chars
        reward = triplet.get('reward', 0.0)
        
        # Simplify history display
        # history_brief = history[-2:] if len(history) > 2 else history
        # history_text = " -> ".join(history_brief) if history_brief else "None"
        history_text = history

        #   History: {history_text}
        triplets_str += f"""
Triplet {i}:
  Action: {action}
  Observation: {observation}
"""
    
    user_prompt = f"""
The environmental information is as follows:

{env_discription}

Please analyze the following agent interaction sequence and abstract specific tasks from it:

{triplets_str}

The tasks that have been generated before are as follows. Please try not to repeat them:

{exploration_memory}

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""
    
    system_prompt = build_task_extraction_system_prompt(env_type)
    return system_prompt, user_prompt


def parse_tasks_from_response(response: str) -> list:
    """Parse a list of tasks from the model response"""
    tasks = []
    try:
        import re
        
        # Find content inside all <task>...</task> blocks
        task_matches = re.findall(r'<task>(.*?)</task>', response, re.DOTALL)
        
        for task_content in task_matches:
            task_info = {}
            lines = task_content.strip().split('\n')
            
            # Variables for collecting ActionSequence
            collecting_action_sequence = False
            action_sequence_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Description:'):
                    task_info['description'] = line.replace('Description:', '').strip()
                elif line.startswith('Query:'):
                    task_info['query'] = line.replace('Query:', '').strip()
                elif line.startswith('Confidence:'):
                    confidence_str = line.replace('Confidence:', '').strip()
                    try:
                        task_info['confidence'] = float(confidence_str)
                    except ValueError:
                        task_info['confidence'] = 1.0
                elif line.startswith('ActionSequence:'):
                    # ActionSequence may span multiple lines; start collecting
                    collecting_action_sequence = True
                    action_sequence_text = line.replace('ActionSequence:', '').strip()
                    if action_sequence_text:
                        action_sequence_lines.append(action_sequence_text)
                elif collecting_action_sequence:
                    # Stop collecting if a new field starts
                    if (line.startswith('Description:') or 
                        line.startswith('Query:') or 
                        line.startswith('Confidence:')):
                        collecting_action_sequence = False
                    else:
                        action_sequence_lines.append(line)
            
            # Assemble ActionSequence
            if action_sequence_lines:
                task_info['gt'] = '\n'.join(action_sequence_lines)
            else:
                task_info['gt'] = ""
            
            # Check required fields
            if 'description' in task_info and 'query' in task_info:
                if 'confidence' not in task_info:
                    task_info['confidence'] = 1.0
                tasks.append(task_info)
                
    except Exception as e:
        print(f"Error parsing tasks: {e}")
    
    return tasks
