from env_service.env_client import EnvClient



def agent_test(task_id=0):



    client = EnvClient(base_url="http://localhost:8080")

    # Get task list
    env_type = "openworld"

    task_ids = client.get_env_profile(env_type, split='train')

    init_response = client.create_instance(env_type,
                                           str(task_ids[task_id]),
                                           )
    print("init state", init_response)
    instance_id = init_response["info"]["instance_id"]
    query = init_response["state"]
    print(f"Created instance {instance_id} with query: {query}")

    # Translated content (original Chinese removed for privacy/compliance)
    action = {
        'role': 'assistant',
        'content': (
            "To assess whether CATL (Contemporary Amperex Technology Co., Limited) stock is worth buying today, "
            "we need to consider fundamentals, technical indicators, and market sentiment. Since real-time market "
            "data isn't directly accessible in this context, we can proceed with these steps:\n\n"
            "1. Retrieve the latest CATL financial metrics (revenue, net profit, balance sheet items).\n"
            "2. Analyze recent price action and relevant technical indicators.\n"
            "3. Consider industry trends and the broader macroeconomic environment.\n\n"
            "First, we can call a tool to obtain CATL's 2022 financial data from Chinese listed company disclosures:\n\n"
            "```json\n[\n  {\n    \"tool_name\": \"sse_get_data_in_CHN\",\n    \"tool_args\": {\n      \"company_name\": \"CATL\",\n      \"year\": \"2022\"\n    }\n  }\n]\n```\n\n"
            "This call returns CATL's 2022 financials as an initial static snapshot. Remember to also review the latest "
            "news flow, analyst ratings, and your personal investment strategy before making any decision. Let me know "
            "if you want a deeper analysis." )
    }


    result = client.step(instance_id, action)
    print(f"Step result: {result}")

    return 0





if __name__ == "__main__":
    agent_test()
