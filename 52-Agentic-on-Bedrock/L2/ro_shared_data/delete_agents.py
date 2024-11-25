import boto3
import time

# Initialize the Bedrock Agent and IAM clients
bedrock_agent = boto3.client('bedrock-agent', region_name='us-west-2')
# iam = boto3.client('iam')

# List all agents
response = bedrock_agent.list_agents()
agents = response['agentSummaries']

# Iterate through agents and delete those with matching prefix
for agent in agents:

    print(f"Found: {agent['agentName']}")

    agent_id = agent['agentId']

    # Get the agent details to retrieve the IAM role ARN
    agent_details = bedrock_agent.get_agent(agentId=agent_id)
    
    role_arn = agent_details['agent']['agentResourceRoleArn']
    role_name = role_arn.split('/')[-1]

    # List all aliases for the agent
    alias_response = bedrock_agent.list_agent_aliases(agentId=agent_id)
    aliases = alias_response['agentAliasSummaries']

    # Delete each alias
    for alias in aliases:
        alias_id = alias['agentAliasId']
        print(f"Deleting alias: {alias['agentAliasName']} (ID: {alias_id})")
        try:
            bedrock_agent.delete_agent_alias(agentId=agent_id, agentAliasId=alias_id)
            print(f"Deletion initiated for alias: {alias['agentAliasName']}")
            
            # Wait for the alias to be deleted
            while True:
                try:
                    bedrock_agent.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)
                    print(f"Waiting for alias {alias['agentAliasName']} to be deleted...")
                    time.sleep(5)
                except bedrock_agent.exceptions.ResourceNotFoundException:
                    print(f"Alias {alias['agentAliasName']} has been successfully deleted.")
                    break
        except Exception as e:
            print(f"Error deleting alias {alias['agentAliasName']}: {str(e)}")

    print(f"Deleting agent: {agent['agentName']} (ID: {agent_id})")
    
    try:
        # Delete the agent
        bedrock_agent.delete_agent(agentId=agent_id)
        print(f"Deletion initiated for agent: {agent['agentName']}")
        
        # Wait for the agent to be deleted
        while True:
            try:
                bedrock_agent.get_agent(agentId=agent_id)
                print(f"Waiting for agent {agent['agentName']} to be deleted...")
                time.sleep(3)
            except bedrock_agent.exceptions.ResourceNotFoundException:
                print(f"Agent {agent['agentName']} has been successfully deleted.")
                break

        # Delete the IAM role
        # print(f"Deleting IAM role: {role_name}")
        # try:
        #     # Detach all policies from the role
        #     attached_policies = iam.list_attached_role_policies(RoleName=role_name)
        #     for policy in attached_policies['AttachedPolicies']:
        #         iam.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
        #         print(f"Detached policy: {policy['PolicyArn']}")

        #     # Delete policies from the role
        #     inline_policies = iam.list_role_policies(RoleName=role_name)
        #     for policy in inline_policies['PolicyNames']:
        #         iam.delete_role_policy(RoleName=role_name, PolicyName=policy)
        #         print(f"Deleted inline policy: {policy}")

        #     # Delete the role
        #     iam.delete_role(RoleName=role_name)
        #     print(f"IAM role {role_name} has been successfully deleted.")
        # except Exception as e:
        #     print(f"Error deleting IAM role {role_name}: {str(e)}")

    except Exception as e:
        print(f"Error deleting agent {agent['agentName']}: {str(e)}")

print("Agent reset process completed.")
