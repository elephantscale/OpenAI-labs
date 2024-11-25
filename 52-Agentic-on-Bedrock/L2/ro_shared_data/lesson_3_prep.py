import sys
import os
import boto3
import boto3
from helper import *

# print("*" * 50)
print("Lesson 3 Prep")
# print("*" * 50)

def create_action_group(bedrock_agent, agent_id, lambda_function_arn):
    response = bedrock_agent.create_agent_action_group(
        actionGroupName='customer-support-actions',
        agentId=agent_id,
        functionSchema={
            'functions': [
                {
                    'name': 'customerId',
                    'description': 'Get a customer ID given available details. At least one parameter must be sent to the function. This is private information and must not be given to the user.',
                    'parameters': {
                        'email': {'description': 'Email address', 'required': False, 'type': 'string'},
                        'name': {'description': 'First and last name', 'required': False, 'type': 'string'},
                        'phone': {'description': 'Phone number', 'required': False, 'type': 'string'},
                    }
                },
                {
                    'name': 'sendToSupport',
                    'description': 'Send a message to the support team, used for service escalation.',
                    'parameters': {
                        'custId': {'description': 'customer ID', 'required': True, 'type': 'string'},
                        'supportSummary': {'description': 'Summary of the support request', 'required': True, 'type': 'string'}
                    }
                }
            ]
        },
        actionGroupExecutor={'lambda': lambda_function_arn},
        agentVersion='DRAFT',
    )
    return response['agentActionGroup']['actionGroupId']

def main():
    lesson = sys.argv[1] if len(sys.argv) > 1 else None
    
    region_name = 'us-west-2'
    agent_id = os.environ['BEDROCK_AGENT_ID']
    agent_alias_id = os.environ['BEDROCK_AGENT_ALIAS_ID']
    lambda_function_arn = os.environ['LAMBDA_FUNCTION_ARN']

    bedrock_agent = boto3.client(service_name='bedrock-agent', region_name=region_name)

    action_group_id = create_action_group(bedrock_agent, agent_id, lambda_function_arn)
    # print(f"Action group created successfully. {action_group_id}")

    wait_for_action_group_status(agentId=agent_id, actionGroupId=action_group_id, targetStatus='ENABLED')

    bedrock_agent.prepare_agent(agentId=agent_id)
    wait_for_agent_status(agentId=agent_id, targetStatus='PREPARED')

    bedrock_agent.update_agent_alias(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        agentAliasName='MyAgentAlias',
    )
    wait_for_agent_alias_status(agentId=agent_id, agentAliasId=agent_alias_id, targetStatus='PREPARED')

    # print("Agent alias updated successfully.")
    os.environ['ACTION_GROUP_ID'] = action_group_id

if __name__ == "__main__":
    main()