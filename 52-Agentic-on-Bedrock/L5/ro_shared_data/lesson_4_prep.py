import sys
import boto3
import os
from helper import *

# print("*" * 50)
print("Lesson 4 Prep")
# print("*" * 50)

def setup_bedrock_agent():
    bedrock_agent = boto3.client('bedrock-agent', region_name='us-west-2')
    
    update_customer_support_actions(bedrock_agent)
    create_code_interpreter_action(bedrock_agent)
    prepare_and_update_agent(bedrock_agent)

def update_customer_support_actions(bedrock_agent):
    response = bedrock_agent.update_agent_action_group(
        actionGroupName='customer-support-actions',
        actionGroupState='ENABLED',
        actionGroupId=os.environ['ACTION_GROUP_ID'],
        agentId=os.environ['BEDROCK_AGENT_ID'],
        agentVersion='DRAFT',
        actionGroupExecutor={'lambda': os.environ['LAMBDA_FUNCTION_ARN']},
        functionSchema={'functions': get_function_schema()}
    )
    
    wait_for_action_group_status(
        agentId=os.environ['BEDROCK_AGENT_ID'],
        actionGroupId=response['agentActionGroup']['actionGroupId']
    )

def create_code_interpreter_action(bedrock_agent):
    response = bedrock_agent.create_agent_action_group(
        actionGroupName='CodeInterpreterAction',
        actionGroupState='ENABLED',
        agentId=os.environ['BEDROCK_AGENT_ID'],
        agentVersion='DRAFT',
        parentActionGroupSignature='AMAZON.CodeInterpreter'
    )
    
    wait_for_action_group_status(
        agentId=os.environ['BEDROCK_AGENT_ID'],
        actionGroupId=response['agentActionGroup']['actionGroupId']
    )

def prepare_and_update_agent(bedrock_agent):
    bedrock_agent.prepare_agent(agentId=os.environ['BEDROCK_AGENT_ID'])
    wait_for_agent_status(
        agentId=os.environ['BEDROCK_AGENT_ID'],
        targetStatus='PREPARED'
    )

    bedrock_agent.update_agent_alias(
        agentId=os.environ['BEDROCK_AGENT_ID'],
        agentAliasId=os.environ['BEDROCK_AGENT_ALIAS_ID'],
        agentAliasName='test',
    )
    wait_for_agent_alias_status(
        agentId=os.environ['BEDROCK_AGENT_ID'],
        agentAliasId=os.environ['BEDROCK_AGENT_ALIAS_ID'],
        targetStatus='PREPARED'
    )

def get_function_schema():
    return [
        {
            'name': 'customerId',
            'description': 'Get a customer ID given available details. At least one parameter must be sent to the function. This is private information and must not be given to the user.',
            'parameters': {
                'email': {
                    'description': 'Email address',
                    'required': False,
                    'type': 'string'
                },
                'name': {
                    'description': 'First and last name',
                    'required': False,
                    'type': 'string'
                },
                'phone': {
                    'description': 'Phone number',
                    'required': False,
                    'type': 'string'
                },
            }
        },
        {
            'name': 'sendToSupport',
            'description': 'Send a message to the support team, used for service escalation.',
            'parameters': {
                'custId': {
                    'description': 'customer ID',
                    'required': True,
                    'type': 'string'
                },
                'purchaseId': {
                    'description': 'the ID of the purchase, can be found using purchaseSearch',
                    'required': True,
                    'type': 'string'
                },
                'supportSummary': {
                    'description': 'Summary of the support request',
                    'required': True,
                    'type': 'string'
                },
            }
        },
        {
            'name': 'purchaseSearch',
            'description': 'Search for, and get details of a purchases made. Details can be used for raising support requests. You can confirm you have this data, for example "I found your purchase" or "I can\'t find your purchase", but other details are private information and must not be given to the user.',
            'parameters': {
                'custId': {
                    'description': 'customer ID',
                    'required': True,
                    'type': 'string'
                },
                'productDescription': {
                    'description': 'a description of the purchased product to search for',
                    'required': True,
                    'type': 'string'
                },
                'purchaseDate': {
                    'description': 'date of purchase to start search from, in YYYY-MM-DD format',
                    'required': True,
                    'type': 'string'
                },
            }
        }
    ]

if __name__ == "__main__":
    lesson = sys.argv[1] if len(sys.argv) > 1 else None
    setup_bedrock_agent()

