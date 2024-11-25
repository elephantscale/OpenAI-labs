import sys
import boto3
import os
import uuid
import time
from helper import *

# print("*" * 50)
print("Lesson 5 Prep")
# print("*" * 50)

# Constants and environment variables
REGION_NAME = 'us-west-2'
AGENT_ID = os.environ['BEDROCK_AGENT_ID']
AGENT_ALIAS_ID = os.environ['BEDROCK_AGENT_ALIAS_ID']

# Initialize clients
bedrock = boto3.client('bedrock', region_name=REGION_NAME)
bedrock_agent = boto3.client('bedrock-agent', region_name=REGION_NAME)
s3_client = boto3.client('s3', region_name=REGION_NAME)

def create_guardrail():
    create_guardrail_response = bedrock.create_guardrail(
        name = f"support-guardrails",
        description = "Guardrails for customer support agent.",
        topicPolicyConfig={
            'topicsConfig': [
                {
                    "name": "Internal Customer Information",
                    "definition": "Information relating to this or other customers that is only available through internal systems.  Such as a customers ID number or support level. ",
                    "examples": [],
                    "type": "DENY"
                }
            ]
        },
        contentPolicyConfig={
            'filtersConfig': [
                {
                    "type": "SEXUAL",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "HATE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "VIOLENCE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "INSULTS",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "MISCONDUCT",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "HIGH",
                    "outputStrength": "NONE"
                }
            ]
        },
        contextualGroundingPolicyConfig={
            'filtersConfig': [
                {
                    "type": "GROUNDING",
                    "threshold": 0.7
                },
                {
                    "type": "RELEVANCE",
                    "threshold": 0.7
                }
            ]
        },
        blockedInputMessaging = "Sorry, the model cannot answer this question.",
        blockedOutputsMessaging = "Sorry, the model cannot answer this question."
    )
    
    guardrail_id = create_guardrail_response['guardrailId']
    # print(f"Guardrail created successfully. Guardrail ID: {guardrail_id}")
    
    create_guardrail_version_response = bedrock.create_guardrail_version(guardrailIdentifier=guardrail_id)
    guardrail_version = create_guardrail_version_response['version']
    # print(f"Guardrail version created successfully. Guardrail Version: {guardrail_version}")
    
    return guardrail_id, guardrail_version

def update_agent(guardrail_id, guardrail_version):
    agent_details = bedrock_agent.get_agent(agentId=AGENT_ID)
    
    bedrock_agent.update_agent(
        agentId=AGENT_ID,
        agentName=agent_details['agent']['agentName'],
        agentResourceRoleArn=agent_details['agent']['agentResourceRoleArn'],
        instruction=agent_details['agent']['instruction'],
        foundationModel=agent_details['agent']['foundationModel'],
        guardrailConfiguration={
            'guardrailIdentifier': guardrail_id,
            'guardrailVersion': guardrail_version
        },
    )
    
    bedrock_agent.prepare_agent(agentId=AGENT_ID)
    wait_for_agent_status(agentId=AGENT_ID, targetStatus='PREPARED')
    
    bedrock_agent.update_agent_alias(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        agentAliasName='MyAgentAlias',
    )
    wait_for_agent_alias_status(agentId=AGENT_ID, agentAliasId=AGENT_ALIAS_ID, targetStatus='PREPARED')
    # print("Agent alias updated successfully.")

def main():
    guardrail_id, guardrail_version = create_guardrail()
    update_agent(guardrail_id, guardrail_version)

if __name__ == "__main__":
    main()