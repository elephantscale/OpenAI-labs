# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

import boto3
import json
import random
import string
import time
import textwrap

region_name = 'us-west-2'

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_random_suffix(length: int = 5) -> str:
    """
    Generate a random suffix of specified length.

    Args:
        length (int): The length of the random suffix. Defaults to 5.

    Returns:
        str: A random string of uppercase letters and digits.
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def wait_for_agent_status(agentId: str, targetStatus: str):
    """
    Wait for a Bedrock agent to reach a specific status.

    Args:
        agentId (str): The ID of the Bedrock agent.
        targetStatus (str): The desired status to wait for.

    Returns:
        None
    """
    print(f"Waiting for agent status of '{targetStatus}'...")
    agent_status = ''
    bedrock_agent = boto3.client(service_name='bedrock-agent', region_name=region_name)
    while agent_status != targetStatus:
        response = bedrock_agent.get_agent(agentId=agentId)
        agent_status = response['agent']['agentStatus']
        print(f"Agent status: {agent_status}")
        time.sleep(2)
    print(f"Agent reached '{targetStatus}' status.")

def wait_for_agent_alias_status(agentId: str, agentAliasId: str, targetStatus: str):
    """
    Wait for a Bedrock agent alias to reach a specific status.

    Args:
        agentId (str): The ID of the Bedrock agent.
        agentAliasId (str): The ID of the agent alias.
        targetStatus (str): The desired status to wait for.

    Returns:
        None
    """
    print(f"Waiting for agent alias status of '{targetStatus}'...")
    bedrock_agent = boto3.client(service_name = 'bedrock-agent', region_name = region_name)
    while True:
        response = bedrock_agent.get_agent_alias(
            agentId=agentId,
            agentAliasId=agentAliasId
        )
        current_status = response['agentAlias']['agentAliasStatus']
        print(f"Agent alias status: {current_status}")
        if current_status == targetStatus:
            break
        time.sleep(2)
    print(f"Agent alias reached status '{targetStatus}'")


def invoke_agent_and_print(agentId: str, agentAliasId: str, inputText: str, sessionId: str, enableTrace: bool = False, endSession: bool = False, width: int = 70):
    
    bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name=region_name)
    
    response = bedrock_agent_runtime.invoke_agent(
        agentId=agentId,
        agentAliasId=agentAliasId,
        sessionId=sessionId,
        inputText=inputText,
        endSession=endSession,
        enableTrace=enableTrace
    )

    event_stream = response["completion"]
    agent_response = ""

    print(f"User: {textwrap.fill(inputText, width=width)}\n")
    print("Agent:", end=" ", flush=True)

    for event in event_stream:
        if 'chunk' in event:
            chunk_text = event['chunk'].get('bytes', b'').decode('utf-8')
            if not enableTrace:  # Only print chunks if trace is not enabled
                print(textwrap.fill(chunk_text, width=width, subsequent_indent='       '), end='', flush=True)
            agent_response += chunk_text
        elif 'trace' in event and enableTrace:
            trace = event['trace']
            
            if 'trace' in trace:
                trace_details = trace['trace']
                
                if 'orchestrationTrace' in trace_details:
                    orch_trace = trace_details['orchestrationTrace']
                    
                    if 'invocationInput' in orch_trace:
                        inv_input = orch_trace['invocationInput']
                        print("\nInvocation Input:")
                        print(f"  Type: {inv_input.get('invocationType', 'N/A')}")
                        if 'actionGroupInvocationInput' in inv_input:
                            agi = inv_input['actionGroupInvocationInput']
                            print(f"  Action Group: {agi.get('actionGroupName', 'N/A')}")
                            print(f"  Function: {agi.get('function', 'N/A')}")
                            print(f"  Parameters: {agi.get('parameters', 'N/A')}")
                    
                    if 'rationale' in orch_trace:
                        thought = orch_trace['rationale']['text']
                        print(f"\nAgent's thought process:")
                        print(textwrap.fill(thought, width=width, initial_indent='  ', subsequent_indent='  '))
                    
                    if 'observation' in orch_trace:
                        obs = orch_trace['observation']
                        print("\nObservation:")
                        print(f"  Type: {obs.get('type', 'N/A')}")
                        if 'actionGroupInvocationOutput' in obs:
                            print(f"  Action Group Output: {obs['actionGroupInvocationOutput'].get('text', 'N/A')}")
                        if 'knowledgeBaseLookupOutput' in obs:
                            print("  Knowledge Base Lookup:")
                            for ref in obs['knowledgeBaseLookupOutput'].get('retrievedReferences', []):
                                print(f"    - {ref['content'].get('text', 'N/A')[:50]}...")
                        if 'codeInterpreterInvocationOutput' in obs:
                            cio = obs['codeInterpreterInvocationOutput']
                            print("  Code Interpreter Output:")
                            print(f"    Execution Output: {cio.get('executionOutput', 'N/A')[:50]}...")
                            print(f"    Execution Error: {cio.get('executionError', 'N/A')}")
                            print(f"    Execution Timeout: {cio.get('executionTimeout', 'N/A')}")
                        if 'finalResponse' in obs:
                            final_response = obs['finalResponse']['text']
                            print(f"\nFinal response:")
                            print(textwrap.fill(final_response, width=width, initial_indent='  ', subsequent_indent='  '))
                
                if 'guardrailTrace' in trace_details:
                    guard_trace = trace_details['guardrailTrace']
                    print("\nGuardrail Trace:")
                    print(f"  Action: {guard_trace.get('action', 'N/A')}")
                    
                    for assessment in guard_trace.get('inputAssessments', []) + guard_trace.get('outputAssessments', []):
                        if 'contentPolicy' in assessment:
                            for filter in assessment['contentPolicy'].get('filters', []):
                                print(f"  Content Filter: {filter['type']} (Confidence: {filter['confidence']}, Action: {filter['action']})")
                        
                        if 'sensitiveInformationPolicy' in assessment:
                            for pii in assessment['sensitiveInformationPolicy'].get('piiEntities', []):
                                print(f"  PII Detected: {pii['type']} (Action: {pii['action']})")

    print(f"\n\nSession ID: {response.get('sessionId')}")
    # print(f"Memory ID: {response.get('memoryId')}")

    return

def wait_for_action_group_status(agentId: str, actionGroupId: str, targetStatus: str = 'ENABLED') -> str:
    """
    Wait for a Bedrock agent action group to reach a specific status.

    Args:
        agentId (str): The ID of the Bedrock agent.
        actionGroupId (str): The ID of the agent action group.
        targetStatus (str): The desired status to wait for. Defaults to 'ENABLED'.

    Returns:
        str: The current status of the action group.
    """
    bedrock_agent = boto3.client(service_name='bedrock-agent', region_name=region_name)
    action_group_status = ''
    while action_group_status != targetStatus:
        response = bedrock_agent.get_agent_action_group(
            agentId=agentId,
            actionGroupId=actionGroupId,
            agentVersion='DRAFT'
        )
        action_group_status = response['agentActionGroup']['actionGroupState']
        print(f"Action Group status: {action_group_status}")
        time.sleep(2)
    return action_group_status