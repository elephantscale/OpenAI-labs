import boto3
import time

# Initialize the Lambda and IAM clients
lambda_client = boto3.client('lambda', region_name='us-west-2')
iam_client = boto3.client('iam') 

# List all Lambda functions
paginator = lambda_client.get_paginator('list_functions')
for page in paginator.paginate():
    for function in page['Functions']:
        if function['FunctionName'].startswith('dlai-support-agent-'):
            function_name = function['FunctionName']
            print(f"Deleting Lambda function: {function_name}")
            
            try:
                # Get the role associated with the function
                role_arn = function['Role']
                role_name = role_arn.split('/')[-1]
                
                # Delete the Lambda function
                lambda_client.delete_function(FunctionName=function_name)
                print(f"Lambda function {function_name} deleted successfully.")
                
                # # Delete the associated IAM role
                # print(f"Deleting IAM role: {role_name}")
                
                # # First, detach all policies from the role
                # attached_policies = iam_client.list_attached_role_policies(RoleName=role_name)
                # for policy in attached_policies.get('AttachedPolicies', []):
                #     iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
                
                # # Delete inline policies
                # inline_policies = iam_client.list_role_policies(RoleName=role_name)
                # for policy_name in inline_policies.get('PolicyNames', []):
                #     iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
                
                # # Delete the role
                # iam_client.delete_role(RoleName=role_name)
                # print(f"IAM role {role_name} deleted successfully.")
                
            except Exception as e:
                print(f"Error deleting function {function_name} or its role: {str(e)}")
            
            # Wait a bit to avoid hitting API rate limits
            time.sleep(1)

print("Lambda reset process completed.")
