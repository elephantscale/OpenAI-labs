import boto3

def delete_support_guardrails():
    # Create a Bedrock client
    bedrock = boto3.client('bedrock', region_name='us-west-2')

    # List all guardrails
    paginator = bedrock.get_paginator('list_guardrails')
    
    for page in paginator.paginate():
        for guardrail in page['guardrails']:
            if guardrail['name'].startswith('support-guardrails'):
                try:
                    # Delete the guardrail
                    bedrock.delete_guardrail(
                        guardrailIdentifier=guardrail['id'],
                        # guardrailVersion=guardrail['version']
                    )
                    print(f"Deleted guardrail: {guardrail['name']} (ID: {guardrail['id']})")
                except Exception as e:
                    print(f"Error deleting guardrail {guardrail['name']} (ID: {guardrail['id']}): {str(e)}")

if __name__ == "__main__":
    delete_support_guardrails()
    print("Guardrail reset process completed.")

