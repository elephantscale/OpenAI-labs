import json, random

def unpack_parameters(parameters):
    result = {}
    for param in parameters:
        name = param.get("name")
        value = param.get("value")
        
        if name is not None:
            if value is not None:
                # Try to convert to int or float if possible
                try:
                    result[name] = int(value)
                except ValueError:
                    try:
                        result[name] = float(value)
                    except ValueError:
                        result[name] = value  # Keep as string if not a number
            else:
                result[name] = None  # Handle missing value
    
    return result

def lambda_handler(event, context):
    
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])

    unpacked = unpack_parameters(parameters)

    responseBody =  {
        "TEXT": {
            "body": ""
        }
    }

    if function == "customerId":
        email = unpacked.get("email", None)
        name = unpacked.get("name", None)
        phone = unpacked.get("phone", None)

        if any(value is not None and value != "" for value in (email, name, phone)):
            customer_id = random.randint(1000, 9999)
            responseBody['TEXT']['body'] = f"{{'id':{customer_id}}}"
        else:
            responseBody['TEXT']['body'] = "{'error':'Customer not found.'}"

    elif function == "sendToSupport":
        custId = unpacked.get("custId", None)
        supportSummary = unpacked.get("supportSummary", None)
        
        if all(value is not None and value != "" for value in (custId, supportSummary)):
            support_id = random.randint(1000, 9999)
            responseBody['TEXT']['body'] = f"{{'supportId': {support_id}}}"
        else:
            responseBody['TEXT']['body'] = "{'error':'Details missing.'}"

    else:
        responseBody['TEXT']['body'] = "{'error':'unknown function'}"


    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }

    }

    dummy_function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
    print("Response: {}".format(dummy_function_response))

    return dummy_function_response