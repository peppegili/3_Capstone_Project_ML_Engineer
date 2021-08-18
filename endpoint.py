import requests
import json

# URL for the web service
scoring_uri = 'http://c55aee43-1cff-46b2-b394-95a2e22cedbe.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'jPDSVdsJV9nUeE6Jafq3mcIMtc64hAXQ'

# A set of data to score, so we get one results back
data = {"data":
        [
            {
                'age': 50, 
                'anaemia': 1, 
                'creatinine_phosphokinase': 230,
                'diabetes': 0,
                'ejection_fraction': 38,
                'high_blood_pressure': 1,
                'platelets': 390000,
                'serum_creatinine': 1.8,
                'serum_sodium': 135,
                'sex': 1.0,
                'smoking': 0,
                'time': 14
            }
        ]
       }

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
