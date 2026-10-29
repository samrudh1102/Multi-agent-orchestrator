import json

# A standard Python dictionary
user_profile = {
    "name": "Alice",
    "is_member": True,
    "hobbies": ["reading", "cycling"],
    "score": 9.5
}

# Convert the dictionary into a JSON string
json_string = json.dumps(user_profile)

print(json_string)
# Output: {"name": "Alice", "is_member": true, "hobbies": ["reading", "cycling"], "score": 9.5}

print(type(json_string))  # Output: <class 'str'>
