import json
import os

# Read the history from file
with open("outputs/cbam_default/history.json", "r") as f:
    history_list = json.load(f)

# Convert to dictionary with lists of metrics
history_dict = {}
for entry in history_list:
    for key, value in entry.items():
        if key not in history_dict:
            history_dict[key] = []
        history_dict[key].append(value)

# Save back to file
with open("outputs/cbam_default/history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

print("History file converted to correct format")
