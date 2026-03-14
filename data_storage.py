import json
import matplotlib.pyplot as plt

class DataStorage():
    def __init__(self, filename):
        self.filename = filename

    def initialize_json(self):
        with open(self.filename, 'w') as file:
            json.dump({"total_steps": [], "success_rate": [], "success_rate2": [], "sep_rat": []}, file)

    # Append new data
    def append_to_json(self, total_steps, success_rate, success_rate2=None, sep_rat=None):
        try:
            # Open the file for reading
            with open(self.filename, 'r') as file:
                try:
                    data = json.load(file)  # Load existing data
                except json.JSONDecodeError:
                    print(f"Skipping append: Invalid JSON in {self.filename}")
                    return
        except FileNotFoundError:
            print(f"Skipping append: File {self.filename} not found")
            return

        try:
            # Append the new values to the appropriate lists
            data["total_steps"].append(total_steps)
            data["success_rate"].append(success_rate)
            if success_rate2 is not None:
                data["success_rate2"].append(success_rate2)
            if sep_rat is not None:
                data["sep_rat"].append(sep_rat)
        except (KeyError, AttributeError) as e:
            # Skip if structure is invalid
            print(f"Skipping append: Data structure issue in {self.filename} ({e})")
            return

        try:
            # Save the updated data
            with open(self.filename, 'w') as file:
                json.dump(data, file)
        except PermissionError as e:
            print(f"Skipping append: Permission error while saving to {self.filename} ({e})")

