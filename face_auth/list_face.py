# List all models for a user
import sys
import os
import json
import time
import builtins


user = builtins.bm_user

# Check if the models file has been created yet
if not os.path.exists("/usr/local/etc/bm_auth/face_auth/models"):
        print("Face models have not been initialized yet, please run:")
        print("\n\tsudo bm_auth -U " + user + " add\n")
        sys.exit(1)

# Path to the models file
enc_file = f"/usr/local/etc/bm_auth/face_auth/models/{user}.dat"

# Try to load the models file and abort if the user does not have it yet
try:
        encodings = json.load(open(enc_file))
except FileNotFoundError:
        print("No face model known for the user {}, please run:".format(user))
        print("\n\tsudo bm_auth -U " + user + " add\n")
        sys.exit(1)

# Print a header
print("Known face models for {}:".format(user))
print("\n\033[1;29m" + "ID  Date                 Label\033[0m")

# Loop through all encodings and print info about them
for enc in encodings:
        # Start with the id
        print(str(enc["id"]), end="")

        print((4 - len(str(enc["id"]))) * " ", end="")

        # Format the time as ISO in the local timezone
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(enc["time"])), end="")

        # Separate with spaces
        print("  ", end="")
    
    # End with the label
        print(enc["label"])

# Add a closing enter
print()
