# Remove a encoding from the models file
import sys
import os
import json
import builtins


user = builtins.bm_user

# If enough arguments have been passed
if not builtins.bm_args.arguments:
        print("Please add the ID of the model you want to remove as an argument")
        print("For example:")
        print("\n\tbm_auth face remove 0\n")
        print("You can find the IDs by running:")
        print("\n\tbm_auth face list\n")
        sys.exit(1)

# If the models file has been created yet
if not os.path.exists("/usr/local/etc/bm_auth/face_auth/models"):
        print("Face models have not been initialized yet, please run:")
        print("\n\tbm_auth add\n")
        sys.exit(1)

# Path to the models file
enc_file = f"/usr/local/etc/bm_auth/face_auth/models/{user}.dat"

# Try to load the models file and abort if the user does not have it yet
try:
        encodings = json.load(open(enc_file))
except FileNotFoundError:
        print("No face model known for the user {}, please run:".format(user))
        print("\n\tbm_auth add\n")
        sys.exit(1)

# If a encoding with that id has been found
found = False

# Get the ID from the cli arguments
id = builtins.bm_args.arguments[0]

# Loop though all encodings and if they match the argument
for enc in encodings:
        if str(enc["id"]) == id:
                # Double check with the user
                print('This will remove the model called "{label}" for {user}'.format(label=enc["label"], user=user))
                ans = input("Do you want to continue [y/N]: ")

                # Abort if the answer isn't yes
                if (ans.lower() != "y"):
                        print('\nInterpreting as a "NO", aborting')
                        sys.exit(1)

                # Add a padding empty  line
                print()

                # Mark as found and print an enter
                found = True
                break

# Abort if no matching id was found
if not found:
        print("No model with ID {id} exists for {user}".format(id=id, user=user))
        sys.exit(1)

# Remove the entire file if this encoding is the only one
if len(encodings) == 1:
        os.remove(f"/usr/local/etc/bm_auth/face_auth/models/{user}.dat")
        print("Removed last model, face_auth disabled for user")
else:
        # A place holder to contain the encodings that will remain
        new_encodings = []

        # Loop though all encodings and only add those that don't need to be removed
        for enc in encodings:
                if str(enc["id"]) != id:
                        new_encodings.append(enc)

        # Save this new set to disk
        with open(enc_file, "w") as datafile:
                json.dump(new_encodings, datafile)

        print("Removed model {}".format(id))
