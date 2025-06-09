#!/usr/bin/env python3
import sys
import os
import pwd
import getpass
import argparse
import builtins


# Define user
sudo_user = os.environ.get("SUDO_USER")
doas_user = os.environ.get("DOAS_USER")
pkexec_uid = os.environ.get("PKEXEC_UID")
pkexec_user = pwd.getpwuid(int(pkexec_uid))[0] if pkexec_uid else ""
env_user = getpass.getuser()
user = next((u for u in [sudo_user, doas_user, pkexec_user, env_user] if u), "")

if not user:
    print("Could not determine user, please use the --user flag")
    sys.exit(1)

# Args parser
parser = argparse.ArgumentParser(
    description="Command line interface for BM Auth biometric authentication",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prog="bm_auth",
    usage="bm_auth [-U USER] [--plain] [-y] {face,voice} {add,remove,list} ...",
    epilog="For support please visit\nhttps://github.com/vareee/bm_auth",
    add_help=False
)

# Command: face / voice
parser.add_argument(
    "command",
    help="Biometric method to use",
    metavar="command",
    choices=["face", "voice"]
)

# Subcommand: add / remove / list
parser.add_argument(
    "subcommand",
    help="Action to perform",
    metavar="action",
    choices=["add", "remove", "list"]
)

# Optional args for add/remove
parser.add_argument(
    "arguments",
    help="Optional arguments for some commands",
    nargs="*"
)

# Flags
parser.add_argument(
    "-U", "--user",
    default=user,
    help="Set the user account to use"
)


parser.add_argument(
    "-h", "--help",
    action="help",
    default=argparse.SUPPRESS,
    help="Show this help message and exit"
)

# Print help if no args
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(0)

# Parse args
args = parser.parse_args()

# Add to builtins to use in other modules
builtins.bm_args = args
builtins.bm_user = args.user

# If root
if os.geteuid() != 0:
    print("Please run this command as root:")
    print(f"\tsudo bm_auth {' '.join(sys.argv[1:])}")
    sys.exit(1)

# root prohibited
if args.user == "root":
    print("Can't run commands as root, please specify a normal user with --user")
    sys.exit(1)

# Get path to module
base_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(base_dir, f"{args.command}_auth")

script_map = {
    ("face", "add"): "ref_face.py",
    ("face", "remove"): "del_face.py",
    ("face", "list"): "list_face.py",
    ("voice", "add"): "ref_voice.py",
    ("voice", "remove"): "del_voice.py",
}

# If command combination exists
if (args.command, args.subcommand) not in script_map:
    print(f"Unknown command combination: {args.command} {args.subcommand}")
    sys.exit(1)

script_name = script_map[(args.command, args.subcommand)]
script_path = os.path.join(module_dir, script_name)

# If file exists
if not os.path.isfile(script_path):
    print(f"Target script does not exist: {script_path}")
    sys.exit(1)

# Add module dir to sys.path for import
sys.path.insert(0, module_dir)

# Get module name without extension
module_name = os.path.splitext(script_name)[0]

# Import as module
try:
    module = __import__(module_name)
except Exception as e:
    print(f"Failed to import {module_name}: {e}")
    sys.exit(1)

# Invoke main() if exists
if hasattr(module, "main"):
    try:
        module.main()
    except Exception as e:
        print(f"Error while running {module_name}.main(): {e}")
        sys.exit(1)
else:
    sys.exit(1)

