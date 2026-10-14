"""
Manage CrowdGuard accounts from a terminal (works while the server is stopped).

  python api/manage_users.py list
  python api/manage_users.py add maggie --role admin
  python api/manage_users.py add guard1            (role defaults to viewer)
  python api/manage_users.py password admin        (reset a password)
  python api/manage_users.py delete guard1

Passwords are asked for without showing them on screen.
Restart the server after changing accounts here.
"""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auth import ROLES, AuthError, UserStore  # noqa: E402


def ask_password() -> str:
    p1 = getpass.getpass("New password: ")
    p2 = getpass.getpass("Repeat it: ")
    if p1 != p2:
        sys.exit("Passwords do not match")
    return p1


def main():
    ap = argparse.ArgumentParser(description="Manage CrowdGuard user accounts")
    ap.add_argument("--data-dir", default=os.environ.get(
        "CROWDGUARD_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--role", choices=ROLES, default="viewer")
    sub.add_parser("password").add_argument("username")
    sub.add_parser("delete").add_argument("username")
    args = ap.parse_args()

    store = UserStore(args.data_dir)
    try:
        if args.cmd == "list":
            for u in store.list_users():
                print(f"{u['username']:20s} {u['role']}")
            if not store.list_users():
                print("No accounts yet. The server creates 'admin' on first start.")
        elif args.cmd == "add":
            store.add_user(args.username, ask_password(), args.role)
            print(f"Created {args.username} ({args.role})")
        elif args.cmd == "password":
            store.set_password(args.username, ask_password())
            print(f"Password changed for {args.username}; their old logins stop working")
        elif args.cmd == "delete":
            store.delete_user(args.username)
            print(f"Deleted {args.username}")
    except AuthError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
