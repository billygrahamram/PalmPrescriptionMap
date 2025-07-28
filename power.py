import subprocess
import sys
import getpass

def run_command(cmd):
    try:
        subprocess.run(["sudo"] + cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def main():
    print("⚡ Enabling Jetson ORIN AGX Max Performance Mode...")
    run_command(["nvpmodel", "-m", "0"])
    run_command(["jetson_clocks"])
    print("✅ Jetson performance mode enabled.")

if __name__ == "__main__":
    # Check if running as root
    if getpass.getuser() != "root":
        print("🔒 This script must be run with sudo or root privileges.")
        sys.exit(1)
    main()
