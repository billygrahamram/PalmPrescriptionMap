import os
import shutil
import platform
import subprocess
import time
from pathlib import Path 
import sys

import getpass


USE_CLEAN_OUTPUT = False


def get_mounted_drives():
    drives = []
    result = subprocess.run(['lsblk', '-o', 'MOUNTPOINT'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    for line in output.splitlines():
        if line.startswith('/media') or line.startswith('/run/media'):
            drives.append(line.strip())
    return drives

def select_drive(drives):
    print("\nDetected External Drives:")
    for idx, drive in enumerate(drives):
        print(f"{idx + 1}. {drive}")
    print("r. 🔄 Refresh the list of drives")
    print("q. ❌ Exit")
    
    while True:
        choice = input("Select a drive number, 'r' to refresh, or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("👋 Exiting. No drive selected.")
            exit()
        elif choice == 'r':
            print("⏳ Refreshing list of drives...")
            time.sleep(2)  # Wait for 2 seconds to allow any newly connected drives to show up
            drives = get_mounted_drives()  # Re-scan for drives
            if not drives:
                print("⚠️ No external drives found after refresh.")
            else:
                return select_drive(drives)  # Recursively refresh and reselect the drive
        elif choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(drives):
                return drives[choice - 1]
        print("Invalid choice. Try again.")


def navigate_directory(start_path):
    current_path = Path(start_path)
    while True:
        print(f"\nCurrent Directory: {current_path}")
        entries = [e for e in current_path.iterdir()]
        dirs = [e for e in entries if e.is_dir()]

        for idx, d in enumerate(dirs):
            print(f"{idx + 1}. [DIR] {d.name}")
        print("0. ✅ Select this folder")

        choice = input("Enter directory number to go deeper or 0 to select: ")
        if choice.isdigit():
            choice = int(choice)
            if choice == 0:
                return current_path
            elif 1 <= choice <= len(dirs):
                current_path = dirs[choice - 1]
            else:
                print("Invalid choice.")
        else:
            print("Please enter a number.")


def ask_output_preference():
    global USE_CLEAN_OUTPUT
    while True:
        choice = input("🧹 Do you want a clean terminal (no stacked output)? (y/n): ").strip().lower()
        if choice == 'y':
            USE_CLEAN_OUTPUT = True
            break
        elif choice == 'n':
            USE_CLEAN_OUTPUT = False
            break
        else:
            print("Please enter 'y' or 'n'.")

def print_status(message):
    if USE_CLEAN_OUTPUT:
        os.system('clear')  # or 'cls' for Windows
    print(message)



def get_next_project_dir(base_path):
    # Create an "output" folder where all projects will be saved
    output_folder = base_path / "output"
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the "output" folder exists

    # Now create the project folder inside the "output" directory
    count = 1
    while True:
        candidate = output_folder / f"project_{count}"
        if not candidate.exists():
            return candidate
        count += 1




def copy_images(src_folder, dest_folder):
    dest_images_path = dest_folder / "datasets" / "project" / "images"
    dest_images_path.mkdir(parents=True, exist_ok=True)

    jpg_images = list(src_folder.glob("*.JPG"))
    total = len(jpg_images)

    if total == 0:
        print("⚠️ No .JPG images found in the selected folder.")
        return

    print(f"\n📂 Copying {total} .JPG images to: {dest_images_path}\n")

    for i, img in enumerate(jpg_images, start=1):
        shutil.copy2(img, dest_images_path)
        sys.stdout.write(f"\r🟢 Progress: {i}/{total} images copied")
        sys.stdout.flush()

    print_status(f"\n✅ Finished copying {total} images.")



def wait_for_drive():
    print("👋 Welcome! Please insert your USB drive, memory card, or external HDD.")
    print("⏳ Waiting for device... Type 'q' to quit.\n")

    while True:
        drives = get_mounted_drives()
        if drives:
            return drives

        user_input = input("Press Enter to re-scan for drives, or type 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("👋 Exiting. No drive selected.")
            exit()

def main():
    if platform.system() != "Linux":
        print("⚠️ This script is intended for Linux systems.")
        return

    drives = wait_for_drive()
    drives = get_mounted_drives()
    selected_drive = select_drive(drives)
    folder_with_images = navigate_directory(selected_drive)


    project_root = Path(__file__).resolve().parent
    project_dir = get_next_project_dir(project_root)


    copy_images(folder_with_images, project_dir)
    print("Image copied to system!")

    print("Next steps...")


if __name__ == "__main__":
    main()

