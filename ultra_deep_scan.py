
import csv
import subprocess
import os

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def main():
    # Focused on the two previously "empty" ones + a few others to be sure
    targets = [
        "6f560c761b2fc2f577682d0cfda62630f37a3bb0",
        "bb3a3b6675b1844a13ebe368ad693f3dc75b315b"
    ]
    repo_dir = "sglang"
    
    for commit in targets:
        print(f"\n--- Investigating Commit {commit[:8]} ---")
        # 1. List ALL files, including hidden ones
        all_files = run_command(["git", "ls-tree", "-r", "--name-only", commit], cwd=repo_dir)
        if not all_files:
            print("Could not list files (possibly invalid commit or empty repo state)")
            continue
            
        file_list = all_files.splitlines()
        
        # 2. Filter for anything Docker-y (case insensitive)
        dockery = [f for f in file_list if "docker" in f.lower()]
        if dockery:
            print(f"Found Docker-related files: {', '.join(dockery)}")
        else:
            print("No files with 'docker' in the name.")
            
        # 3. Look for files that might be Dockerfiles but named differently (e.g. .devcontainer/something)
        # Check files in .devcontainer or .github
        hidden_stuff = [f for f in file_list if f.startswith(".")]
        if hidden_stuff:
            print(f"Hidden files/folders present: {len(hidden_stuff)}")
            for h in hidden_stuff[:10]: # show some
                if "config" in h or "setup" in h or "build" in h:
                    print(f"  Potential build config: {h}")
        
        # 4. Grep for "FROM " inside ALL files to find unnamed Dockerfiles
        # This is heavy, so we only do it if we found nothing
        if not dockery:
            print("Searching file contents for 'FROM '...")
            # We can't easily grep inside git objects without checking them out or using git grep
            # git grep "FROM " <commit>
            from_grep = run_command(["git", "grep", "-l", "^FROM ", commit], cwd=repo_dir)
            if from_grep:
                potential = [line.split(":", 1)[1] for line in from_grep.splitlines() if ":" in line]
                print(f"Found files containing '^FROM ': {', '.join(potential)}")
            else:
                print("No files starting with 'FROM ' found.")

if __name__ == "__main__":
    main()
