
import csv
import subprocess
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def main():
    csv_path = "commit-status/missing_benchmarks.csv"
    repo_dir = "sglang"
    
    if not os.path.exists(repo_dir):
        print(f"Error: {repo_dir} directory not found.")
        return

    print(f"Scanning commits from {csv_path}...\n")
    
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            commit = row['commit_hash']
            # Search for any file with 'Dockerfile' in its name at this commit
            # git ls-tree -r --name-only <commit> | grep -i Dockerfile
            files = run_command(["git", "ls-tree", "-r", "--name-only", commit], cwd=repo_dir)
            
            dockerfiles = []
            if files:
                dockerfiles = [f for f in files.splitlines() if "Dockerfile" in f]
            
            results.append({
                "commit": commit,
                "subject": row['subject'],
                "found_dockerfiles": dockerfiles,
                "count": len(dockerfiles)
            })
            
            if dockerfiles:
                print(f"✅ Found {len(dockerfiles)} Dockerfiles in {commit[:8]}: {', '.join(dockerfiles)}")
            else:
                print(f"❌ No Dockerfiles in {commit[:8]}")

    # Write a new report
    output_path = "commit-status/missing_dockerfile_search.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["commit_hash", "dockerfile_count", "dockerfile_paths", "subject"])
        for res in results:
            writer.writerow([
                res['commit'],
                res['count'],
                ";".join(res['found_dockerfiles']),
                res['subject']
            ])
            
    print(f"\nReport written to {output_path}")

if __name__ == "__main__":
    main()
