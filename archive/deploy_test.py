import os, json, glob

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath("main_gen139.py")) # Assuming running from root
GIT_DIR = os.path.join(BASE_DIR, "hosting")

print("🔍 INSPECTING FILES VS MANIFEST...")

# Find all manifests
manifests = glob.glob(os.path.join(GIT_DIR, "*", "manifest.json"))
if not manifests:
    print("❌ CRITICAL: No manifest.json found in hosting/ folder.")
else:
    for m in manifests:
        folder = os.path.dirname(m)
        folder_name = os.path.basename(folder)
        print(f"\n📂 Checking Day: {folder_name}")
        
        with open(m) as f:
            data = json.load(f)
            
        for ep in data.get("episodes", []):
            filename = ep['filename']
            file_path = os.path.join(folder, filename)
            
            if os.path.exists(file_path):
                print(f"   ✅ OK: {filename}")
            else:
                print(f"   ❌ MISSING: Manifest expects '{filename}' but it is NOT in the folder.")
                print(f"      (This causes the 'No Podcast Found' error!)")

print("\n------------------------------------------------")
print("👉 IF YOU SAW RED 'MISSING' ERRORS ABOVE:")
print("   Run 'python3 main_gen201.py' again to regenerate the manifest,")
print("   then run 'python3 deploy_github.py' to push the fix.")