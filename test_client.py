import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def run_test():
    student_records = []
    
    print("--- 1. Registering Individuals (Multi-Angle) ---")
    base_dir = "individuals"
    
    # Loop through each person's folder
    for student_id in os.listdir(base_dir):
        student_folder = os.path.join(base_dir, student_id)
        
        if os.path.isdir(student_folder):
            for file_name in os.listdir(student_folder):
                # Only process image files
                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    file_path = os.path.join(student_folder, file_name)
                    
                    # Send photo to the /register endpoint
                    with open(file_path, "rb") as f:
                        resp = requests.post(f"{BASE_URL}/register", files={"file": f})
                        
                    if resp.status_code == 200:
                        embedding = resp.json()["embedding"]
                        # Save the embedding attached to the folder name (student_id)
                        student_records.append({"student_id": student_id, "embedding": embedding})
                        print(f"✅ Registered: {student_id.capitalize()} (Angle: {file_name})")
                    else:
                        print(f"❌ Failed for {student_id} - {file_name}: {resp.text}")

    print(f"\nTotal embeddings loaded into temporary DB: {len(student_records)}")

    print("\n--- 2. Testing Group Photo ---")
    group_photo_path = "group5.jpg"
    
    # Send the group photo AND all registered arrays to the AI
    with open(group_photo_path, "rb") as f:
        data = {"students_json": json.dumps(student_records)}
        resp = requests.post(f"{BASE_URL}/process-attendance", files={"file": f}, data=data)
        
    if resp.status_code == 200:
        result = resp.json()
        total_faces = result['total_faces_detected']
        matched = result['matched_student_ids']
        
        print(f"\n🔍 Total Faces Found in Room: {total_faces}")
        print(f"👤 Known Students Matched: {len(matched)}")
        print(f"❓ Unknown Faces Ignored: {total_faces - len(matched)}")
        
        print("\n🎯 Present Roster:")
        for m in matched:
            print(f"   - {m.capitalize()}")
    else:
        print(f"Detection failed: {resp.text}")

if __name__ == "__main__":
    run_test()