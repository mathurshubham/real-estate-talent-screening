import kagglehub
import json
import os

def fetch_data():
    try:
        # Load the latest version
        path = kagglehub.dataset_download("aryan208/hr-interview-questions-and-ideal-answers")
        
        # Find the json file in the downloaded path
        json_file = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("hr_interview_questions_dataset.json"):
                    json_file = os.path.join(root, file)
                    break
        
        if not json_file:
            print("No JSON file found in dataset.")
            # Search for any json file just in case
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".json"):
                        json_file = os.path.join(root, file)
                        break
        
        if not json_file:
            print("No JSON file found in dataset.")
            return

        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # The data structure might be a list of dicts or a dict with a key
        # Based on usual Kaggle JSONs, it's often a list or has a key 'data'
        questions = []
        if isinstance(data, list):
            for item in data:
                # Look for 'question' key
                for k, v in item.items():
                    if 'question' in k.lower():
                        questions.append(v)
                        break
        elif isinstance(data, dict):
            # Check for keys like 'rows', 'data', or just the values
            for k, v in data.items():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            for ik, iv in item.items():
                                if 'question' in ik.lower():
                                    questions.append(iv)
                                    break
        
        if not questions:
            print(f"Could not find questions in JSON. Data sample: {str(data)[:200]}")
            return

        top_questions = questions[:50]
        
        # Map to our structure
        pillars = ['Skill', 'Training', 'Attitude', 'Results']
        mapped_questions = []
        
        for i, q in enumerate(top_questions):
            pillar = pillars[i % len(pillars)]
            mapped_questions.append({
                "id": f"K{i+1}",
                "text": q,
                "type": "rating" if pillar == 'Skill' else "mcq",
                "pillar": pillar,
                "category": "Kaggle HR",
                "options": [
                    {"label": "Exceeds Expectations", "value": 5},
                    {"label": "Standard Professional", "value": 4},
                    {"label": "Developing", "value": 3},
                    {"label": "Needs Improvement", "value": 2},
                    {"label": "Poor", "value": 1}
                ] if pillar != 'Skill' else None
            })

        output_path = "src/data/kaggleQuestions.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(mapped_questions, f, indent=2)
            
        print(f"Successfully saved {len(mapped_questions)} questions to {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_data()
