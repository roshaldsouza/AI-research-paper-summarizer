# test_summarizer.py - Quick test without PDF

import requests

def test_ollama():
    """Test if Ollama is working"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Summarize the theory of relativity in 2 sentences.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()['response']
            print("✅ Ollama is working!\n")
            print("Test response:")
            print(result)
        else:
            print(f"❌ Error: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama!")
        print("Make sure you've run: ollama serve")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Testing Ollama connection...\n")
    test_ollama()