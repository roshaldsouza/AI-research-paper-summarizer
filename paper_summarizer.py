import requests
import PyPDF2
import sys
import re

class PDFChatbot:
    def __init__(self, pdf_path, model='llama3'):
        self.model = model
        self.pdf_path = pdf_path
        self.chunks = []
        self.full_text = ""
        
        print("\n" + "="*60)
        print("📚 Loading your PDF...")
        self.load_pdf()
        print(f"✅ Loaded {len(self.full_text)} characters")
        print(f"✅ Created {len(self.chunks)} chunks")
        print("="*60)
    
    def load_pdf(self):
        """Extract and chunk the PDF - memory efficient"""
        # Extract text with limit
        text = ""
        max_chars = 50000  # Limit to prevent memory issues
        
        with open(self.pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            total_pages = len(pdf.pages)
            print(f"   PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                if len(text) > max_chars:
                    print(f"   (Loaded first {i} pages to save memory)")
                    break
                text += page.extract_text() + "\n"
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        self.full_text = text[:max_chars]  # Truncate if needed
        
        # Create chunks
        chunk_size = 600
        overlap = 50
        chunks_created = 0
        
        for start in range(0, len(self.full_text), chunk_size - overlap):
            end = min(start + chunk_size, len(self.full_text))
            chunk = self.full_text[start:end]
            
            # Break at sentence
            if end < len(self.full_text):
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:
                    chunk = chunk[:last_period + 1]
            
            if chunk.strip() and len(chunk) > 50:
                self.chunks.append(chunk.strip())
                chunks_created += 1
                
                # Safety limit
                if chunks_created > 100:
                    break
    
    def find_relevant_chunks(self, question, top_k=3):
        """Find chunks most relevant to the question"""
        question_lower = question.lower()
        question_words = set(w for w in question_lower.split() if len(w) > 3)
        
        scored = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            
            # Count matching words
            score = 0
            for word in question_words:
                if word in chunk_lower:
                    score += 1
            
            # Bonus for question words appearing together
            if any(word in chunk_lower for word in question_words):
                score += 0.5
            
            scored.append((score, i, chunk))
        
        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Return top chunks
        return [chunk for _, _, chunk in scored[:top_k]]
    
    def ask_ollama(self, question, context):
        """Ask Ollama with context"""
        prompt = f"""Answer this question based on the research paper context below.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context provided
- Be specific and concise
- If not in context, say "I cannot find that information in the provided sections"

Answer:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"API Error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Is 'ollama serve' running?"
        except requests.exceptions.Timeout:
            return "❌ Request timed out. Try a simpler question."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def answer_question(self, question):
        """Main function to answer a question"""
        print("\n🔍 Searching relevant sections...")
        relevant = self.find_relevant_chunks(question, top_k=3)
        
        if not relevant:
            return "❌ Could not find relevant sections for this question."
        
        print(f"✓ Found {len(relevant)} relevant sections")
        print("🤖 Generating answer...")
        
        # Build context (keep it short)
        context_parts = []
        for i, chunk in enumerate(relevant):
            context_parts.append(f"[Excerpt {i+1}]\n{chunk[:400]}")  # Limit chunk size
        
        context = "\n\n".join(context_parts)
        
        # Get answer
        answer = self.ask_ollama(question, context)
        return answer
    
    def chat(self):
        """Interactive chat loop"""
        print("\n💬 Ask questions about your PDF! (Type 'quit' to exit)")
        print("\n📝 Example questions:")
        print("  • What is the main research question?")
        print("  • What methodology was used?")
        print("  • What were the key findings?")
        print("  • summary (for full summary)")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Special command for summary
                if question.lower() == 'summary':
                    question = "Provide a summary of this paper: main question, methodology, findings, and conclusions."
                
                # Get answer
                answer = self.answer_question(question)
                
                print("\n" + "-"*60)
                print("💡 Answer:")
                print("-"*60)
                print(answer)
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    print("\n🎓 Interactive PDF Q&A with RAG")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print("  python pdf_chat.py your_paper.pdf")
        print("  python pdf_chat.py your_paper.pdf llama3")
        print("\n⚠️  Make sure 'ollama serve' is running!\n")
        return
    
    pdf_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else 'llama3'
    
    # Check if file exists
    import os
    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_path}")
        return
    
    try:
        # Initialize chatbot
        print(f"🤖 Using model: {model}")
        chatbot = PDFChatbot(pdf_path, model)
        
        # Start interactive chat
        chatbot.chat()
        
    except MemoryError:
        print("\n❌ PDF is too large! Try a smaller file.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()