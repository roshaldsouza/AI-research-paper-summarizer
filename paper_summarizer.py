import requests
import PyPDF2
from typing import List
import os
import sys
import re

class PaperSummarizer:
    def __init__(self, model_name='llama3', ollama_url='http://localhost:11434'):
        """Initialize the RAG-based paper summarizer"""
        self.model_name = model_name
        self.ollama_url = ollama_url
        print(f"✅ Summarizer ready! Using model: {model_name}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for period, question mark, or exclamation
                last_sentence = max(
                    chunk.rfind('. '),
                    chunk.rfind('? '),
                    chunk.rfind('! ')
                )
                if last_sentence > chunk_size * 0.4:
                    end = start + last_sentence + 2
                    chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def retrieve_relevant_chunks(self, chunks: List[str], top_k: int = 4) -> List[str]:
        """Retrieve most relevant chunks using keyword matching"""
        # Keywords to look for in research papers
        keywords = [
            'abstract', 'introduction', 'method', 'approach', 'result', 
            'finding', 'conclusion', 'contribution', 'propose', 'show',
            'demonstrate', 'experiment', 'evaluation', 'performance',
            'research', 'study', 'analysis', 'data', 'model'
        ]
        
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count keyword matches
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            # Prefer chunks with more content
            score += len(chunk) / 1000
            scored_chunks.append((score, chunk))
        
        # Sort by score and take top-k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation"""
        try:
            print("   Calling Ollama API...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            raise Exception("❌ Cannot connect to Ollama. Make sure 'ollama serve' is running!")
        except requests.exceptions.Timeout:
            raise Exception("❌ Request timed out. Try a smaller document or different model.")
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def summarize_paper(self, file_path: str) -> dict:
        """Main function to summarize a research paper using RAG"""
        print(f"\n{'='*70}")
        print(f"📄 Processing: {os.path.basename(file_path)}")
        print(f"{'='*70}\n")
        
        # Step 1: Extract text
        print("📄 Extracting text from PDF...")
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        print(f"   ✓ Extracted {len(text)} characters\n")
        
        if len(text) < 100:
            raise Exception("Not enough text extracted. The PDF might be image-based or corrupted.")
        
        # Step 2: Chunk the document
        print("✂️  Chunking document...")
        chunks = self.chunk_text(text)
        print(f"   ✓ Created {len(chunks)} chunks\n")
        
        # Step 3: Retrieve relevant chunks (RAG)
        print("🔍 Retrieving relevant sections (RAG)...")
        relevant_chunks = self.retrieve_relevant_chunks(chunks, top_k=4)
        print(f"   ✓ Selected {len(relevant_chunks)} most relevant chunks\n")
        
        # Step 4: Generate summary
        print("🤖 Generating summary with Ollama (this may take 30-60 seconds)...")
        
        # Build context from relevant chunks
        context = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        prompt = f"""You are an expert at summarizing research papers. Based on the following excerpts from a research paper, provide a comprehensive summary.

Your summary should include:
1. The main research problem or question
2. The methodology or approach used
3. Key findings and results
4. Main conclusions and contributions

Research paper excerpts:
{context}

Write a clear, well-structured summary in 3-4 paragraphs. Be specific and focus on the most important information."""

        summary = self.call_ollama(prompt)
        
        print("   ✓ Summary complete!\n")
        
        return {
            'file': file_path,
            'chunks_total': len(chunks),
            'chunks_used': len(relevant_chunks),
            'summary': summary,
            'text_length': len(text)
        }


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🎓 AI Research Paper Summarizer with RAG")
    print("="*70)
    
    # Parse command line arguments
    model_name = 'llama3'  # default
    file_path = None
    
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        model_name = sys.argv[2]
    elif len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("\nHow to use:")
        print("  python paper_summarizer.py your_paper.pdf")
        print("  python paper_summarizer.py your_paper.pdf llama3")
        print("\nOr enter path below:")
        file_path = input("\n📁 File path: ").strip()
        file_path = file_path.strip('"').strip("'")
    
    if not file_path:
        print("❌ No file path provided!")
        return
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    try:
        # Initialize summarizer
        summarizer = PaperSummarizer(model_name=model_name)
        
        # Generate summary
        result = summarizer.summarize_paper(file_path)
        
        # Display results
        print("="*70)
        print("📋 SUMMARY")
        print("="*70 + "\n")
        print(result['summary'])
        print("\n" + "="*70)
        print(f"📊 Statistics:")
        print(f"   • Document size: {result['text_length']:,} characters")
        print(f"   • Total chunks: {result['chunks_total']}")
        print(f"   • Chunks analyzed: {result['chunks_used']}")
        print("="*70 + "\n")
        
        # Save to file
        save = input("💾 Save summary to file? (y/n): ").strip().lower()
        if save == 'y':
            output_file = os.path.splitext(file_path)[0] + "_summary.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"SUMMARY: {os.path.basename(file_path)}\n")
                f.write("="*70 + "\n\n")
                f.write(result['summary'])
                f.write("\n\n" + "="*70 + "\n")
                f.write(f"Generated by AI Research Paper Summarizer\n")
            print(f"✅ Summary saved to: {output_file}\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        print("Debug info:")
        traceback.print_exc()


if __name__ == "__main__":
    main()