"""generator.py : Handles answer generation using a RAG approach."""

import os
from retriever import load_vector_store, get_relevant_documents  # Assumes retriever.py is in the same directory or PYTHONPATH

VECTOR_STORE_PATH = r"C:\Users\dell\Downloads\constitution_qa_project\vector_store\faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Must match the model used in retriever.py
LLM_MODEL_NAME = "google/flan-t5-small"  # Lightweight, instruction-tuned model

def initialize_llm_pipeline(model_name: str):
    """Initializes a Hugging Face pipeline for text generation."""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        print(f"Initializing tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        generator_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        print("LLM pipeline successfully initialized.")
        return generator_pipeline
    except ImportError as ie:
        print(f"ImportError: {ie}. Installing dependencies...")
        os.system("pip3 install transformers torch")
        print("Please re-run the script after installation.")
        return None
    except Exception as e:
        print(f"An error occurred while initializing the LLM pipeline: {e}")
        return None

def format_prompt(query: str, retrieved_docs: list) -> str:
    """Formats the prompt for the LLM using the query and retrieved documents."""
    context_str = "\n\n".join([
        f"Source Article: {doc.metadata.get('article_number', 'N/A')} (From Title: {doc.metadata.get('title', 'N/A')})\nContent: {doc.page_content}"
        for doc in retrieved_docs
    ])

    prompt = (
        "Based *only* on the following articles from the Moroccan Constitution, please answer the question. "
        "Cite the specific article numbers that support your answer. "
        "If the provided articles do not contain the answer, state that the information is not found in the provided context.\n\n"
        "Context from the Moroccan Constitution:\n"
        "-------------------------------------\n"
        f"{context_str}\n"
        "-------------------------------------\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(query: str, vector_store, llm_pipeline, k_retrieval: int = 3) -> str:
    """Generates an answer to the query using the RAG approach."""
    if not vector_store or not llm_pipeline:
        return "Error: Vector store or LLM pipeline not initialized."

    retrieved_docs = get_relevant_documents(query, vector_store, k=k_retrieval)
    if not retrieved_docs:
        return "No relevant documents found for your query."

    prompt = format_prompt(query, retrieved_docs)
    print("\n--- Generated Prompt for LLM ---")
    print(prompt)
    print("--- End of Prompt ---\n")

    try:
        print("Generating answer with LLM...")
        generated_output = llm_pipeline(prompt, max_length=512, num_return_sequences=1, clean_up_tokenization_spaces=True)
        answer = generated_output[0]["generated_text"]
        print("Answer generated successfully.")
        return answer
    except Exception as e:
        print(f"An error occurred during answer generation: {e}")
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    print("Starting RAG generator script...")

    vs = load_vector_store(VECTOR_STORE_PATH, EMBEDDING_MODEL)
    llm_pipe = initialize_llm_pipeline(LLM_MODEL_NAME)

    if vs and llm_pipe:
        sample_query = "What does the constitution say about the King?"
        print(f"\nTesting RAG system with query:\n{sample_query}")

        final_answer = generate_answer(sample_query, vs, llm_pipe, k_retrieval=3)

        print("\n--- Final Answer ---")
        print(final_answer)
        print("--- End of Answer ---")
    else:
        print("Failed to initialize vector store or LLM. Aborting test.")

    print("RAG generator script finished.")