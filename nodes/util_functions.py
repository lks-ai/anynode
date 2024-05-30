import os
import hashlib
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class FunctionRegistry:
    def __init__(self, registry_dir="output/anynode"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "function_registry.json")
        self.registry = self.load_registry()
        self.chroma_client = self.init_chromadb()
        
    def init_chromadb(self):
        settings = Settings(chroma_dir=os.path.join(self.registry_dir, "chroma_db"))
        client = chromadb.Client(settings)
        return client

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=4)

    def hash_prompt(self, prompt):
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def add_function(self, prompt, function_code):
        prompt_hash = self.hash_prompt(prompt)
        self.registry[prompt_hash] = function_code
        self.save_registry()

    def get_function(self, prompt):
        prompt_hash = self.hash_prompt(prompt)
        return self.registry.get(prompt_hash, None)

    def query_chromadb(self, prompt, top_k=1):
        collection = self.chroma_client.get_or_create_collection(name="function_registry")
        results = collection.query(query_texts=[prompt], top_k=top_k)
        if results['documents']:
            return results['documents'][0]['content']
        return None

    def add_function_to_chromadb(self, prompt, function_code):
        collection = self.chroma_client.get_or_create_collection(name="function_registry")
        document = {
            "content": function_code,
            "metadata": {"prompt": prompt}
        }
        collection.add_documents([document])

    def get_function_with_rag(self, prompt):
        function_code = self.get_function(prompt)
        if function_code is None:
            function_code = self.query_chromadb(prompt)
        return function_code

    def add_function_to_registry(self, prompt, function_code):
        self.add_function(prompt, function_code)
        self.add_function_to_chromadb(prompt, function_code)

if __name__ == "__main__":
    # Example Usage
    registry = FunctionRegistry()

    # Adding a function to the registry
    prompt = "Generate a function that multiplies the input by 5."
    function_code = """
    def generated_function(input_data):
        return input_data * 5
    """
    registry.add_function_to_registry(prompt, function_code)

    # Retrieving a function from the registry
    retrieved_function = registry.get_function_with_rag(prompt)
    print("Retrieved Function:\n", retrieved_function)
