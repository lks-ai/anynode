""" 
Save and Recall functions across Comfy with Persistent functions
shouts to @risunobushi on discord for helping me crack this
"""
import os
import hashlib
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class FunctionRegistry:
    def __init__(self, registry_dir="output/anynode", schema="default", version="1.0"):
        print('PATH BASENAME', os.path.basename)
        self.registry_dir = os.path.abspath(os.path.join(os.getcwd(), registry_dir))
        print('REGISTRY DIR', self.registry_dir, flush=True)
        os.makedirs(self.registry_dir, exist_ok=True)
        self.schema = schema
        self.version = version
        self.registry_file = os.path.join(self.registry_dir, f"function_registry_{self.schema}.json")
        self.registry = self.load_registry()
        self.chroma_client = self.init_chromadb()

    def init_chromadb(self):
        folder = os.path.join(self.registry_dir, f"chroma_db_{self.schema}")
        os.makedirs(folder, exist_ok=True)
        fe = os.path.exists(folder)
        print(f"ChromaDB Path: {folder}, exists: {fe}")
        settings = Settings(persist_directory=folder)
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

    def add_function(self, prompt, function_code, imports, comment, input_types):
        prompt_hash = self.hash_prompt(prompt)
        self.registry[prompt_hash] = {
            "prompt": prompt,
            "imports": imports,
            "function": function_code,
            "comment": comment,
            "version": self.version
        }
        self.save_registry()
        self.add_function_to_chromadb(prompt_hash, prompt, function_code, imports, comment, input_types)
        return prompt_hash

    def get_function(self, prompt):
        prompt_hash = self.hash_prompt(prompt)
        function_data = self.registry.get(prompt_hash, None)
        if function_data:
            return (function_data, prompt_hash)
        return (None, None)

    def query_chromadb(self, prompt, input_types, top_k=1):
        collection = self.chroma_client.get_or_create_collection(name="function_registry")
        filters = {"input_types": input_types}
        results = collection.query(query_texts=[prompt], top_k=top_k, filter_metadata=filters)
        if results['documents']:
            return results['documents'][0]['content']
        return None

    def add_function_to_chromadb(self, prompt_hash, prompt, function_code, imports, comment, input_types):
        collection = self.chroma_client.get_or_create_collection(name="function_registry")
        metadata = {
            "prompt": prompt,
            "function": function_code,
            "imports": "\n".join(imports),
            "comment": comment,
            "input_types": "\n".join(input_types),
            "version": self.version,
        }
        print(metadata)
        collection.add(
            documents=[prompt],
            metadatas=[metadata],
            ids=[prompt_hash]
        )
        #collection.add_documents([document])

    def get_function_with_rag(self, prompt, input_types, top_k=1):
        function_code, prompt_hash = self.get_function(prompt)
        if function_code is None:
            function_code = self.query_chromadb(prompt, "\n".join(input_types), top_k=top_k)
        return function_code

if __name__ == "__main__":
    # Example Usage
    registry = FunctionRegistry(schema="default", version="1.0")

    # Adding a function to the registry
    prompt = "Generate a function that multiplies the input by 5."
    function_code = """
    def generated_function(input_data):
        return input_data * 5
    """
    imports = ["numpy", "math"]
    comment = "This function multiplies the input by 5."
    input_types = ["int"]
    registry.add_function(prompt, function_code, imports, comment, input_types)

    # Retrieving a function from the registry with top_k results
    top_k = 3
    retrieved_function = registry.get_function_with_rag(prompt, input_types, top_k=top_k)
    print("Retrieved Function:\n", retrieved_function)
