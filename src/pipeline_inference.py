import chromadb
from vectorstore import query_vectorstore
from llm_agent import get_object_description

chroma_client = chromadb.PersistentClient(path="../vector_store/")
collection = chroma_client.get_collection("object_embeddings")

def main():
    while True:
        # Accept query input from the user
        query = input("Your question: ")

        # Check for exit condition
        if query.lower() == 'exit':
            print("User terminated the session.")
            break
        
        object_desciption = get_object_description(query)
        results = query_vectorstore(object_desciption, collection)
        print(results)

if __name__ == "__main__":
    main()