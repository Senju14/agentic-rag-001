import os
from pinecone import Pinecone 
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_HOST = os.getenv("PINECONE_HOST") 

def clear_pinecone():
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pinecone.list_indexes().names()
    if PINECONE_INDEX not in indexes:
        print(f"Pinecone index '{PINECONE_INDEX}' not found.")
        return
        
    index_args = {"name": PINECONE_INDEX}
    if PINECONE_HOST:
        index_args["host"] = PINECONE_HOST

    index = pinecone.Index(**index_args)
    index.delete(delete_all=True)
    print(f"Pinecone: All vectors deleted from index '{PINECONE_INDEX}'.")
 

# -------------------------
if __name__ == "__main__":
    confirm = input("Are you sure you want to DELETE ALL data in Pinecone? (yes/no): ")
    if confirm.lower() == "no":
        print("Aborted.")
    else:
        clear_pinecone()
        print("All data cleared.")
