from config import Config, Singleton
import pinecone
import openai

cfg = Config()


def memory_factory(memory_name='auto-gpt'):
    """
    Returns a memory object.
    If there are pinecone and openAI api keys, it will return a PineconeMemory object.
    Otherwise, it will return a SimpleMemory object.
    If there is an exception initializing the PineconeMemory object, it will return a SimpleMemory object.
    """
    try:
        if cfg.pinecone_api_key and cfg.pinecone_region and cfg.openai_api_key:
            pinecone_mem = PineconeMemory(memory_name)
        else:
            return SimpleMemory()
        # test the openai api since we need it to make embeddings.
        get_ada_embedding("test")
        return pinecone_mem
    except Exception as e:
        print(f"Error initializing memory {e}\nUsing Simple Memory.")
        return SimpleMemory()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class PineconeMemory(metaclass=Singleton):
    def __init__(self, table_name):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with memory.
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)
        self.namespace = 'permanent_memory'

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def delete(self, ids):
        _text = f"Deleting data from memory at index: {ids}"
        self.index.delete(ids)
        return _text

    def overwrite(self, index, vector_id, data):
        _text = f"Overwriting data in memory at index: {index} with data:\n {data}"
        query_embedding = get_ada_embedding(data)
        self.index.update(index, id=vector_id, vector=query_embedding)
        return _text

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()


class SimpleMemory(metaclass=Singleton):
    """A simple memory that stores the data in a list"""

    def __init__(self):
        self.memory = []

    def add(self, data):
        self.memory.append(data)
        _text = f"""Committing memory with string "{data}" """
        return _text

    def get(self, index):
        return self.memory[index]

    def delete(self, key):
        if key >= 0 and key < len(self.memory):
            _text = "Deleting memory with key " + str(key)
            del self.memory[key]
            print(_text)
            return _text
        else:
            print("Invalid key, cannot delete memory.")
            return None

    def overwrite(self, key, data):
        if int(key) >= 0 and key < len(self.memory):
            _text = "Overwriting memory with key " + \
                    str(key) + " and string " + data
            self.memory[key] = data
            print(_text)
            return _text
        else:
            print("Invalid key, cannot overwrite memory.")
            return None

    def clear(self):
        self.memory = []
        return "Obliviated"

    def get_relevant(self, _):
        """
        Returns all the data in the memory.
        In more compliated memory objects this will return the most relevant data
        """
        return self.memory

    def get_stats(self):
        return {
            "size": len(self.memory),
            "total_chars": sum([len(item) for item in self.memory])
        }
