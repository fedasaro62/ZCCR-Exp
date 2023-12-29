import numpy as np
from typing import List, Tuple
import faiss

class Indexer:
    def __init__(self,
                emb_size: int=512,
                device = 'CPU',
                nprobe=1):
        
        # nprobe by default is 1
        self.emb_size     = emb_size

        self.index        = faiss.index_factory(emb_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        if device == 'GPU':
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.index.nprobe = nprobe

    def add_content(self,
                    content_embedding: np.ndarray):
        
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """

        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == self.emb_size, 'Expected embedding size of {}, got {}'.format(self.emb_size, content_embedding.shape[-1])
        content_embedding = content_embedding.reshape(1, -1)
        faiss.normalize_L2(content_embedding)
        self.index.add(content_embedding)

    def retrieve(self,
                 query_embedding: np.ndarray,
                 k: int) -> Tuple[List[str], List[float]]:
        
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """

        query_embedding            = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        self.index.search(query_embedding, k)