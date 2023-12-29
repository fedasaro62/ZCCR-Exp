from typing import List, Tuple
import torch

class Vanilla:
    def __init__(self, emb_size: int=512):
        # to get total length of flat index: index.xb.size()
        # to get number of embeddings in index: index.xb.size() // EMB_SIZE
        self.emb_size       = emb_size
        self.assets_matrix  = torch.empty(emb_size)
        self.faissId2MVId   = []

    def add_content(self, content_embedding: np.ndarray, content_id: str):
        """
        :param content_embedding: embedding having shape (N, EMB_SIZE)
        """
        assert len(content_embedding.shape) == 1, 'Expecting one content at a time'
        assert content_embedding.shape[-1] == self.emb_size, 'Expected embedding size of {}, got {}'.format(self.emb_size, content_embedding.shape[-1])
        content_embedding  = torch.tensor(content_embedding)
        self.assets_matrix = torch.vstack((self.assets_matrix,content_embedding))
        # print(self.assets_matrix.shape)
        self.faissId2MVId.append(content_id)

    def retrieve(self, query_embedding: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        :param query_embedding: np.ndarray having shape (EMB_SIZE,)
        :param k: retrieve top_k contents from the pool
        """
        # print(self.assets_matrix.shape)
        query_embedding            = torch.tensor(query_embedding)
        topk_idx                   = torch.topk(torch.matmul(query_embedding,self.assets_matrix.t()), k).indices
        similarities               = []
        mv_content_ids             = [self.faissId2MVId[idx] for idx in topk_idx]
        
        return mv_content_ids, similarities