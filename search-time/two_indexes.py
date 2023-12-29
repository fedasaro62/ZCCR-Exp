import numpy as np
import time
import csv
import os
from tqdm import tqdm
import gc
import argparse

from indexer import Indexer

EMB_SIZE = 512
K = 1000
N_QUERIES = 1000
SEARCH_SPACE_SIZE = 1000000

def create_csv():
    # Specify the CSV file name
    csv_file_name = "search_results.csv"

    # Specify the column headers
    fieldnames = [
        "n indexes",
        "emb size",
        "k",
        "search space size",
        "device",
        "mean time",
        "std",
    ]

    # Write only the header to the CSV file
    with open(csv_file_name, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def add_to_gpu_index(index, data):
    row_norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / row_norms
    index.index.add(data)

def index_test(
        search_space_size: int,
        n_indexes: int,
        device: str,
):
    
    size = int(search_space_size/n_indexes)

    pools = []
    indexes = []
    for _ in range(n_indexes):

        pool = np.random.rand(size, EMB_SIZE).astype(np.float32)
        pools.append(pool)

        index = Indexer(
            emb_size=EMB_SIZE,
            device=device,
            )
        indexes.append(index)

    queries = np.random.rand(N_QUERIES, EMB_SIZE).astype(np.float32)
    
    if device == 'CPU':
        for pool,index in tqdm(zip(pools, indexes)):
            for content in tqdm(pool):
                index.add_content(content)
    
    if device == 'GPU':
        for pool,index in tqdm(zip(pools, indexes)):
            add_to_gpu_index(index, pool)
    
    time_list           = []
    for query in tqdm(queries):
        start_time                    = time.time()
        for index in indexes:
            index.retrieve(query, K)
        elapsed                       = (time.time() - start_time) * 1000
        time_list.append(elapsed)
    
    row = [n_indexes,
           EMB_SIZE,
           K,
           search_space_size,
           device,
           round(np.mean(time_list), 4),
           round(np.std(time_list), 4),
           ]
    
    with open('search_results.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform a search using Faiss on GPU or CPU.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for search (default: CPU)")

    args = parser.parse_args()

    if args.gpu:
        device = "GPU"
    else:
        device = "CPU"

    if not os.path.exists('search_results.csv'):
        create_csv()

    for n_indexes in range(1,3):
        index_test(search_space_size=SEARCH_SPACE_SIZE,
                        n_indexes=n_indexes,
                        device=device)
        gc.collect()
        time.sleep(5)