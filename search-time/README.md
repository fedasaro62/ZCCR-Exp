# Project Name

## Retrieval Search Time Testing with Faiss Index

### Overview

This project aims to evaluate the retrieval search time using the Faiss index in two different scenarios:
1. Single Macro Faiss Index
2. Two Half-Sized Indexes

We will investigate the performance of these scenarios by averaging 100 queries with the following configurations:
- Search Space Size: 1,000,000
- k (Number of Retrieved Elements): 100
- Vector Dimension: 512

### Setting Up the Project

#### Prerequisites

Before running the experiments, ensure you have the following installed:
- Python 3.x
- Virtualenv or Conda (for creating virtual environments)

#### Faiss Installation

The Faiss library provides GPU implementations.

1. **GPU Environment:**

   ```bash
   # Create and activate a virtual environment
   virtualenv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate

   # Install faiss-gpu
   pip install faiss-gpu