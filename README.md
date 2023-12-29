# Zero-Shot Content-Based Recommender System (ZCCR)

This repository contains experiments and code for the paper titled "Zero-Shot Content-Based Recommender System (ZCCR)." The ZCCR system leverages the CLIP and ALBEF architectures for content-based recommendations.

## Folder Structure

### 1. `clip`
This folder contains dependencies for the CLIP architecture.

### 2. `albef`
This folder contains dependencies for the ALBEF architecture. To use ALBEF, download the 14M pretrained `.pth` checkpoint from [ALBEF GitHub](https://github.com/salesforce/ALBEF?tab=readme-ov-file).

### 3. `data`
- `annotations`: Input annotations file and images for Flickr. Download Flickr images from [Kaggle - Flickr Image Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
- `MSCOCO`: Download MSCOCO images from [MSCOCO Website](https://cocodataset.org/#download). Use `karpathy_coco_split.json` as the annotation file with captions.
- `MSCOCO classified` and `FLICKR30k classified`: Processed versions of MSCOCO and Flickr30k with associated tags for both images and captions.

### 4. `preprocessing`
Contains scripts to generate classified MSCOCO and FLICKR30k annotations along with associated images.

### 5. `retrieval`
Experiments of CLIP and ALBEF on MSCOCO and FLICKR30k 1k validation splits.

### 6. `search-time`
Comparison of search time between one and two FAISS indexes.

### 7. `tagger`
Comparison of ZCCR with Baseline Tagger (BT) and Baseline Tagger + BERT encoding + Agglomerative clustering (BTBA).

### 8. `charts`
Output charts of the ablation study of clustering components and comparisons among baseline and ZCCR.

## Generating Charts
Use `charts.ipynb` to generate charts from the raw results CSV available in the `results` folder.

## Instructions for Setup

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
```bash
source venv/bin/activate # Linux/macOS
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

<!-- ## Citation

If you find this code or the ZCCR system useful, please consider citing our paper:

```bibtex
@article{your-paper-citation-info,
  title={Zero-Shot Content-Based Recommender System (ZCCR)},
  author={Author Name and Another Author},
  journal={Journal Name or Conference Name},
  year={Year},
  volume={Volume},
  number={Number},
  pages={Page Range},
  doi={Your DOI},
  url={Link to Your Paper},
} -->