# Semantic Communication for Visual Question Answering

This project investigates semantic communication frameworks for Disaster Visual Question Answering (DisasterVQA) using BLIP-2 feature extraction, real-valued neural networks (RVNNs), and complex-valued neural networks (CVNNs).

## Project Pipeline

1. Dataset preprocessing and question tokenization
2. BLIP-2 feature extraction
3. Semantic encoding and transmission through AWGN channels
4. RVNN-based semantic communication
5. CVNN-based semantic communication
6. Performance evaluation on DisasterVQA

## Key Results

| Model | Validation Accuracy |
|---------|---------|
| RVNN + Real Channel | 74.5% |
| RVNN + Complex Channel | 78.3% |
| CVNN + Complex Channel | 78.4% |

## Repository Structure

- `nb1_preprocess.ipynb` – Dataset preprocessing
- `nb2_blip_features.ipynb` – BLIP-2 feature extraction
- `nb3_rvnn_pipeline.ipynb` – RVNN experiments
- `nb4_cvnn_pipeline.ipynb` – CVNN experiments

- TorchCVNN
- Hugging Face Transformers
- DisasterVQA
