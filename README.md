## Fraud Detection in Bitcoin Networks via Temporal Graph Neural Networks (TGNN)

This repository contains the implementation of a Temporal Graph Neural Network (TGNN) designed to detect illicit (fraudulent) transactions within the Elliptic Bitcoin dataset. By treating financial transactions as a dynamic, time-evolving graph, the model effectively captures both the topological structure of transaction flows and their temporal dependencies.

The architecture integrates Graph Attention Networks (GAT) for spatial feature extraction with Gated Recurrent Units (GRU) for sequential temporal modeling, providing a highly robust framework for financial anomaly detection.

## 📌 Core Capabilities

* **Spatial-Temporal Graph Processing**: Extracts structural representations from transaction graphs using multi-head GAT layers and passes summarized graph states through a GRU to maintain temporal context across sequential timesteps.
* **Class Imbalance Optimization**: Implements class-weighted Cross-Entropy loss to heavily penalize the misclassification of minority illicit nodes against the majority licit nodes.
* **Epistemic Uncertainty Estimation**: Utilizes Temporal Monte Carlo (MC) Dropout (50 forward passes) to quantify the model's prediction confidence, providing actionable uncertainty metrics critical for real-world risk assessment and financial compliance.
* **Sequential BPTT Training**: Employs Backpropagation Through Time (BPTT) to train the model sequentially across historical timesteps, ensuring strict temporal data leakage prevention.

## 📂 Dataset Architecture

The project utilizes the **Elliptic Dataset**, mapping Bitcoin transactions to a directed graph:
* **Nodes**: 203,769 transactions.
* **Edges**: 234,355 directed payment flows.
* **Features**: 165 local and aggregate features per node.
* **Temporal Split**: 
    * Timesteps 1-34: Training (BPTT)
    * Timesteps 35-41: Validation
    * Timesteps 42-49: Out-of-time Test Evaluation

## 🧠 Model Architecture (`TemporalFraudGNN`)

1.  **Spatial Pass (Graph Attention)**:
    * Input -> GATv2Conv (Multi-head) -> BatchNorm -> LeakyReLU -> Dropout
    * GATv2Conv -> BatchNorm -> LeakyReLU -> Dropout
2.  **Temporal Pass (Recurrent Memory)**:
    * Global Mean Pooling extracts a generalized graph summary.
    * GRU cell updates the hidden state (`h_next`) using the current graph summary and the prior timestep's memory (`h_prev`).
3.  **Fusion & Classification**:
    * Spatial features and the expanded temporal context are concatenated.
    * Multi-Layer Perceptron (MLP) outputs the final transaction logits.

## 📊 Visualizations and Results

*Note: Please attach the generated visualizations in this section to provide a comprehensive view of the graph topology and model confidence.*

### 1. Transaction Network Topology
This plot visualizes the complex, directed payment flows at a specific temporal snapshot. Node colors represent the ground truth labels (Illicit, Licit, Unknown), highlighting the structural clustering of fraudulent entities.
> ## Elliptic Dataset - Transaction Network (Timestep 1)
![Alt text](https://github.com/ujjalmaji/elliptic_transaction_detection_TGNN/blob/main/elliptic_bitcoin_Greaph_TimeStamp-1.png?raw=true)
### 2. Uncertainty vs. Prediction Confidence
This scatter plot maps the Epistemic Uncertainty (Variance) derived from Temporal MC Dropout against the Mean Probability of Fraud. It illustrates the model's decision boundary and highlights transactions where the model is highly uncertain, allowing for manual auditing of edge cases.
> ## Temporal TGNN Uncertainty vs. Fraud Probability (Test Set)
![Alt text](https://github.com/ujjalmaji/elliptic_transaction_detection_TGNN/blob/main/uncertanity_matrix2.png?raw=true)

## ⚙️ Installation & Requirements

The implementation is built on Python 3 and relies heavily on PyTorch ecosystem libraries.

```bash
!pip install torch torchvision torchaudio
!pip install torch-geometric
!pip install networkx pandas numpy scikit-learn matplotlib
```

## 📖 Usage Guide

1.  **Data Preparation**: Ensure the Elliptic dataset CSV files (`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`) are located in the designated directory (`/content/drive/MyDrive/Elliptic_Dataset/` or modify the `base_path` accordingly).
2.  **Execution**: Run the notebook cells sequentially. The pipeline handles data ingestion, temporal normalization (StandardScaler fit strictly on training timesteps), and PyG `Data` object construction.
3.  **Training**: The training loop automatically applies gradient clipping and early stopping based on the Validation F1-Score.
4.  **Weights**: The optimal model state dict is saved to the specified directory as `best_robust_fraud_gnn_weights_with_class_weighted_loss_updated.pth`.

## 📈 Performance Evaluation

The model evaluates out-of-time data (Timesteps 42-49) using strict chronological separation. Standard metrics computed include Precision, Recall, F1-Score, and ROC-AUC specifically tailored for the minority Class 1 (Fraud).
