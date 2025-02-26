# Disease Prediction Model   

## Overview  
This project aims to predict future diseases in a patient's medical history using their past diagnoses. The model leverages temporal patterns and disease comorbidities (co-occurrences) to forecast short-term (next time step) and long-term future conditions.  

## Dataset & Training Setup  
- **Data Split**: Patients are divided into training and validation sets.  
- **Training Task**: For each patient and timestamp *t*, predict diseases at *t+1* using diagnoses up to *t*. This encourages learning causal relationships between sequential diagnoses.  
- **Validation Task**: Predict **all future diseases** (>*t+1*) for each patient, simulating real-world forecasting.  

## Model Architecture  
The model combines **Graph Neural Networks (GNNs)** and **Recurrent Neural Networks (RNNs)**:  

### 1. **Comorbidity Graph**  
- **Nodes**: Represent diseases.  
- **Edges**: Created if two diseases co-occur in ≥1 training patient. Edge weights are computed using the **Jaccard Index** to prioritize frequently co-occurring diseases.

### 2. **Neural Network Components**  
- **GATv2Conv Layers**: Update disease embeddings using the comorbidity graph.  
- **GRU (RNN)**: Processes temporal sequences of a patient’s disease history.  
- **Classifier**: Maps the GRU’s final hidden state to scores for all 1,238 diseases.  

## Training Process  
- **Loss Function**: **Bayesian Personalized Ranking (BPR)** ensures scores for observed future diseases are higher than for unobserved ones.  
- **Negative Sampling**:  
  - **Random Negatives**: Randomly sampled diseases not in the patient’s history.  
  - **Hard Negatives**: Diseases strongly correlated with the patient’s existing conditions (via comorbidity graph neighbors), sampled with probability proportional to edge weights.  
  - Hard negatives are used sparingly (10% probability) to avoid overfitting.  
- **Optimizer**: AdamW with batch size 100 for 3 epochs.  

## Validation & Evaluation  
- **Metrics**: Cumulative precision and recall.  
- **Method**: For each patient and timestamp *t*, the model predicts top-*k* diseases (where *k* = actual future disease count). Predictions are compared against the patient's real future conditions.
