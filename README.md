
# LoRA and q-LoRA Implementations

## Overview

This repository contains implementations of Low-Rank Adaptation (LoRA) and Quantized Low-Rank Adaptation (q-LoRA) integrated with a BERT model. These techniques are used to adapt pre-trained language models efficiently for specific tasks by introducing low-rank updates to the model's weights.

### Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) introduces low-rank matrices to adapt the pre-trained model's weights without fully fine-tuning the entire model. This technique is computationally efficient and reduces the number of trainable parameters. The LoRA layer adds two low-rank matrices to the existing weights, which are trained during the adaptation process.

**Advantages of LoRA:**
- Reduces the number of trainable parameters.
- Efficient adaptation of large pre-trained models.
- Maintains the original model structure.

### Quantized Low-Rank Adaptation (q-LoRA)

Quantized Low-Rank Adaptation (q-LoRA) extends the LoRA approach by quantizing the low-rank matrices. Quantization reduces the precision of the matrices to lower bit-width (e.g., 8-bit integers), which further reduces memory usage and improves computational efficiency. The quantized matrices are dequantized during the forward pass for computations.

**Advantages of q-LoRA:**
- Further reduces memory usage compared to LoRA.
- Improves computational efficiency.
- Maintains performance with reduced precision.

## Implementation Details

### LoRA Implementation

The LoRA implementation consists of the following steps:

1. **Import Libraries**: Import necessary libraries including PyTorch, Transformers, and Datasets.
2. **Define LoRA Layer and Model**: Define the `LoRALayer` class to introduce low-rank adaptation and the `LoRABertModel` class to integrate this layer into a BERT model.
3. **Initialize Tokenizer and Model**: Initialize the BERT tokenizer and the `LoRABertModel`.
4. **Load and Prepare Dataset**: Load the SST-2 dataset from GLUE, tokenize it, and prepare DataLoader for training and validation.
5. **Training Loop**: Train the model using AdamW optimizer and linear learning rate scheduler.
6. **Validation Loop**: Evaluate the model on the validation set and calculate accuracy.

### q-LoRA Implementation

The q-LoRA implementation consists of the following steps:

1. **Import Libraries**: Import necessary libraries including PyTorch, Transformers, and Datasets.
2. **Define Quantized LoRA Layer and Model**: Define the `QuantizedLoRALayer` class to introduce quantized low-rank adaptation and the `qLoRABertModel` class to integrate this layer into a BERT model.
3. **Initialize Tokenizer and Model**: Initialize the BERT tokenizer and the `qLoRABertModel`.
4. **Load and Prepare Dataset**: Load the SST-2 dataset from GLUE, tokenize it, and prepare DataLoader for training and validation.
5. **Training Loop**: Train the model using AdamW optimizer and linear learning rate scheduler.
6. **Validation Loop**: Evaluate the model on the validation set and calculate accuracy.

## Files

- `LoRA_implementation.ipynb`: Jupyter notebook implementing LoRA with a BERT model.
- `qLoRA_implementation.ipynb`: Jupyter notebook implementing q-LoRA with a BERT model.

## Usage

### Prerequisites

Ensure you have the following libraries installed:
```bash
pip install torch transformers datasets
```

### Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/bharadwajvyadavalli/lora-adapt
```

2. Navigate to the repository directory:
```bash
cd lora-adapt
```

3. Open the notebooks in Jupyter:
```bash
jupyter notebook
```

4. Run the cells in `LoRA_implementation.ipynb` and `qLoRA_implementation.ipynb` to execute the code and train the models.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
