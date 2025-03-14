# AI Grammar Correction

An application that utilizes Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and text embeddings to analyze and correct grammatical errors in text.

## Features

- **Deep Learning Models**: Uses CNN and LSTM architectures to effectively learn and correct grammatical structures.
- **Text Embeddings**: Incorporates advanced embedding techniques to understand contextual nuances in language.
- **Modular Design**: Structured with clear modules for attention mechanisms, encoding, decoding, and training processes.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CatalinPoata/AI_Grammar_Correction.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd AI_Grammar_Correction
   ```

3. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   ```

## Usage

1. **Prepare your dataset**: Ensure you have a dataset with correct and incorrect sentences for training.

2. **Train the model**:

   ```bash
   python train.py
   ```

3. **Use the trained model for correction**:

   ```bash
   python main.py --input "Your sentence with error"
   ```

   This will output the corrected sentence.

## Project Structure

- `Attention.py`: Contains the implementation of the attention mechanism.
- `Encoder.py`: Defines the encoder model architecture.
- `Decoder.py`: Defines the decoder model architecture.
- `OneStepDecoder.py`: Implements the one-step decoding process.
- `EncDecMain.py`: Integrates the encoder and decoder models.
- `train.py`: Script for training the model.
- `main.py`: Main script for running the grammar correction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

---

For more details, visit the [AI Grammar Correction GitHub repository](https://github.com/CatalinPoata/AI_Grammar_Correction).
