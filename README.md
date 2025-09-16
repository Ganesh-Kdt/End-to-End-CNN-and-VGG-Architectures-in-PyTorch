# End-to-End CNN and VGG Architectures in PyTorch

A comprehensive implementation of Convolutional Neural Networks (CNN) and VGG architectures built from scratch using PyTorch for alphanumeric character classification (0-9 digits and A-Z letters).

## 📁 Project Structure

```
├── Builid-Neural-Network-From-Scratch_part_1_2.ipynb  # CNN implementation and training
├── Builid-Neural-Network-From-Scratch_part3.ipynb     # Advanced CNN techniques
├── Builid-Neural-Network-From-Scratch-part4.ipynb     # VGG architecture implementation
├── dataset.csv                                         # Training dataset
├── cnn_dataset.zip                                     # Compressed dataset
├── Builid-Neural-Network-From-Scratch_weights.txt.txt # Model weights
└── README.md                                           # Project documentation
```

## 🎯 Project Overview

This project demonstrates the implementation of deep learning models from scratch, focusing on:

- **Custom CNN Architecture**: Building convolutional neural networks with custom layers for image recognition
- **VGG Implementation**: Implementing the VGG architecture for multi-class image classification
- **Alphanumeric Classification**: Training models to classify handwritten/printed digits (0-9) and letters (A-Z)
- **Model Evaluation**: Comprehensive evaluation using accuracy, loss analysis, and classification metrics

## 📊 Dataset

The project uses an image dataset for alphanumeric character recognition:
- **CNN Dataset** (`cnn_dataset.zip`): Contains images of handwritten/printed characters
- **Classes**: 36 total classes (digits 0-9 and letters A-Z)
- **Format**: Image files organized by character class
- **Additional**: `dataset.csv` contains supplementary structured data
- **Split**: 70% training, 15% validation, 15% testing

## 🚀 Key Features

### Part 1 & 2: CNN Foundation
- Image data preprocessing and visualization
- Custom CNN architecture implementation for multi-class classification
- Cross Entropy Loss function for 36-class classification
- Adam optimizer with learning rate scheduling
- Training and validation loops with image data loaders
- Classification metrics and performance analysis

### Part 3: Advanced CNN Techniques
- Enhanced model architectures for better feature extraction
- Advanced training strategies and regularization
- Image augmentation and data handling improvements

### Part 4: VGG Architecture
- VGG network implementation from scratch
- Deep convolutional architecture with multiple layers
- Model training and weight saving functionality
- Performance comparison with custom CNN

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning metrics and utilities

## 📈 Model Performance

The models are evaluated using:
- **Accuracy**: Multi-class classification accuracy on validation and test sets
- **Loss**: Cross Entropy loss tracking for 36-class classification
- **Confusion Matrix**: Detailed analysis of per-class performance
- **Training Visualization**: Loss and accuracy curves over epochs

## 🔧 Usage

1. **Setup Environment**:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Run the Notebooks**:
   - Start with `Builid-Neural-Network-From-Scratch_part_1_2.ipynb` for basic CNN implementation
   - Progress through parts 3 and 4 for advanced techniques and VGG implementation

3. **Dataset**:
   - Extract `cnn_dataset.zip` to access the image dataset
   - Images should be organized by character class (0-9, A-Z)
   - Ensure `dataset.csv` is in the project root

## 📝 Model Architecture Details

### Custom CNN
- Convolutional layers with ReLU activation for feature extraction
- MaxPooling for spatial dimensionality reduction
- Fully connected layers for 36-class classification
- Dropout for regularization and overfitting prevention

### VGG Architecture
- Multiple convolutional blocks for hierarchical feature learning
- 3x3 convolutional filters throughout the network
- Deep architecture with 16+ layers for complex pattern recognition
- Global average pooling for spatial feature aggregation
- Dense classification head for 36-class output

## 🎯 Learning Objectives

This project covers:
- Understanding CNN architecture from first principles for image recognition
- Implementing VGG networks in PyTorch for deep learning
- Multi-class classification techniques for 36 alphanumeric characters
- Model training best practices for computer vision tasks
- Performance evaluation and visualization for image classification
- Weight saving and model persistence for deployment

## 📄 License

This project is for educational purposes, demonstrating deep learning concepts and implementations.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or additional features.