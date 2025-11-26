# AgriScan - Plant Disease Detection System

A machine learning-powered web application for detecting diseases in agricultural crops, with a current focus on cassava plant disease identification. This project uses deep learning models to analyze leaf images and provide real-time disease diagnosis to help farmers make informed decisions about crop management.

## Project Overview

AgriScan is a class project that demonstrates the application of computer vision and deep learning in agriculture. The system currently supports cassava disease detection through image classification, with plans to expand to other crops and add object detection capabilities for more comprehensive plant health analysis.

### Currently Supported Crops & Diseases

#### Cassava (Currently Implemented - Image Classification)
- Bacterial Blight
- Brown Spot  
- Green Mite
- Healthy
- Mosaic

### Future Crop Support (Dataset Available)

#### Cashew (Future Implementation)
- Anthracnose
- Gumosis
- Healthy
- Leaf Miner
- Red Rust

#### Maize (Future Implementation)
- Fall Armyworm
- Grasshopper
- Healthy
- Leaf Beetle
- Leaf Blight
- Leaf Spot
- Streak Virus

#### Tomato (Future Implementation)
- Healthy
- Leaf Blight
- Leaf Curl
- Septoria Leaf Spot
- Verticillium Wilt

## Current Features

- **Real-time Cassava Disease Detection**: Upload cassava leaf images and get instant disease predictions
- **User-friendly Interface**: Web-based interface built with Gradio
- **High Accuracy**: Uses EfficientNet-B1 model pretrained on ImageNet
- **Confidence Scoring**: Provides percentage confidence for each prediction (5 cassava disease classes)
- **Cross-platform**: Runs on any device with a web browser

## Planned Features

- **Object Detection**: Detect and localize diseased areas on plant leaves with bounding boxes
- **Multi-crop Detection**: Support for multiple crop types in a single interface
- **Disease Severity Assessment**: Quantify the extent of disease spread on leaves
- **Real-time Video Analysis**: Process live camera feeds for field monitoring
- **Batch Processing**: Analyze multiple images simultaneously

## Technology Stack

### Current Implementation
- **Backend**: Python
- **Deep Learning Framework**: PyTorch
- **Model Architecture**: EfficientNet-B1 (Image Classification)
- **Web Interface**: Gradio
- **Image Processing**: PIL (Python Imaging Library)
- **Package Management**: Poetry (pyproject.toml)

### Planned Technologies
- **Object Detection Framework**: YOLO v8/v9 or Detectron2
- **Detection Backend**: OpenCV for image processing
- **Advanced Models**: Vision Transformers (ViT) for improved accuracy
- **Mobile Deployment**: TensorFlow Lite or ONNX Runtime

## Project Structure

```
├── app.py
├── deploy.py
├── images
│   ├── cassava-mosaic.jpg
│   ├── CBBD.jpg
│   └── healthy.jpeg
├── models
│   ├── cassava_model.onnx
│   └── model_pretrained_True.pth
├── pyproject.toml
├── README.md
├── requirements.txt
└── utils
    ├── __pycache__
    │   └── utils.cpython-312.pyc
    └── utils.py
                
```

## Installation & Setup

### Prerequisites

- Python 3.12 or higher
- pip or conda package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agri-scan/backend
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision gradio pillow numpy pathlib
   ```

3. **Download the trained model**
   - Ensure `model_pretrained_True.pth` is in the `model/` directory

4. **Run the application**
   ```bash
   python app/inference.py
   ```

5. **Access the web interface**
   - Open your browser and go to `http://localhost:7860`
   - Or use the public Gradio link provided in the terminal

## Usage

### Web Interface (Current)

1. Launch the application using `python app/inference.py`
2. Upload an image of a **cassava leaf**
3. Wait for the model to process the image
4. View the disease prediction results with confidence percentages for the 5 cassava disease classes

### Programmatic Usage (Current)

```python
from app.utils.utils import predict_disease
from PIL import Image

# Load a cassava leaf image
image = Image.open("path/to/cassava_leaf_image.jpg")

# Get prediction (returns probabilities for 5 cassava diseases)
results = predict_disease(image)

# Print results
for disease, confidence in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{disease}: {confidence*100:.2f}%")
```

### Future Usage (Object Detection)

```python
# Planned object detection functionality
from app.utils.detection import detect_disease_regions

# Load image
image = Image.open("path/to/plant_image.jpg")

# Get detection results with bounding boxes
detections = detect_disease_regions(image)

# Results will include:
# - Disease type
# - Confidence score
# - Bounding box coordinates
# - Affected area percentage
```

## Model Details

### Current Classification Model
- **Architecture**: EfficientNet-B1
- **Input Size**: 224x224 pixels
- **Preprocessing**: Resize, normalize with ImageNet statistics
- **Output**: Probability distribution over 5 cassava disease classes
- **Framework**: PyTorch with TorchScript optimization
- **Classes**: 5 (bacterial blight, brown spot, green mite, healthy, mosaic)

### Planned Detection Model
- **Architecture**: YOLO v8 or Detectron2
- **Input Size**: Variable (preserving aspect ratio)
- **Output**: Bounding boxes + classification for diseased regions
- **Features**: Multi-scale detection, real-time inference
- **Training**: Transfer learning from COCO dataset

## Dataset Information

### CCMT Dataset Citation

This project uses the **CCMT Dataset** for crop pest and disease detection:

> Mensah, P. K., Akoto-Adjepong, V., Adu, K., Ayidzoe, M. A., Bediako, E. A., Nyarko-Boateng, O., Boateng, S., Donkor, E. F., Bawah, F. U., Awarayi, N. S., Nimbe, P., Nti, I. K., Abdulai, M., Adjei, R. R., Opoku, M., Abdulai, S., & Amu-Mensah, F. (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. https://doi.org/10.1016/j.dib.2023.109306

### Dataset Overview

The CCMT Dataset is sourced from local farms in Ghana and provides comprehensive crop pest and disease data validated by expert plant virologists.

**Dataset Statistics:**
- **Raw Images**: 24,881 total images
  - Cashew: 6,549 images
  - Cassava: 7,508 images  
  - Maize: 5,389 images
  - Tomato: 5,435 images

- **Augmented Images**: 102,976 total images (used in this project)
  - Cashew: 25,811 images across 5 disease classes
  - Cassava: 26,330 images across 5 disease classes ✅ (currently implemented)
  - Maize: 23,657 images across 7 disease classes
  - Tomato: 27,178 images across 5 disease classes

**Dataset Features:**
- All images are de-identified for privacy
- Expert validation by plant virologists
- Freely available for research community
- Categorized into 22 disease classes across 4 crops
- Split into training and testing sets

## Performance

The system provides confidence scores for each of the 5 cassava disease predictions, helping users understand the reliability of the diagnosis. The EfficientNet-B1 architecture ensures a good balance between accuracy and inference speed for cassava disease detection.

## Development & Research

This project includes Jupyter notebooks for:
- **Data Preprocessing** (`data_preprocessing.ipynb`): Dataset preparation and augmentation
- **Video Processing** (`video_preprocessing.ipynb`): Processing video data for training and object detection research

## Roadmap & Future Enhancements

### Phase 1 (Current) ✅
- [x] Cassava disease detection (5 classes)
- [x] Web interface with Gradio
- [x] EfficientNet-B1 model implementation
- [x] Basic image classification pipeline

### Phase 2 (Next Release)
- [ ] Mobile app with camera integration
- [ ] **Object Detection Implementation** 
  - [ ] YOLO v8 model integration
  - [ ] Bounding box annotation tools
  - [ ] Disease localization on leaves
  - [ ] Severity assessment based on affected area
- [ ] Multi-crop classification models
- [ ] Enhanced web interface for object detection results

### Phase 3 (Advanced Features)
- [ ] **Real-time Video Analysis** 
  - [ ] Live camera feed processing
  - [ ] Batch video processing
  - [ ] Temporal disease tracking
- [ ] **Multi-crop Object Detection**
  - [ ] Cashew disease localization
  - [ ] Maize pest detection
  - [ ] Tomato disease mapping

### Phase 4 (Long-term Vision)
- [ ] Treatment recommendations per detected disease
- [ ] Disease progression tracking over time
- [ ] Multi-language support
- [ ] Offline capability for field use
- [ ] Integration with IoT sensors and drones
- [ ] Farmer community platform with shared insights

## Research Applications

### Computer Vision Techniques
- **Image Classification**: Current implementation using EfficientNet-B1
- **Object Detection**: Planned feature for disease localization using YOLO v8
- **Semantic Segmentation**: Future research for precise disease mapping
- **Instance Segmentation**: Advanced disease boundary detection

### Agricultural Impact
- **Precision Agriculture**: Targeted treatment of affected areas only
- **Early Detection**: Catch diseases before they spread extensively
- **Yield Optimization**: Reduce crop loss through timely intervention
- **Resource Efficiency**: Minimize pesticide use through precise application

## Contributing

This is a class project, but contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is created for educational purposes as part of a class assignment. The CCMT dataset is freely available for research use.

## Team

Auma Denis aumaadinaa@gmail.com annotator

## Contact

For questions about this project, please contact andrewayiko15@gmail.com.

## References

Mensah, P. K., Akoto-Adjepong, V., Adu, K., Ayidzoe, M. A., Bediako, E. A., Nyarko-Boateng, O., Boateng, S., Donkor, E. F., Bawah, F. U., Awarayi, N. S., Nimbe, P., Nti, I. K., Abdulai, M., Adjei, R. R., Opoku, M., Abdulai, S., & Amu-Mensah, F. (2023). CCMT: Dataset for crop pest and disease detection. *Data in Brief*, 49, 109306. https://doi.org/10.1016/j.dib.2023.109306

---

**Note**: AgriScan is currently deployed as a functional cassava disease detection system for real-world agricultural use. While this system demonstrates high accuracy using EfficientNet-B1 architecture trained on expert-validated data, it should be used as a supportive tool alongside traditional agricultural practices. For critical decisions affecting crop management or large-scale agricultural operations, we recommend consulting with local agricultural extension officers or plant pathologists. The object detection and multi-crop features are planned for future releases to enhance the system's capabilities.
