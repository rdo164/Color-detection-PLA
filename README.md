# Color Detection 🚀
```
Detect and identify dominant colors in images using clustering techniques.

Analyze image color palettes with ease.

![License](https://img.shields.io/github/license/rdo164/Color-detection-PLA)
![GitHub stars](https://img.shields.io/github/stars/rdo164/Color-detection-PLA?style=social)
![GitHub forks](https://img.shields.io/github/forks/rdo164/Color-detection-PLA?style=social)
![GitHub issues](https://img.shields.io/github/issues/rdo164/Color-detection-PLA)
![GitHub pull requests](https://img.shields.io/github/issues-pr/rdo164/Color-detection-PLA)
![GitHub last commit](https://img.shields.io/github/last-commit/rdo164/Color-detection-PLA)

<p align="left">
  <a href="https://www.python.org" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.6+-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/Numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white" alt="Numpy">
  </a>
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/scikit_learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  </a>
  <a href="https://matplotlib.org/" target="_blank">
    <img src="https://img.shields.io/badge/Matplotlib-%23F7931E.svg?style=flat-square&logo=matplotlib&logoColor=white" alt="Matplotlib">
  </a>
  <a href="https://pillow.readthedocs.io/en/stable/" target="_blank">
    <img src="https://img.shields.io/badge/Pillow-%23000000.svg?style=flat-square&logo=python&logoColor=yellow" alt="Pillow">
  </a>
</p>

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

## About

This project focuses on detecting the dominant colors within an image using clustering algorithms. It's designed to be a simple and effective tool for analyzing image color palettes. The primary goal is to provide a clear representation of the most prominent colors in a given image, which can be useful in various applications such as image analysis, design, and data visualization.

The project addresses the need for automated color palette extraction, eliminating the manual effort of identifying dominant colors. It's targeted towards developers, designers, and data scientists who require a quick and accurate method for color analysis.

The core technology involves using Python along with libraries like NumPy for numerical computations, scikit-learn for clustering, Pillow for image processing, and Matplotlib for visualization. The architecture is straightforward: load an image, apply a clustering algorithm (e.g., KMeans) to group similar colors, and then represent the cluster centers as the dominant colors. The unique selling point is its ease of use and the ability to quickly extract meaningful color information from images.

## ✨ Features

- 🎯 **Dominant Color Extraction**: Identifies the most prominent colors in an image.
- ⚡ **Performance**: Efficient clustering algorithms for fast processing.
- 🎨 **Customizable**: Allows users to specify the number of dominant colors to extract.
- 📊 **Visualization**: Generates a color palette visualization using Matplotlib.
- 🖼️ **Image Format Support**: Supports various image formats through the Pillow library.
- 🛠️ **Extensible**: Can be easily integrated into other image processing pipelines.

## 🎬 Demo

### Screenshots
![Color Detection Example](screenshots/color_detection_example.png)
*Example of color detection output showing the original image and extracted color palette.*

## 🚀 Quick Start

Clone and run in 3 steps:
```bash
git clone https://github.com/rdo164/Color-detection-PLA.git
cd Color-detection-PLA
pip install -r requirements.txt
python color_detection.py image.jpg 5
```

This will process `image.jpg` and extract the 5 most dominant colors.

## 📦 Installation

### Prerequisites
- Python 3.6+
- pip

### Steps:

# Clone repository
```bash
git clone https://github.com/rdo164/Color-detection-PLA.git
cd Color-detection-PLA
```

# Install dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def detect_dominant_colors(image_path, num_colors):
    """
    Detects the dominant colors in an image using KMeans clustering.

    Args:
        image_path (str): The path to the image file.
        num_colors (int): The number of dominant colors to extract.

    Returns:
        list: A list of RGB tuples representing the dominant colors.
    """
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize for faster processing
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10) # Added n_init
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def visualize_colors(dominant_colors):
    """
    Visualizes the dominant colors as a color palette.

    Args:
        dominant_colors (list): A list of RGB tuples representing the dominant colors.
    """
    plt.imshow([dominant_colors])
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    image_path = 'image.jpg'  # Replace with your image path
    num_colors = 5
    dominant_colors = detect_dominant_colors(image_path, num_colors)
    print("Dominant Colors:", dominant_colors)
    visualize_colors(dominant_colors)
```

To use the script, save the code as `color_detection.py`, replace `'image.jpg'` with the path to your image, and run the script.  You can also pass the image path and number of colors as command-line arguments.

### Command Line Usage

```bash
python color_detection.py image.jpg 5
```

This will process `image.jpg` and extract the 5 most dominant colors.

## ⚙️ Configuration

The number of clusters (dominant colors) can be configured directly in the script:

```python
num_colors = 5  # Change this value
```

You can also configure the image resizing dimensions for performance:

```python
image = image.resize((100, 100))  # Adjust these values
```

## 📁 Project Structure

```
Color-detection-PLA/
├── color_detection.py   # Main script for color detection
├── requirements.txt     # List of dependencies
├── screenshots/         # Example screenshots
│   └── color_detection_example.png
├── LICENSE              # License file
└── README.md            # Project documentation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) (create this file) for details.

### Quick Contribution Steps
1. 🍴 Fork the repository
2. 🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. ✅ Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

## Testing

The project currently lacks automated tests. Contributions to add unit tests and integration tests are highly encouraged.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## 💬 Support

- 📧 **Email**: your.email@example.com (replace with your email)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rdo164/Color-detection-PLA/issues)

## 🙏 Acknowledgments

- 📚 **Libraries used**:
  - [NumPy](https://numpy.org/) - For numerical computations.
  - [scikit-learn](https://scikit-learn.org/) - For KMeans clustering.
  - [Pillow](https://pillow.readthedocs.io/en/stable/) - For image processing.
  - [Matplotlib](https://matplotlib.org/) - For visualization.
```
