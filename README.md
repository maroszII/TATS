# TATS: Toolbox for Augmenting Time Series

**TATS (Toolbox for Augmenting Time Series)** is a MATLAB-based research toolbox designed for systematic evaluation of time series augmentation methods in supervised classification tasks. The toolbox integrates a broad range of augmentation techniques with both classical and deep learning classifiers, providing a unified and extensible experimental framework for reproducible research.

**Authors:**  
Dawid Warchoł, Mariusz Oszust  
Rzeszow University of Technology  

---

## Description

The primary goal of TATS is to facilitate experimental analysis of temporal data augmentation techniques and their impact on classification performance. The toolbox enables controlled comparisons between augmentation methods under multiple classifiers and datasets, supporting repeated experiments, statistical analysis, and result visualization.

---

## Dataset Information

TATS provides interfaces for loading and processing a variety of publicly available benchmark time series datasets commonly used in activity recognition, gesture analysis, biomedical signal processing, and general time series classification. 

The toolbox currently supports the following datasets:

- FLORENCE  
- KARD  
- MSRA  
- UTD-MHAD  
- UTKinect  
- VISAPP  
- AReM  
- AUSLAN  
- ECG  
- EEG  
- GesturePhaseDetect  
- KickVsPunch  
- LIBRAS  
- MovementAAL  
- OccupancyDetect  
- Ozone  
- Pendigits  

---

## Code Information

The toolbox is organized into modular components including augmentation methods implemented as independent MATLAB functions, classification modules supporting both classical and neural network-based models, dataset loaders, and evaluation and visualization routines. The main entry point of the toolbox is the `main.m` script.

---

## Methodology

For a selected dataset, TATS applies one or more augmentation methods to the training set while keeping the test set unchanged. A chosen classifier is then trained and evaluated repeatedly to account for randomness in training and data partitioning. Performance is assessed primarily using classification accuracy, complemented by analysis using box plots and non-parametric statistical significance testing based on the Wilcoxon signed-rank test. Pairwise statistical comparisons are summarized using a heatmap of p-values.

---

## Requirements

### MATLAB Version
- MATLAB 2023b or newer (tested up to MATLAB 2025a)

### Required Toolboxes
- Signal Processing Toolbox  
- Statistics and Machine Learning Toolbox  
- Deep Learning Toolbox  

### Optional
- Parallel Computing Toolbox  
- NVIDIA GPU with CUDA support is recommended for deep learning classifiers

---

## Installation

Clone the repository and set the repository root as the working directory in MATLAB. Ensure that all required toolboxes are installed. No additional installation steps are required.

---

## Usage

Run the main demonstration script:

```matlab
run main.m
````

The script guides the user through dataset selection, augmentation method selection, classifier configuration, repeated experimental evaluation, and result visualization. Batch evaluation of multiple augmentation methods can be enabled by uncommenting the corresponding sections in `main.m`.

---

## References

The datasets supported by TATS originate from publicly available sources. If you use this toolbox in your research, please cite the original dataset providers accordingly.

* FLORENCE 3D Actions Dataset:
  [https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/](https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/)

* KARD Dataset:
  [https://data.mendeley.com/datasets/k28dtm7tr6/1](https://data.mendeley.com/datasets/k28dtm7tr6/1)

* MSR Action3D Dataset:
  [https://sites.google.com/view/wanqingli/data-sets/msr-action3d](https://sites.google.com/view/wanqingli/data-sets/msr-action3d)

* UTD-MHAD Dataset:
  [https://personal.utdallas.edu/~kehtar/UTD-MHAD.html](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)

* UTKinect / HOJ3D Dataset:
  [http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html](http://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html)

* VISAPP Dataset:
  [https://www.scitepress.org/Link.aspx?doi=10.5220/0004217606200625](https://www.scitepress.org/Link.aspx?doi=10.5220/0004217606200625)
  [https://web.archive.org/web/20121025131124/https://mll.sehir.edu.tr/visapp2013](https://web.archive.org/web/20121025131124/https://mll.sehir.edu.tr/visapp2013)

* AReM Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem)

* AUSLAN Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/115/australian+sign+language+signs+high+quality](https://archive.ics.uci.edu/dataset/115/australian+sign+language+signs+high+quality)

* ECG Dataset:
  [https://www.cs.cmu.edu/~bobski/data/data.html](https://www.cs.cmu.edu/~bobski/data/data.html)

* EEG Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/121/eeg+database](https://archive.ics.uci.edu/dataset/121/eeg+database)

* Gesture Phase Segmentation Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation](https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation)

* Kick vs Punch Dataset:
  [http://mocap.cs.cmu.edu](http://mocap.cs.cmu.edu)
  [https://zenodo.org/records/10852865](https://zenodo.org/records/10852865)
  [http://link.springer.com/article/10.1007/s10618-015-0425-y](http://link.springer.com/article/10.1007/s10618-015-0425-y)

* LIBRAS Movement Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/181/libras+movement](https://archive.ics.uci.edu/dataset/181/libras+movement)

* MovementAAL Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/348/indoor+user+movement+prediction+from+rss+data](https://archive.ics.uci.edu/dataset/348/indoor+user+movement+prediction+from+rss+data)

* Occupancy Detection Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/357/occupancy+detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection)

* Ozone Level Detection Dataset (UCI ML Repository):
  [https://archive.ics.uci.edu/dataset/172/ozone+level+detection](https://archive.ics.uci.edu/dataset/172/ozone+level+detection)

* Pen-Based Recognition of Handwritten Digits (Pendigits):
  [https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)

---

## License

This project is licensed under the MIT License.
MIT License
Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
