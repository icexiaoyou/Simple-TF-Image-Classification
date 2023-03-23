@[TOC](Simple Tensorflow Image Classification)
- [*1. Description*](#-1-description-)
- [*2. Preparation*](#-2-preparation-)
  * [2.1. Install Visual Studio Community 2022](#21-install-visual-studio-community-2022)
  * [2.2. Install Anaconda](#22-install-anaconda)
- [*3. Usage*](#-3-usage-)
- [*4. FlowChart*](#-4-flowchart-)

# *1. Description*

Image classification by **tensorflow** and **opencv**.<br>
Only four python scripts can complete the training and deployment of custom model.


# *2. Preparation*
## 2.1. Install Visual Studio Community 2022
Install the item named "C++ for Desktop Development".

## 2.2. Install Anaconda
```python
// Run Anaconda Prompt, coding...
conda create -n tensorflow python>3.8
activate tensorflow
pip install tensorflow>2.8
pip install labelImg
pip install opencv-python
pip install pillow
```


# *3. Usage*

1.Put images in folder named **"voc_dataset"**, subfolders named **"classes_index"**.

2.Run **"preprocess.py"** to rename and resize the images.

3.Run **labelImg** to annotate all the images.

4.Run **"train.py"** to generate a new model.

5.Run **"test_model.py"** to test the accuracy of model.

6.Run **"deploy.py"** on computer or embedded device to classify object real-time.

 # *4. FlowChart*
 ![Alt](https://img-blog.csdnimg.cn/fda95eb8067944d0a5219b05685171be.png#pic_center)
