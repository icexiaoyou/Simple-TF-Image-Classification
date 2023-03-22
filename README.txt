---[01]Description---

Image classification by tensorflow and opencv.
Only four python scripts can complete the training and deployment of custom model.


---[02]Preparation---

1.Install Visual Studio Community 2022
install the item named "C++ for Desktop Development"

2.Install Anaconda
$ conda create -n tensorflow python>3.8
$ activate tensorflow
$ pip install tensorflow>2.8
$ pip install labelImg
$ pip install opencv-python
$ pip install pillow


---[03]Usage---

1.Put images in folder named "pic_source", subfolders named "classes_index".

2.Run "preprocess.py" to rename and resize the images.

3.Run labelImg to annotate all the images.

4.Run "train.py" to generate a new model.

5.Run "test_model.py" to test the accuracy of model.

6.Run "deploy.py" on computer or embedded device to classify object real-time.