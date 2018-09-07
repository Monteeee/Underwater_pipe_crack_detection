# Underwater_pipe_crack_detection

#### Directory Content: 
**Demo**, a simple demo, { raw images -> preprocessing (segmenting, cropping) -> large scale classifier for classes 
-> small scale classifier for detecting structures -> show results }

**Networks**, test code for all models we tried, including VGG16(with FC retrained and with SVM as final classifier), MobileNet 
and a shallow convolutional network for comparison.
* **fa_svm.py** file is still under modification, it will not work on your computer.
* since the image dataset is too large, they are not here. and they are **non-public** too. if you know us and have the right 
  to use the original dataset please send email for our modified one.

**Preprocessing**, a script for augmenting images, a script for simply turning images into grayscale and 3 channels, a script for splitting the data randomly.
