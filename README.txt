CS7602 : Machine Learning Techniques
========================================

Course Project
-------------

--------------------------------------------
Landmark Recogntion
--------------------------------------------

Authors:
--------
Nandhini S	-	2017103559
Agalya P	-	2017103505

--------------------------------------------

Folders/Files Provided:
-----------------------
- Code
	- landmark recognition

- Dataset
	- data (folder)
		- train (dataset images)
			-25 folders each containing 1700 images approximately
		- validation (test images)
			-25 folders each containing 400 images approximately

-------------------------------------------------------------------------------------------------------------------------------

TO DO:
-----------------------
Initially run data_preprocess.py 
$ :python data_preprocess.py
Then run vgg16_final.py to use VGG16 model and visualise the predictions
$ :python vgg16_final.py
To know the loss and accuracy on other models tried run basic_CNN.py and XceptionV1.py 
$ :python basic_CNN.py
$ :python XceptionV1.py
------------------------------------------------------------------------------------------------------------------------------

The Flow :
-----------

The project proceeds in the following manner working through different codes and files present where you can run particular code to get desired output :

1.Run data_preprocess.py to do the initial data cleaning and preprocessing.This removes not found and not downloaded properly images from train and validation folder images
-> Output:Valid images with pixel values rescaled from [0, 255] to [0, 1] interval

2. Run basic_CNN.py to train and evaluate the model(neglected since produces less accuracy and overfits)
-> Output: loss and accuracy on evaluation

3. Run XceptionV1.py to train and evaluate on the XceptionV1 model(neglected since produces less accuracy and overfits)
-> Output: loss and accuracy on evaluation

4. Run vgg16_final.py to get the final predictions on the image
-> Output: The prdecition of landmark of the image 


----------------------------------------------------------------------------------------------------------------------------------


Results:
--------

Github Repository : https://github.com/adityasurana/Google-Landmark-Recognition-Challenge

-----------------------------------------------------------------------------------------------------------------------------------
