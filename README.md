-------------------FOR WEBAPP-------------------
Description:
	This python package is a web application where users enter text queries or click on photos to retrieve related images. The dataset used in this package is the ImageNet image set. 
	The images and the text queries are processed using the OpenAI CLIP algorithm and compared using their cosine similarity. 
	The search is displayed through a flask webapp where users can choose to either enter a text search query or select an image from an existing search. 
	This search will return the top 5 results and also a web of images similar to the search that users can interact with.
	
	
Installation:
	- Dataset:
		https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

	- Install required toolkit:
		CUDA 				(https://medium.com/@jjlovesstudying/python-cuda-set-up-on-windows-10-for-gpu-support-78126284b085)

	- Required packages: 
		flask 				(https://flask.palletsprojects.com/en/2.2.x/installation/)
		flask_sqlalchemy 		(https://flask-sqlalchemy.palletsprojects.com/en/3.0.x/quickstart/#installation)
		pytorch 			(https://pytorch.org/get-started/locally/)
		clip 				(https://github.com/openai/CLIP)
		PIL 				(https://pillow.readthedocs.io/en/stable/installation.html)
		numpy 				(https://numpy.org/install/)
		tqdm 				(https://pypi.org/project/tqdm/)
		sklearn 			(https://scikit-learn.org/stable/install.html)
		wtforms 			(https://pypi.org/project/WTForms/)

	Step 1: Download dataset, Install CUDA, and install required python packages with links and instruction provided above.
	Step 2: After downloading the dataset, the dataset will be in a folder called ILSVRC. Take all the folders from "*\ILSVRC\Data\train\*" and move it into our local "static" file so flask can access it. Move it to "*\Clip\static\IMAGES\". Then take both of the "*\ILSVRC\Data\test" and "*\ILSVRC\Data\val" from the dataset and put those two folders into the "*\Clip\static\IMAGES\" folder. The IMAGES folder should have 1002 subfolders. (READ NEXT STEP IF CONFUSED) 
	Step 2: Make sure that the "Clip" file structure follows the following structure:
		|---/Clip 
		|	 |--/all_images.npy
		|	 |--/all_labels.npy
		|	 |--/app.py
		|	 |--/clipthing.py
		|	 |--/forms.py
		|	 |--/main... reason for depression.py
		|		
		|	 |--/npyfiles
		|		 |--/n01440764.npy
		|		 |--/n01443537.npy
		|		 |--/n01484850.npy
		|		 |--/n01491361.npy
		|			... (total 1002 files)
		|	 |--/npylabels
		|		 |--/n01440764_LABELS.npy
		|		 |--/n01443537_LABELS.npy
		|		 |--/n01484850_LABELS.npy
		|		 |--/n01491361_LABELS.npy
		|			... (total 1002 files)
		|	 |--/static
		|		 |--/searchthing.png
		|		 |--/IMAGES
		|			 |--/n01440764
		|				 |--/n01440764_18.JPEG
		|				 |--/n01440764_36.JPEG
		|				 |--/n01440764_37.JPEG
		|				 |--/n01440764_39.JPEG
		|					... (total 1300 files)
		|			 |--/n01443537
		|				 |--/n01443537_2.JPEG
		|				 |--/n01443537_16.JPEG
		|				 |--/n01443537_50.JPEG
		|				 |--/n01443537_71.JPEG
		|					... (total 1300 files)
		|					
		|				... (total 1002 folders)
		|				
		|			 |--/test
		|				 |--/ILSVRC2012_test_00000001.JPEG
		|				 |--/ILSVRC2012_test_00000002.JPEG
		|				 |--/ILSVRC2012_test_00000003.JPEG
		|				 |--/ILSVRC2012_test_00000004.JPEG
		|					... (total 100000 files)
		|			 |--/val
		|				 |--/ILSVRC2012_val_00000001.JPEG
		|				 |--/ILSVRC2012_val_00000002.JPEG
		|				 |--/ILSVRC2012_val_00000003.JPEG
		|				 |--/ILSVRC2012_val_00000004.JPEG
		|					... (total 50000 files)
		|	 |--/templates
		|		 |--/_formhelpers.html
		|		 |--/index.html
		
		
Execution:
	Step 1: Go to the "Clip" folder and open "main... reason for depression.py" in your preferred IDE. Run the python file. I used VSCode
	Step 2: Open up a browser and go to the following URL: http://127.0.0.1:5000
	Step 3: Type search query into the search bar and press "Search"
	Step 4: Wait and images will show up 
	Step 5: Click an image in the top 50 web of images
	Step 6: Wait about 5 seconds (For backend to process image. May take less time if computer is faster)
		Note: You will know that the search is done if you see this in the python terminal after "Found Image": '127.0.0.1 - - [{TIME}] "GET /more?ID={ID number}&sim={similarity} HTTP/1.1" 200'

-------------------FOR TSNE-------------------
Description:
	The file tsne.py demonstrates the capability of tSNE visualization with
	a subset of a classifcation dataset. This package takes images in a
	dataset, embeds them with the CLIP image encoder into a similarity embedding
	space, and then further embeds them in R^2 via the tSNE algorithm. An output
	plot of the results of this process is saved upon execution.

Installation:
	The following packages are required, install them with
		pip install {PackageName}
	The most recent versions will suffice.
	torch
	torchvision
	clip
	sklearn
	matplotlib
	
	If you wish to work with the ImageNet-1k datastet as in the
	example, it can be downloaded from https://image-net.org/download.php.
	Be sure to point the Dataset() class to your dataset file location.

Execution:
	To run the program, simply enter in the terminal:
		python tsne.py