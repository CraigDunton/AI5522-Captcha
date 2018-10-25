### Before you get started

To run these scripts, you need the following installed:

1. Python 3
2. OpenCV 3 w/ Python extensions
 - I highly recommend these OpenCV installation guides: 
   https://www.pyimagesearch.com/opencv-tutorials-resources-guides/ 
3. The python libraries listed in requirements.txt
 - Try running "pip3 install -r requirements.txt"

### Step 1: Download the dataset

The data set can be found at this link:

https://s3-us-west-2.amazonaws.com/mlif-example-code/solving_captchas_code_examples.zip

once extracted, put the "generated_letter_images" folder in this projects main directory

### Step 2: Extract single letters from CAPTCHA images

Run:

python3 extract_single_letters_from_captchas.py

The results will be stored in the "extracted_letter_images" folder.

**Note: you only need to run this once**


### Step 3: Train the neural network to recognize single letters

Run:

python3 train_model.py
