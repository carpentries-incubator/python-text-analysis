---
title: Setup
exercises: 20
---
## Download the helpers.py file
This file contains helper functions that we'll use throughout the workshop.

1. Click the link below to open the Github page. * [helpers.py](code/helpers.py)
2. Right click on the "Raw" button and select ```Save As```. Place it into a new folder called "text-analysis" on your Desktop (e.g., `/Users/username/Desktop/text-analysis/helpers.py`)

## Choose Colab or Local Install
You can either run this workshop on your local machine or Google Colab. Colab was chosen to ensure all learners have similar processing power (using Google's servers), and to streamline the setup required for the workshop. If you prefer to setup a local environment and download all necessary packages, please review the next section. However, unless your local machine has at least 12 GB of RAM AND a graphics card with CUDA enabled, your code may run slower than the rest of the learners.

## Google Colab Setup
We will be using [Google Colab](https://research.google.com/colaboratory/faq.html) to run Python code in our browsers.

1. Visit the [Google Colab website](https://colab.research.google.com/) and click "New notebook" from the pop-up that shows up
2. Visit [Google Drive](https://drive.google.com/drive/my-drive) and find a newly created "Colab Notebooks" folder stored under MyDrive, ```/My Drive/Colab Notebooks```
3. Create a folder named ```text-analysis``` in the Colab Notebooks folder on Google Drive. The path should look something like this: ```/My Drive/Colab Notebooks/text-analysis/```.
4. Upload the helpers.py file to the text-analysis directory.
4. **Note for instructors**: At the start of each section, create a new Colab file within the text-analysis folder by navigating to ```/My Drive/Colab Notebooks/text-analysis/``` within [Google Drive](https://drive.google.com/drive/my-drive), followed by clicking ```New -> More -> Google Colaboratory```

## Anaconda Local Setup (Alternative to Colab)
Software dependencies for data science can be complicated and easily conflict with each other, therefore we will be using a self-contained 'virtual environment' to manage their installation.

1. Install [Anaconda](https://www.anaconda.com/products/distribution). Anaconda is a software manager that will help manage dependencies and properly configure use of your graphics card (if you have one). Follow the installation instructions specified for your operating system.
2. Download the appropriate environment.yml file to the working directory where you will be doing this workshop. Make sure to right click on the "raw" file and select save as to save a copy.
   1. [CPU environment](files/environment_cpu.yml) configures an environment for CPU only. If you are not sure what environment to use, use this one.
   2. [CUDA 11.8 environment](files/environment_cuda118.yml) configures an environment for a CUDA 11.8 enabled graphics card.
3. Launch Anaconda Navigator. Select ```Environments -> Import```. Select the environment file you downloaded for the import. For the name of the environment, type in ```text-analysis```. WARNING: This environment will likely take at least 10 minutes to configure and may take up to an hour. Give the installer time to configure the environment.
4. Once installed, click the play button next to the text-analysis environment and select ```Open with Python``` to open a python terminal. You can close and relaunch this python terminal as needed.
