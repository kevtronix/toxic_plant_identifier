# Plant Identification
This is an exploration into learning about finetuning machine learning models. 


# Important Note
All images scraped from iNaturalist (https://www.inaturalist.org/). These images cannot be used for commercial purposes.

More info on the iNaturalist Open Dataset can be found here: https://github.com/inaturalist/inaturalist-open-data  

Scraping procedure & code can be found here: https://github.com/hans-elliott99/toxic-plant-classification/blob/main/notebooks/scrape-iNaturalist.ipynb

# Dataset
The dataset consists of 6,000 images of plants. The images are classified into 3 classes: 
- Poisonous
- Non-poisonous
- Unknown

# Model
The model is a pretrained ResNet50 model with the final layer replaced with a linear layer with 3 outputs. The model is trained on the dataset and achieves an accuracy of 99.5% on the test set.

