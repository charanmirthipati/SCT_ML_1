For information see http://research.microsoft.com/en-us/projects/asirra/corpus.aspx

Contact: jelson

SVM implementation for cat/dog image classification:

1. Install dependencies:
	pip install -r requirements.txt

2. Train and evaluate:
	python svm_cats_dogs.py --dataset-dir PetImages --image-size 64 --test-size 0.2

3. Faster trial run (optional):
	python svm_cats_dogs.py --dataset-dir PetImages --max-per-class 2000

The script trains a Linear SVM, prints accuracy and a classification report,
and saves the trained model to svm_cat_dog.joblib.

Web app:

1. Train the model first so svm_cat_dog.joblib exists.
2. Start the Flask UI:
	python app.py
3. Open the local address shown in the terminal.

The web page lets you upload one image at a time and shows the prediction in a more polished interface.
