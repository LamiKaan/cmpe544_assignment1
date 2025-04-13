# To reproduce results

### 1. Clone repo and set environment

Clone the repo:

```bash
git clone https://github.com/LamiKaan/cmpe544_assignment1.git
```

Navigate inside the root directory:

```bash
cd cmpe544_assignment1
```

Create virtual environment:

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Put data files in proper directories

For expectation maximization algorithm to run properly, its dataset "dataset.npy" file needs to be saved under "data/" directory:

```
data
  |__ dataset.npy
```

For feature extractor and classifiers to run properly, the train/test images and corresponding labels for the subset of the Quick Draw dataset needs to be saved under "data/quickdraw_subset_np/" directory:

```
data
  |__ quickdraw_subset_np
                  |__ test_features.npy
                  |__ test_labels.npy
                  |__ train_images.npy
                  |__ train_labels.npy
```

So, the final directory structure should look like:

```
data
  |__ dataset.npy
  |__ quickdraw_subset_np
                  |__ train_images.npy
                  |__ train_labels.npy
                  |__ test_images.npy
                  |__ test_labels.npy
```

### 3. Run corresponding python files to reproduce results

For expectation maximization:

```
python em/em.py
```

For feature extraction:

```
python feature_extractor/feature_extractor.py
```

For k-NN classifier:

```
python knn/knn_classifier.py
```

For Naive-Bayes classifier:

```
python naive_bayes/naive_bayes_classifier.py
```

For logistic regression classifier:

```
python logistic_regression/logistic_regression_classifier.py
```
