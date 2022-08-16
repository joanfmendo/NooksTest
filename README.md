# Nooks test - Joan Felipe Mendoza

## Contents

The project consists of two core files containing the scripts:
- **```training.py```**: contains the ```TrainClassModel``` class, which allows training a text classification model, using 4 different algorithms and optimising its hyperparameters.
- **```utils.py```**: contains the ```NlpUtils``` class, which includes auxiliary functions such as text cleaning and derivation of partial transcripts.

It also includes other relevant files:
- **Classification models**: ```model_10.p``` for classifying 10-second texts and ```model_5.p``` for classifying 5-second texts.
- **```books.ipynb```**: a Jupyter notebook that includes the step-by-step training of the models and a very brief explanation of the decisions taken.

***Important***: Please check that you have all the packages included in the **```requirements.txt```** file installed. Packages such as ```optuna``` may require other auxiliary packages to be installed. Other packages such as ```nltk``` require certain steps before use (see https://www.nltk.org/data.html).

## How to use the pre-trained models

```python
import pandas as pd
from pickle import load
from datetime import datetime

from training import TrainClassModel

# Read data
test_data = pd.read_csv('data/evaluation_data.csv', index_col=0)
# Load Model
best_model = load(open("model_10.p", "rb"))

text_col = "transcript"
start = datetime.now()
# Make predictions
preds = TrainClassModel.predict(best_model, test_data[text_col], clean=True)
test_data["predictions"] = preds

end = datetime.now()

print(f"Time elapsed: {end-start} s")
print(f"Predictions done: {len(preds)}")
print(f"Time per prediction: {(end-start)/len(preds)} s")

# Save predictions to file
test_data.to_csv("data/evaluation_preds.csv")
```