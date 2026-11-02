# Spam Classifier — corrected project files

## Before running
Keep your existing dataset at:
`data/spam.csv`

The dataset is not included in this package because it was not attached with the notebook.

## 1) Install packages
Open the VS Code terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

## 2) Run the notebook
Open `Spam_Classifier_Fixed.ipynb` in Jupyter/VS Code.
Choose **Restart Kernel and Run All** (or run every cell from top to bottom).
The last cells save and verify:
- `vectorizer.pkl`
- `model.pkl`

Important: the saved model is the fitted `VotingClassifier`, not the unfitted standalone `mnb`.

## 3) Start Streamlit
In the same project folder, run:

```bash
streamlit run app.py
```

If `spam.csv` is in a different location, update the candidate path in cell 1 of the notebook.
