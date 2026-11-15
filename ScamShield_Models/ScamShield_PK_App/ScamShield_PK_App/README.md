# ScamShield PK

A Streamlit interface for the Logistic Regression + TF-IDF model trained in Google Colab.

## 1. Add your model files

Copy these two files from Colab into this folder, beside `app.py`:

- `scamshield_model.pkl`
- `scamshield_vectorizer.pkl`

Only load pickle/joblib files that you created and trust.

## 2. Run locally

Install Python 3.10 or newer, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Publish

Push this folder and the two model files to a GitHub repository (if the model files are too large for GitHub, use Streamlit Community Cloud's supported storage approach or another trusted hosting method). Then create an app at https://share.streamlit.io/ and select the repository, branch, and `app.py`.

## Important evaluation notes

- The app is a prototype and not a certified fraud detector.
- Report actual test-set precision, recall, confusion matrix, and five misclassified examples from your notebook.
- The Pakistan-style messages are synthetic examples, not a representative real-world dataset.
- Do not claim Urdu/Roman Urdu performance unless you separately collect and test suitable labelled examples.
- The model score is not calibrated as a real-world fraud probability.
- Disclose the dataset, synthetic examples, scikit-learn, Streamlit, and any AI tools used.
