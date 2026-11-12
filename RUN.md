# How to run (tested with Python 3.13, Flask 3, scikit-learn 1.9)

python -m venv env
env\Scripts\activate          (Windows)   |   source env/bin/activate  (Linux/macOS)
pip install -r requirements.txt
python train_model.py         (run once: rebuilds data/diseasePredictModel.pkl for YOUR scikit-learn version)
python startApp.py
Open http://127.0.0.1:5000/

Changes made to the original repo:
- README said strtServer.py, the real file is startApp.py
- Original requirements were pinned to old versions that do not install on new Python; replaced with unpinned list
- Old .pkl model could not be loaded with new scikit-learn -> added train_model.py to retrain from dataset.csv
- Fixed trailing-space bug ("Diabetes ", "Hypertension ") and a spelling mismatch (hemorrhoids) that caused HTTP 500 errors
