# CSEL204-IMRAD
Language Detection using Naives Bayes Classifiers

# Naive Bayes Language Detector

This project is a simple language detection tool using a Naive Bayes classifier trained on character n-grams. It uses the [Tatoeba Project](https://tatoeba.org/) dataset for training and testing, supporting dozens of languages.

## Notes
- **Large file:** The Tatoeba `sentences.csv` is large. It is not included in this repository. The script will handle downloading and extraction for you.
- **Accuracy:** Accuracy depends on the number of sentences per language and the n-gram size. You can adjust these in the script.
- **Supported languages:** Only languages supported by both Tatoeba and Google Translate are included by default.
- 
## Setup
1. **Install Python 3.7+** (recommended: Python 3.12)
2. **Install dependencies:**
   ```bash
   pip install googletrans==4.0.0rc1
   ```
   (Other dependencies are from the Python standard library.)

## Usage
Run the script:
```bash
python language_detector.py
```
- On first run, if `sentences.csv` is missing, the script will download and extract it automatically (about 200MB download, 1GB extracted).
- The script will load sentences, train the classifier, evaluate accuracy, and then allow you to enter your own sentences for language detection.



## License
This project is for educational purposes. The Tatoeba dataset is released under the [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) license.

## Acknowledgments
- [Tatoeba Project](https://tatoeba.org/) for the multilingual sentence dataset.
- [googletrans](https://github.com/ssut/py-googletrans) for language code mapping.
