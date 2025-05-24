import re
from collections import defaultdict, Counter
import math
import csv
from googletrans import LANGUAGES
import os
import sys
import unicodedata
import urllib.request
import tarfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

NGRAM = 3

if not os.path.exists('sentences.csv'):
    print('sentences.csv not found. Please do not close the program. Downloading from Tatoeba...')
    url = 'https://downloads.tatoeba.org/exports/sentences.tar.bz2'
    urllib.request.urlretrieve(url, 'sentences.tar.bz2')
    print('Extracting sentences.csv from archive...')
    with tarfile.open('sentences.tar.bz2', 'r:bz2') as tar:
        tar.extract('sentences.csv')
    print('Download and extraction complete.')

LANG_NAMES = {
    'eng': 'English', 'fra': 'French', 'spa': 'Spanish', 'deu': 'German', 'ita': 'Italian', 'por': 'Portuguese',
    'rus': 'Russian', 'jpn': 'Japanese', 'cmn': 'Mandarin Chinese', 'zho': 'Chinese', 'ara': 'Arabic',
    'hin': 'Hindi', 'ben': 'Bengali', 'kor': 'Korean', 'vie': 'Vietnamese', 'tur': 'Turkish', 'nld': 'Dutch',
    'swe': 'Swedish', 'fin': 'Finnish', 'dan': 'Danish', 'ell': 'Greek', 'pol': 'Polish', 'ron': 'Romanian',
    'ces': 'Czech', 'hun': 'Hungarian', 'ukr': 'Ukrainian', 'heb': 'Hebrew', 'tha': 'Thai', 'ind': 'Indonesian',
    'msa': 'Malay', 'tam': 'Tamil', 'tel': 'Telugu', 'mar': 'Marathi', 'urd': 'Urdu', 'guj': 'Gujarati',
    'bul': 'Bulgarian', 'hrv': 'Croatian', 'srp': 'Serbian', 'slv': 'Slovenian', 'slk': 'Slovak', 'lit': 'Lithuanian',
    'lav': 'Latvian', 'est': 'Estonian', 'aze': 'Azerbaijani', 'uzb': 'Uzbek', 'kaz': 'Kazakh', 'tgl': 'Tagalog',
    'fil': 'Filipino', 'mal': 'Malayalam', 'pan': 'Punjabi', 'yor': 'Yoruba', 'ibo': 'Igbo', 'hau': 'Hausa',
    'amh': 'Amharic', 'som': 'Somali', 'swh': 'Swahili', 'afr': 'Afrikaans', 'eus': 'Basque', 'glg': 'Galician',
    'cat': 'Catalan', 'isl': 'Icelandic', 'bos': 'Bosnian', 'mkd': 'Macedonian', 'sqi': 'Albanian', 'tat': 'Tatar',
    'kur': 'Kurdish', 'pus': 'Pashto', 'tgk': 'Tajik', 'nep': 'Nepali', 'sin': 'Sinhala', 'mya': 'Burmese',
    'khm': 'Khmer', 'lao': 'Lao', 'mon': 'Mongolian', 'ceb': 'Cebuano', 'ilo': 'Ilocano', 'war': 'Waray',
    'oci': 'Occitan', 'ast': 'Asturian', 'cym': 'Welsh', 'bre': 'Breton', 'bel': 'Belarusian', 'epo': 'Esperanto',
    'fry': 'Western Frisian', 'gle': 'Irish', 'gla': 'Scottish Gaelic', 'ltz': 'Luxembourgish', 'mlt': 'Maltese',
    'yid': 'Yiddish', 'zul': 'Zulu', 'xho': 'Xhosa', 'sot': 'Southern Sotho', 'tsn': 'Tswana', 'tso': 'Tsonga',
    'ssw': 'Swati', 'ven': 'Venda', 'nbl': 'South Ndebele', 'kin': 'Kinyarwanda', 'nya': 'Nyanja', 'sna': 'Shona',
    'kon': 'Kongo', 'lin': 'Lingala', 'sag': 'Sango', 'run': 'Rundi', 'lub': 'Luba-Katanga', 'kik': 'Kikuyu',
    'orm': 'Oromo', 'tir': 'Tigrinya', 'aar': 'Afar', 'aka': 'Akan', 'hye': 'Armenian', 'asm': 'Assamese',
    'aym': 'Aymara', 'bam': 'Bambara', 'bak': 'Bashkir', 'dzo': 'Dzongkha', 'fij': 'Fijian', 'fao': 'Faroese',
    'grn': 'Guarani', 'hat': 'Haitian', 'her': 'Herero', 'ido': 'Ido', 'iii': 'Nuosu', 'ipk': 'Inupiaq',
    'jav': 'Javanese', 'kal': 'Kalaallisut', 'kau': 'Kanuri', 'kas': 'Kashmiri', 'cor': 'Cornish', 'lug': 'Ganda',
    'lim': 'Limburgish', 'mlg': 'Malagasy', 'nau': 'Nauru', 'nde': 'North Ndebele', 'ndo': 'Ndonga', 'nor': 'Norwegian',
    'oss': 'Ossetian', 'pli': 'Pali', 'que': 'Quechua', 'roh': 'Romansh', 'srd': 'Sardinian', 'snd': 'Sindhi',
    'sme': 'Northern Sami', 'smo': 'Samoan', 'sun': 'Sundanese', 'ton': 'Tongan', 'vol': 'Volapük', 'wln': 'Walloon',
    'wol': 'Wolof', 'zha': 'Zhuang'
}

ISO1_TO_ISO3 = {
    'en': 'eng', 'fr': 'fra', 'es': 'spa', 'de': 'deu', 'it': 'ita', 'pt': 'por', 'ru': 'rus', 'ja': 'jpn',
    'zh-cn': 'cmn', 'zh-tw': 'zho', 'ar': 'ara', 'hi': 'hin', 'bn': 'ben', 'ko': 'kor', 'vi': 'vie', 'tr': 'tur',
    'nl': 'nld', 'sv': 'swe', 'fi': 'fin', 'da': 'dan', 'el': 'ell', 'pl': 'pol', 'ro': 'ron', 'cs': 'ces',
    'hu': 'hun', 'uk': 'ukr', 'he': 'heb', 'th': 'tha', 'id': 'ind', 'ms': 'msa', 'ta': 'tam', 'te': 'tel',
    'mr': 'mar', 'ur': 'urd', 'gu': 'guj', 'bg': 'bul', 'hr': 'hrv', 'sr': 'srp', 'sl': 'slv', 'sk': 'slk',
    'lt': 'lit', 'lv': 'lav', 'et': 'est', 'az': 'aze', 'uz': 'uzb', 'kk': 'kaz', 'tl': 'tgl', 'fil': 'fil',
    'ml': 'mal', 'pa': 'pan', 'yo': 'yor', 'ig': 'ibo', 'ha': 'hau', 'am': 'amh', 'so': 'som', 'sw': 'swh',
    'af': 'afr', 'eu': 'eus', 'gl': 'glg', 'ca': 'cat', 'is': 'isl', 'bs': 'bos', 'mk': 'mkd', 'sq': 'sqi',
    'tt': 'tat', 'ku': 'kur', 'ps': 'pus', 'tg': 'tgk', 'ne': 'nep', 'si': 'sin', 'my': 'mya', 'km': 'khm',
    'lo': 'lao', 'mn': 'mon', 'ceb': 'ceb', 'ilo': 'ilo', 'war': 'war', 'oc': 'oci', 'ast': 'ast', 'cy': 'cym',
    'br': 'bre', 'be': 'bel', 'eo': 'epo', 'fy': 'fry', 'ga': 'gle', 'gd': 'gla', 'lb': 'ltz', 'mt': 'mlt',
    'yi': 'yid', 'zu': 'zul', 'xh': 'xho', 'st': 'sot', 'tn': 'tsn', 'ts': 'tso', 'ss': 'ssw', 've': 'ven',
    'nr': 'nbl', 'rw': 'kin', 'ny': 'nya', 'sn': 'sna', 'kg': 'kon', 'ln': 'lin', 'sg': 'sag', 'rn': 'run',
    'lu': 'lub', 'ki': 'kik', 'om': 'orm', 'ti': 'tir', 'aa': 'aar', 'ak': 'aka', 'hy': 'hye', 'as': 'asm',
    'ay': 'aym', 'bm': 'bam', 'ba': 'bak', 'dz': 'dzo', 'fj': 'fij', 'fo': 'fao', 'gn': 'grn', 'ht': 'hat',
    'hz': 'her', 'io': 'ido', 'ii': 'iii', 'ik': 'ipk', 'jv': 'jav', 'kl': 'kal', 'kr': 'kau', 'ks': 'kas',
    'kw': 'cor', 'lg': 'lug', 'li': 'lim', 'mg': 'mlg', 'na': 'nau', 'nd': 'nde', 'ng': 'ndo', 'no': 'nor',
    'os': 'oss', 'pi': 'pli', 'qu': 'que', 'rm': 'roh', 'sc': 'srd', 'sd': 'snd', 'se': 'sme', 'sm': 'smo',
    'su': 'sun', 'to': 'ton', 'vo': 'vol', 'wa': 'wln', 'wo': 'wol', 'za': 'zha'
}

SENTENCES_PER_LANG = 100
MIN_SENTENCES = 10
COMMON_LANGS = [
    'cmn', 'spa', 'eng', 'hin', 'ben', 'por', 'rus', 'jpn', 'pan', 'mar',
    'tel', 'tur', 'kor', 'fra', 'tgl'
]

TRAIN_DATA = defaultdict(list)
supported_tatoeba_langs = set(ISO1_TO_ISO3.values())
with open('sentences.csv', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) < 3:
            continue
        _, lang, sentence = row
        if lang in supported_tatoeba_langs and len(TRAIN_DATA[lang]) < SENTENCES_PER_LANG:
            TRAIN_DATA[lang].append(sentence)
            print(f"Loading {lang} ({LANG_NAMES.get(lang, lang)}): {len(TRAIN_DATA[lang])}/{SENTENCES_PER_LANG}", end='\r')
print()

if os.environ.get('COMMON_LANGS_ONLY') == '1':
    TRAIN_DATA = {lang: sents for lang, sents in TRAIN_DATA.items() if lang in COMMON_LANGS}
    languages = list(TRAIN_DATA.keys())
    print(f"Loaded {len(languages)} of the 15 most common languages (with Tagalog/Filipino) with at least {MIN_SENTENCES} sentences each.")
    for lang in COMMON_LANGS:
        print(f"{lang}: {len(TRAIN_DATA.get(lang, []))} sentences")

def extract_ngrams(text, n=NGRAM):
    text = ''.join(c for c in text.lower() if unicodedata.category(c).startswith('L'))
    return [text[i:i+n] for i in range(len(text)-n+1)]

class NaiveBayesLanguageDetector:
    def __init__(self):
        self.lang_ngram_counts = defaultdict(Counter)
        self.lang_total_ngrams = defaultdict(int)
        self.lang_priors = defaultdict(float)
        self.vocab = set()
    def train(self, data):
        total_texts = sum(len(texts) for texts in data.values())
        for lang, texts in data.items():
            self.lang_priors[lang] = math.log(len(texts) / total_texts)
            for text in texts:
                ngrams = extract_ngrams(text)
                self.lang_ngram_counts[lang].update(ngrams)
                self.lang_total_ngrams[lang] += len(ngrams)
                self.vocab.update(ngrams)
        self.vocab_size = len(self.vocab)
    def predict(self, text):
        ngrams = extract_ngrams(text)
        scores = {}
        for lang in self.lang_ngram_counts:
            log_prob = self.lang_priors[lang]
            for ng in ngrams:
                count = self.lang_ngram_counts[lang][ng]
                prob = (count + 1) / (self.lang_total_ngrams[lang] + self.vocab_size)
                log_prob += math.log(prob)
            scores[lang] = log_prob
        return max(scores, key=scores.get), scores

if __name__ == "__main__":
    train_data = {}
    test_data = {}
    for lang, sents in TRAIN_DATA.items():
        if len(sents) >= 100:
            train_data[lang] = sents[:90]
            test_data[lang] = sents[90:100]
        else:
            train_data[lang] = sents
            test_data[lang] = []
    detector = NaiveBayesLanguageDetector()
    detector.train(train_data)
    total = 0
    correct = 0
    per_lang_results = {}
    for lang, sents in test_data.items():
        lang_total = len(sents)
        lang_correct = 0
        for sent in sents:
            pred_lang, _ = detector.predict(sent)
            if pred_lang == lang:
                lang_correct += 1
        if lang_total > 0:
            per_lang_results[lang] = (lang_correct, lang_total)
            total += lang_total
            correct += lang_correct
    with open("language_detector_log.txt", "w", encoding="utf-8") as logf:
        for lang in sorted(per_lang_results.keys()):
            lang_name = LANG_NAMES.get(lang, lang)
            lang_correct, lang_total = per_lang_results[lang]
            acc = lang_correct / lang_total if lang_total > 0 else 0
            logf.write(f"{lang} ({lang_name}): {acc:.2%} ({lang_correct}/{lang_total})\n")
        if total > 0:
            overall_acc = correct / total
            logf.write(f"\nOverall accuracy: {overall_acc:.2%} ({correct}/{total})\n")
        else:
            logf.write("Not enough test data to calculate accuracy.\n")
    if total > 0:
        print(f"Test accuracy: {overall_acc:.2%} ({correct}/{total})")
    else:
        print("Not enough test data to calculate accuracy.")

    # After accuracy test, allow user to enter sentences for prediction
    print("\nEnter text to detect language (empty to quit):")
    while True:
        text = input("> ").strip()
        if not text:
            break
        lang, _ = detector.predict(text)
        lang_name = LANG_NAMES.get(lang, lang)
        print(f"Predicted language: {lang} ({lang_name})")

google_tatoeba_langs = set(LANGUAGES.keys())
TRAIN_DATA = {lang: sents for lang, sents in TRAIN_DATA.items() if lang in google_tatoeba_langs}
languages = list(TRAIN_DATA.keys())
print(f"Loaded {len(languages)} Google Translate-supported languages with at least {MIN_SENTENCES} sentences each.")
google_tatoeba_langs = set(ISO1_TO_ISO3.values())
TRAIN_DATA = {lang: sents for lang, sents in TRAIN_DATA.items() if lang in google_tatoeba_langs}
languages = list(TRAIN_DATA.keys())
print(f"Loaded {len(languages)} Google Translate-supported languages with at least {MIN_SENTENCES} sentences each.")
print()