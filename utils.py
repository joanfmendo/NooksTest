import json
import re

import pandas as pd
import unicodedata

import numpy as np
from nltk.corpus import stopwords

LANG_DICT = {"es": "spanish", "en": "english", "pt": "portuguese"}
LANGUAGES = ["es", "en", "pt"]
STOP_WRDS = []


class NlpUtils:
    """
    :Date: 2022-08-14
    :Author: Joan Felipe Mendoza
    :Description: NLP and other utilities
    """

    @staticmethod
    def remove_punctuation(text):
        try:
            punctuation = {
                "/",
                '"',
                "(",
                ")",
                ".",
                ",",
                "%",
                ";",
                "?",
                "¿",
                "!",
                "¡",
                ":",
                "$",
                "&",
                ">",
                "<",
                "-",
                "_",
                "°",
                "|",
                "¬",
                "\\",
                "*",
                "+",
                "[",
                "]",
                "{",
                "}",
                "=",
                "'",
                "…",
            }
            for sign in punctuation:
                text = text.replace(sign, " ")
            return text
        except Exception:
            return None

    def clean_text(self, text, lang="en"):
        try:
            stpwrds = list(STOP_WRDS)  # Configurar stopwords
            if lang in LANG_DICT:
                stpwrds += stopwords.words(LANG_DICT[lang])
            else:
                for lng in LANGUAGES:
                    stpwrds += stopwords.words(LANG_DICT[lng])
            output = text.lower()  # converts to lowercase
            output = (
                unicodedata.normalize("NFKD", output)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
            output = re.sub("[\r\n\f]", " ", output)
            output = re.sub("rt @[A-Za-z0-9_]+: ", "", output)  # removes RTs
            output = re.sub(
                " ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", "", output
            )  # removes URLs
            output = self.remove_punctuation(output)  # removes punctuation
            output = " ".join(
                [word for word in output.split() if word not in stpwrds]
            )  # stopwords
            output = re.sub(" +", " ", output)
            return output
        except Exception:
            return ""

    @staticmethod
    def text_to_json(text):
        if text not in [None, np.nan]:
            output = re.sub(r"^\(", r'{"', text)
            output = re.sub(r"$", r'"}', output)
            output = output.replace("): ", '": "').replace("/(", '", "')
            output = json.loads(output)
        else:
            output = {}
        return output

    @staticmethod
    def filter_dict(d, n):
        values = [v for k, v in d.items() if float(k) <= n]
        if values:
            return values[-1]
        else:
            return None

    def get_partial_transcripts(self, df, filter_values):
        derived_values = []
        for _, row in df.iterrows():
            dv = {}
            t = row["partial_transcripts"]
            d = self.text_to_json(t)
            for n in filter_values:
                dv[f"transcript_{n}"] = self.filter_dict(d, n)
            derived_values.append(dv)
        return pd.concat([df, pd.DataFrame(derived_values)], axis=1)
