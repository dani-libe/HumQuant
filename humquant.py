import os
import spacy
import pickle
from collections import namedtuple
import random
from cassis import load_typesystem, load_cas_from_xmi
import pandas as pd
import numpy as np
from pprint import pprint

NumberAnnotation = namedtuple("NumberAnnotation", ["tokens", "span"])
AnnotationSpan = namedtuple("AnnotationSpan", ["begin", "end"])

# Concatenate consecutive numeral-like tokens into a single one
ENABLE_FIRST_RULE = True

# Save predicted numbers without any predicted unit
SAVE_NUMBERS_WO_UNITS = False


# Get the spacy pipeline that will be used to process the input raw text
def get_spacy_pipeline(enable_first_rule=True):
  improved_en_pipeline = spacy.load("en_core_web_sm")

  ruler = improved_en_pipeline.add_pipe("entity_ruler", config={"overwrite_ents": True})

  if enable_first_rule:
    ruler.add_patterns([{"label": "CARDINAL", "pattern": [{"LIKE_NUM": True, "OP": "+"}]}]) # Merge consecutive number tokens

  # Rules to detect standardized percentages
  percentages_patterns = [
    [{'LIKE_NUM': True}, {'LOWER': {'IN': ['%', 'percent', 'percentage', 'percentages']}}],
    [{'LIKE_NUM': True}, {'LOWER': 'per'}, {'LOWER': 'cent'}]
  ]

  for pattern in percentages_patterns:
    ruler.add_patterns([{"label": "PERCENT", "pattern": pattern}])

  # Rules to detect standardized dates
  MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
  dates_patterns = [
    [{'LIKE_NUM': True}, {'LOWER': {'IN': MONTHS}}, {'LIKE_NUM': True}],
    [{'LIKE_NUM': True}, {'LOWER': 'of'}, {'LOWER': {'IN': MONTHS}}, {'LIKE_NUM': True}],
    [{'LOWER': {'IN': MONTHS}}, {'LIKE_NUM': True}],
    [{'LIKE_NUM': True}, {'LOWER': {'IN': MONTHS}}]
  ]

  for pattern in dates_patterns:
    ruler.add_patterns([{"label": "DATE", "pattern": pattern}])

  return improved_en_pipeline


def extract_numbers(doc):
  numbers = []
  named_entities = doc.ents

  for ent in named_entities:
    # If the label is related to numerals and is only one token long or is made only of NUM-tagged tokens, I consider it a possible number
    if (ent.label_ == "PERCENT") or (ent.label_ in ["MONEY", "QUANTITY", "CARDINAL"] and (len(ent) == 1 or all(map(lambda t: t.pos_ == "NUM", ent)))):
      if ent.label_ != "MONEY" and ent.start - 1 >= 0 and doc[ent.start - 1:ent.start][0].pos_ != "PROPN":
        span = AnnotationSpan(ent[0].idx, ent[-1].idx + len(ent[-1].text))
        ann = NumberAnnotation(ent, span)
        numbers.append([ann])
    else:
      # If the label is not related to numerals or it is but is more than a token long (or is longer but is not made only of NUM-tagged tokens), I consider only the NUM-tagged tokens as possible numbers (and ig two or more of them are consecutive I consider them as a possible single number)
      if ent.label_ not in ["ORDINAL", "DATE", "TIME"]:
        nums_group = []
        for token in ent:
          if token.pos_ == "NUM" and token.i - 1 >= 0 and doc[token.i - 1:token.i][0].pos_ != "PROPN":
            token_span = AnnotationSpan(token.idx, token.idx + len(token.text))
            num_ann = NumberAnnotation(doc[token.i:token.i + 1], token_span)
            nums_group.append(num_ann)
          else:
            if len(nums_group) > 0:
              numbers.append(nums_group)
              nums_group = []
        
        if len(nums_group) > 0:
          numbers.append(nums_group)
  
  return numbers


def predict_units(numbers):
  def search_for_tags(span, tags):
    tokens = []
    i = 0
    
    while i < len(span) and (span[i].pos_ not in tags or span[i].text in ["%", "percent"]):
      i += 1
      
    while i < len(span) and span[i].pos_ in tags:
      tokens.append((span[i], span[i].idx, span[i].idx + len(span[i].text)))
      i += 1

    return tokens

  
  # Get closest consecutive tokens that match specific constraints
  def get_cct(number_ann):
    tokens_before_number = [elem for elem in number_ann.root.head.subtree if elem.i < number_ann.start]
    tokens_after_number = [elem for elem in number_ann.root.head.subtree if elem.i >= number_ann.end]

    # Currencies symbols are taken from https://github.com/vmasek/CurrencyConverter/blob/master/symbols.csv
    CURRENCIES_SYMBOLS = set(pd.read_csv("data/currencies_symbols.csv", sep="\t")[["Code ISO 4217", "Symbol"]].values.ravel())
    relevant_tokens = []
    for token in tokens_before_number:
      if token.text.upper() in CURRENCIES_SYMBOLS:
        relevant_tokens.append((token, token.idx, token.idx + len(token.text)))
        break
      
    if len(relevant_tokens) == 0:
      relevant_tokens = search_for_tags(tokens_after_number, ["NOUN", "PROPN", "ADJ", "SYM"])

    return relevant_tokens

  
  units_data = {}

  for number_ann in numbers:
    pred_number = number_ann[0].tokens.doc.char_span(number_ann[0].span.begin, number_ann[-1].span.end)

    k = (pred_number.text, number_ann[0].span.begin, number_ann[-1].span.end)
    units_data[k] = get_cct(pred_number)


  return units_data


def get_predicted_units(fnames, en_pipeline, save_numbers_wo_units=False):
  pred_units = {}

  for fname in fnames:
    with open(fname, "r") as f:
      source_text = f.read()
    source_doc = en_pipeline(source_text)
    extracted_numbers = extract_numbers(source_doc)
    pred_units_data = predict_units(extracted_numbers)

    pred_units[fname] = {}
    for num, unit in pred_units_data.items():
      pred_num = (num[0], pd.Interval(num[1], num[2], closed="left"))

      if len(unit) > 0:
        pred_unit = (unit[0][0].doc.text[unit[0][1]:unit[-1][2]], pd.Interval(unit[0][1], unit[-1][2], closed="left"))
        
        try:
          pred_units[fname][pred_num].append(pred_unit)
        except KeyError:
          pred_units[fname][pred_num] = [pred_unit]
      else:
        if save_numbers_wo_units:
          pred_units[fname][pred_num] = []

  return pred_units


if __name__ == "__main__":
  fnames = ["data/annotations_and_sources/source/a1_22580.txt"]
  en_pipeline = get_spacy_pipeline(enable_first_rule=ENABLE_FIRST_RULE)
  pred_units = get_predicted_units(fnames, en_pipeline, save_numbers_wo_units=SAVE_NUMBERS_WO_UNITS)
  pprint(pred_units)
