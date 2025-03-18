import os
import sys
import numpy
import pandas as pd
import re

# loading the initial data
taxonomy = pd.read_excel('insurance_taxonomy.xlsx')
insurance = pd.read_excel('ml_insurance_challenge.xlsx')

# Made the initial plan:
# Rule based + RL or auto learner 
# initial va fi rule based, sau dupa niste keywords 
# iar companiile obvious vor primii label
# in acelasi timp, ml ul invata din predictiile sale si cele date de rule based
# dupa analizeaza nuantele din fiecare companie si la fel se invata
# apoi in fucntie de bumarul de keywords uri gasite intr o companie ml ul va avea o influenta mai mare asupra deciziei finale
# acesta poate adapta si regulile de baza (inca nu sunt sigur de asta)
# dar pe scurt invata in timp ce le da label si analizeaza si nuantele 
# iar datset ul nu e unul traditional ci invata pe masura ce analizeaza mai mult 


labels = taxonomy['label'].dropna().unique()
words_per_label = {}
for l in labels:
    l_words = re.split(r'\W+', l.lower())
    l_words = [w for w in l_words if w]
    words_per_label[l] = set(l_words)


def match_words_from_company_to_label(company_text):
    words = set(company_text.lower().split())
    matched_labels = []

    for l, l_words in words_per_label.items():
        if l_words.issubset(words):
            matched_labels.append(l)

    return matched_labels

def create_company_row(row):
    fields = []

    for col in ['description', 'business_tags', 'sector', 'category', 'niche']:
        val = row.get(col)
        if pd.notnull(val):
            fields.append(str(val))

    return ' '.join(fields)


insurance['company_text'] = insurance.apply(create_company_row, axis=1)

company_text = insurance.loc[1, 'company_text']
matched = match_words_from_company_to_label(company_text)
print(company_text)
print(matched)
print(len(matched))
print(len(taxonomy))
