import os
import sys
import numpy
import pandas as pd
import re
import numpy as np
from collections import deque

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


#       STEP 1: Rule based

# taxonomy = labels ; insurance = companies
taxonomy = pd.read_excel('insurance_taxonomy.xlsx')
insurance = pd.read_excel('ml_insurance_challenge.xlsx')

# extrag label-urile si fac un dictionar de set-uri pt fiecare cuvant din label-rui
labels = taxonomy['label'].dropna().unique()
words_per_label = {}
for l in labels:
    l_words = re.split(r'\W+', l.lower())
    l_words = [w for w in l_words if w]
    words_per_label[l] = set(l_words)


# pt fiecare company text care e defapt totul dintr-o companie concatenat fac un keyword mathcing
# cumva un mod trivial de gasire a lab-urilor dar il folosesc pt dataset pt train si peedictii + invatare
def match_words_from_company_to_label(company_text):
    tokens = re.split(r'\W+', company_text.lower())
    words = set([w for w in tokens if w])
    matched_labels = []

    for l, l_words in words_per_label.items():
        if l_words.issubset(words):
            matched_labels.append(l)

    return matched_labels

# funtie care uneste toate coloanele unei companii intt-o singura linie pt mathcing mai usor
def create_company_row(row):
    fields = []

    for col in ['description', 'business_tags', 'sector', 'category', 'niche']:
        val = row.get(col)
        if pd.notnull(val):
            fields.append(str(val))

    return ' '.join(fields)

# campuri noi in dataset pt textul mare si label-urile gasite
insurance['company_text'] = insurance.apply(create_company_row, axis=1)
insurance['initial_label'] = insurance['company_text'].apply(match_words_from_company_to_label)


#       STEP 2: ML
def label_count(labels):
    return len(labels)

# un datadet mic de training
# incerc un fel de invatare supervizata -> adica consider ca dataset de invatare companiile care au una sau doua label-uri
# am ales pana la doua label-uri ca sa am un dataset ul putin mai mare si poate ajuta si la 'nuantare' 
train_df = insurance[insurance['initial_label'].apply(label_count).between(1, 2)]

# toate label-urile care apar in dataset
all_labels = sorted(set([l for labels in train_df['initial_label'] for l in labels]))

# transform label-urile intr-o matrice si scot datele pt training
mlb = MultiLabelBinarizer(classes=all_labels)
x_train_texts = train_df['company_text'].tolist()
y_train_lists = train_df['initial_label'].tolist()

# transform lista de label-uri intr-o matrice de aparitii
y_train = mlb.fit_transform(y_train_lists)

# transform textul intr-o matrice 
vectorizer = CountVectorizer()
vectorizer.fit(x_train_texts)

# antrenez un model de clasificare pt fiecare label
classifiers = {}
for i, label_name in enumerate(all_labels):
    clf = SGDClassifier(loss='log_loss')
    classifiers[label_name] = clf

X_train_vec = vectorizer.transform(x_train_texts)
for idx, label_name in enumerate(all_labels):
    y_train_bin = y_train[:, idx]
    classifiers[label_name].fit(X_train_vec, y_train_bin)

# bag predictiile pt fiecare label intr-o matrice careia ii fac un f1_score
# practic evaluarea initiala a modelului
def evaluate_multilabel(classifiers, X, Y):
    preds = []
    for label_name in all_labels:
        pred_bin = classifiers[label_name].predict(X)
        preds.append(pred_bin)
    
    preds_matrix = np.vstack(preds).T

    f1 = f1_score(Y, preds_matrix, average='macro', zero_division=0)
    return f1

# increderea curenta a modelului
# incerc sa fac increderea sa fie gen cat de mult pot sa ma incred ca are dreptate ml ul in clasificare
model_trust = evaluate_multilabel(classifiers, X_train_vec, y_train)
print("initial: ", model_trust)


#       STEP 3: Prediction

# clasa pentru a calcula increderea in model
class TrustCalculator:
    def __init__(self, initial_trust):
        self.trust = initial_trust
        self.best_success = 0.0
        self.stale_count = 0
        
    def update(self, success_rate):
        if success_rate > self.best_success:
            improvement = success_rate - self.best_success
            self.trust = min(1.0, self.trust + (improvement * 1.5))
            self.best_success = success_rate
            self.stale_count = 0
        else:
            decay = 0.05 + (self.stale_count * 0.01)
            self.trust = max(0.3, self.trust - decay)
            self.stale_count += 1

trust_calculator = TrustCalculator(initial_trust=model_trust)

# calcularea deciziei
def decision(ml_label, ml_prob, keyword_labels, trust):
    base_threshold = 0.65 - (trust * 0.15)
    keyword_conf = len(keyword_labels) / (len(keyword_labels) + 2)
    
    ml_weight = np.tanh(trust * 3)
    hybrid_score = (ml_weight * ml_prob) + ((1 - ml_weight) * keyword_conf)
    
    if hybrid_score > base_threshold + 0.15:
        return ml_label
    elif hybrid_score > base_threshold and keyword_labels:
        return keyword_labels[0] if trust < 0.4 else ml_label
    elif keyword_labels:
        return keyword_labels[0]
    else:
        return ml_label

# predictia in functie de increderea pe care o am in ml
def predict_with_confidence(classifiers, text):
    vec = vectorizer.transform([text])
    confidences = {}
    
    for label_name in all_labels:
        prob = classifiers[label_name].predict_proba(vec)[0][1]
        confidences[label_name] = prob
    
    best_label = max(confidences, key=confidences.get)
    return best_label, confidences[best_label], confidences


#       STEP 4: Learning and updating

chunk_size = 50
row_count = 0
final_labels = []
history_buffer = deque(maxlen=200)

for idx, row in insurance.iterrows():
    text = row['company_text']
    kw = row['initial_label']
    
    best_label, best_prob, all_confs = predict_with_confidence(classifiers, text)
    
    final_label = decision(
        ml_label=best_label,
        ml_prob=best_prob,
        keyword_labels=kw,
        trust=trust_calculator.trust
    )
    
    if 1 <= len(kw) <= 2:
        xv = vectorizer.transform([text])
        y_bin = np.zeros(len(all_labels), dtype=int)
        
        for lab in kw:
            if lab in all_labels:
                i = all_labels.index(lab)
                y_bin[i] = 1
        
        for i, lab in enumerate(all_labels):
            classifiers[lab].partial_fit(xv, [y_bin[i]], classes=[0,1])
            
        outcome = 1 if final_label == best_label else 0
        history_buffer.append(outcome)
    
    row_count += 1
    if row_count % chunk_size == 0 and history_buffer:
        success_rate = np.mean(history_buffer)

        if len(history_buffer) == history_buffer.maxlen:
            trend = np.polyfit(range(len(history_buffer)), history_buffer, 1)[0]
            success_rate += trend * 2
        
        trust_calculator.update(success_rate)
        print(f"Row {row_count} - Current trust: {trust_calculator.trust:.2f}")
    
    final_labels.append(final_label)
    print(f"Row {idx}: Decision: {final_label}")

insurance["final_label"] = final_labels


#       STEP 5: output salvat in excel
insurance.to_excel('insurance_labels.xlsx', index=False)
