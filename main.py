import os
import sys
import numpy
import pandas as pd

# loading the initial data
taxonomy = pd.read_excel('insurance_taxonomy.xlsx')
insurance = pd.read_excel('ml_insurance_challenge.xlsx')

print(insurance)

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
