# Company Classifier

## Table of Contents

- [Overview](#overview)
- [Idea and Workflow](#workflow)
- [Analysis](#analysis)

## Overview <a name = "overview"></a>

In acest proiect am implementat un sistem hybrid da labelling a companiilor. Ideea are 2 parti: rule-based si supervised ML. In final sistem asigneaza fiecarei companii cate un label.

## Idea and Workflow <a name = "workflow"></a>

Idea se invarte in jurul unui superveised learning. Initial, partea rule-based, foloseste cuvinte din label-urile data si face un keyword matching ca sa gaseasca seturi de label-uri pentru fiecare companie. Din label-urile gasite, companiile cu 1 sau 2 label-rui sunt folosite pe post de un dataset. Cat progreseaza sistemul, la fiecare 50 de comapnii updateza clasifierele cu noua data folosing *partial_fit* si ajusteaza trust-ul pe care il are in ml. Acest trust reprezinta increderea pe care o am in ML pentru a da rezultatul care trebuie. Decizia finala se face intr-un mod dinamic, folosindu-se de predictiile ML-ului, in functie de trust-ul acestuia si label-urile gasite la keywords matching.

**Worflow pe pasi**:
1. Rule-Based Labeling
    * Citesc excel-urile si creez dataset-ruile
    * Concatenez toate field-urile unei companii intr-un text, si pentru fiecare text rezultat caut sa gasesc daca este vreun label prezent.
2. Model Setup
    * Subeset-ul pt training de una sau doua label-rui
    * Transform label-urile intr-o matrice binara folosind *MultiLabelBinarizer*
    * Vectorizez text-ul concatenat folosind CountVectorizer
    * Pt fiecare label unic folosesc un clasificator (*SGDClassifier*) antrenat pe data vectorizata
3. Prection and Trust
    * In clasa TrustCalculator practic mentin increderea in predictiile ml-ului. Tristul este updatat in functie de predcitiile recente
    * Functia de decizie ia in calcul predictiile ML ului impreuna cu trust-ul acesteia si rezultatele din keyword matching
4. Online learning and Updating
    * Decizia finala e bazata pe pe scorul calculat hybrid
    * Pt companiile cu 1 label sau 2 updatez classifierele
    * La fiecare 50 de companii updatez trustul in ML
5. Output
    * Salvez labelurile

## Analysis <a name = "analysis"></a>

Cred ca strength-ul metodei mele vine de la abordarea hibrida. Incerc sa imbin viteza rule-based-ului cu flexibilitatea ML-ului, obtinand un rezultat cat mai bun intr-un timp cat mai mic. Online Learning-ul ii permite modelului sa de updateze dinamic ceea ce ajuta la adaptarea la noi date. Logica de *incredere* cumva balanseaza output-ul in functie de cat de bine prezice ml-ul.
Marele weakness-uri ale acestei idei cred ca vine din simplitatea rule-based-ului si dataset-ul mic. Metoda mea curenta de rule-based este foarte simplista si nu ia in considerare sinonime sau mici variatii ale cuvintelor. Cred ca acesta este un aspect care sigur poate fi imbunatatit. Pe de alta parte, dataset-ul mic format din comapnii cu una sau doua label-uri limiteaza foarte mult predcitiile initiale si starea din care pleaca ML-ul.
Overall, modelul acest hybrid exceleaza in cazurile in the rule-based-urile dau leabl-uri putin (1 sau 2) iar modelul poate sa invete continuu. Insa,pentru companii cu descriptii mai ambigue sau mai multe label-uri sistemul poate sa greseasca, insa aceste cazuri ar trebuii diminuate cu cat ml-ul invata mai mult si se imbunatateste. Ma gandesc ca in aceste cazuri s-ar putea folosii niste tehnici mai avansate de keyword matching, cum ar fii gasirea de sinonime sau cuvinte similare.
Un aspect pe care sistemul se bazeaza insa nu este sigur este ca consider ca output-ul la rule-based este un adevar concret. Adica imi asum ca label-urile gasite initial sunt corecte, fapt care poate duce la "posoning-ul" dataset-ului de invatare.
Cred ca acest model hybrid balanseaza destul de bine acuratetea si eficienta computationala, iar in timp modelul ar putea avea o acuratete din ce in ce mai buna.
Acest proiect m-a ajutat sa inteleg mai bine conceptul de *online learning* si cum solutiile simple pot duce la rezultate bune in timp.
In continuare cred ca as putea sa invat mai multe despre rule-based si cum sa eficientizez aceasta cautare, dar si sa explorez alte modele de ml si metode de a aborda aceasta provocare.
Am ales aceasta cale pentru ca mi s-a parut ca acest sistem hybrid ofera si eficienta dar si o acuratete buna, cu posibilitatea de a devenii din ce in ce mai bun. Cred ca alte cai pe care le-as fi ales ar fi fost ori full rule-based, cu mult mai multe reguli si metode de a gasii similitudini, ori full ML in care in care as incerca sa-l fac sa invete despre companii si sa aleaga el label-ul corect. Am ales sa nu merg pe alta cale deoarece am crezut ca din acest mod pot sa invat cel mai mult si pot sa l modelez cel mai mult dupa cum ma gandesc si cum mi-am imaginat.