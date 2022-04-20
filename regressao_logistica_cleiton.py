import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

####################################################################################################

def preparaBaseStroke (base):
    base = pd.read_csv(base)

    previsores = base.iloc[:, 0:11].values

    classe = base.iloc[:, 11].values

    label_encoder = LabelEncoder()

    previsores[:,1]  = label_encoder.fit_transform(previsores[:,1])
    previsores[:,2]  = label_encoder.fit_transform(previsores[:,2])
    previsores[:,5]  = label_encoder.fit_transform(previsores[:,5])
    previsores[:,6]  = label_encoder.fit_transform(previsores[:,6])
    previsores[:,7]  = label_encoder.fit_transform(previsores[:,7])
    previsores[:,8]  = label_encoder.fit_transform(previsores[:,8])
    previsores[:,9]  = label_encoder.fit_transform(previsores[:,9])
    previsores[:,10] = label_encoder.fit_transform(previsores[:,10])

    X_train, X_test, y_train, y_test = train_test_split (previsores, classe, test_size=0.25, random_state = 42)
    return X_train, X_test, y_train, y_test

def preparaBaseHeart (base):
    base = pd.read_csv(base)

    previsores = base.iloc[:, 1:18].values

    classe = base.iloc[:, 0].values

    label_encoder = LabelEncoder()

    previsores[:,1]  = label_encoder.fit_transform(previsores[:,1])
    previsores[:,2]  = label_encoder.fit_transform(previsores[:,2])
    previsores[:,3]  = label_encoder.fit_transform(previsores[:,3])
    previsores[:,6]  = label_encoder.fit_transform(previsores[:,6])
    previsores[:,7]  = label_encoder.fit_transform(previsores[:,7])
    previsores[:,8]  = label_encoder.fit_transform(previsores[:,8])
    previsores[:,9]  = label_encoder.fit_transform(previsores[:,9])
    previsores[:,10] = label_encoder.fit_transform(previsores[:,10])
    previsores[:,11] = label_encoder.fit_transform(previsores[:,11])
    previsores[:,12] = label_encoder.fit_transform(previsores[:,12])
    previsores[:,14] = label_encoder.fit_transform(previsores[:,14])
    previsores[:,15] = label_encoder.fit_transform(previsores[:,15])
    previsores[:,16] = label_encoder.fit_transform(previsores[:,16])

    X_train, X_test, y_train, y_test = train_test_split (previsores, classe, test_size = 0.25, random_state = 42)
    return X_train, X_test, y_train, y_test

def naiveBayes (X_train, X_test, y_train, y_test):
    naive = GaussianNB()
    naive.fit (X_train, y_train)
    y_pred = naive.predict (X_test)

    print ("Naive Bayes")
    print (confusion_matrix (y_test, y_pred))
    print (accuracy_score (y_test, y_pred))

def arvoreDecisao (X_train, X_test, y_train, y_test):
    arvore = DecisionTreeClassifier (criterion = 'entropy')
    arvore.fit (X_train, y_train)
    y_pred = arvore.predict (X_test)

    print ("Árvore de Decisão")
    print (confusion_matrix (y_test, y_pred))
    print (accuracy_score (y_test,y_pred))

def florestaRandomica (X_train, X_test, y_train, y_test):
    floresta = RandomForestClassifier (n_estimators = 400, max_features = 8, criterion = 'gini', random_state = 1)
    floresta.fit (X_train, y_train)
    y_pred = floresta.predict (X_test)

    print ("Floresta Randômica")
    print (confusion_matrix (y_test, y_pred))
    print (accuracy_score (y_test, y_pred)) 

def regressaoLogistica (X_train, X_test, y_train, y_test):
    regressao = LogisticRegression (random_state = 1)
    regressao.fit (X_train, y_train)
    y_pred = regressao.predict (X_test)

    print ("Regressão Logística")
    print (confusion_matrix (y_test, y_pred))
    print (accuracy_score (y_test, y_pred)) 

def main():
    base_stroke = 'https://raw.githubusercontent.com/ccopceski/machine-learn/main/healthcare-dataset-stroke.csv'
    base_heart  = 'https://raw.githubusercontent.com/ccopceski/machine-learn/main/heart_2020_cleaned.csv'

    menu = int(input ("Escolha a base de dados:\n1 - Derrames\n2 - Infartos"))

    if (menu == 1):
        X_train, X_test, y_train, y_test = preparaBaseStroke (base_stroke)

        print ("\nInfartos:\n")
        naiveBayes (X_train, X_test, y_train, y_test)

        arvoreDecisao (X_train, X_test, y_train, y_test)

        florestaRandomica (X_train, X_test, y_train, y_test)

        regressaoLogistica (X_train, X_test, y_train, y_test)

    else:
        if (menu == 2):
            X_train, X_test, y_train, y_test = preparaBaseHeart (base_heart)
            
            print ("\nDerrames:\n")
            naiveBayes (X_train, X_test, y_train, y_test)

            arvoreDecisao (X_train, X_test, y_train, y_test)

            florestaRandomica (X_train, X_test, y_train, y_test)

            regressaoLogistica (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()