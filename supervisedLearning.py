import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate,learning_curve
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA
from pca import getnumComp,plotKneeComp




files = ['datasets/stacking/clusteredData.csv']




def getBestHyperparams(model,hyperparams,X,y):

    iter = 1
    
    for i in hyperparams:
         le = len(hyperparams[i])
         iter = iter*le
         
    
    iter = int(iter*0.05)

    gridSearch = RandomizedSearchCV(model,param_distributions=hyperparams,scoring="accuracy",cv=10,n_jobs=-1,n_iter=600)
    gridSearch.fit(X,y)
    return gridSearch.best_params_

def PlotClassMetrics(metrics,modelname):

    accuracy = metrics["accuracy"]
    precision = metrics["precision_macro"]
    recall = metrics["recall_macro"]
    f1 = metrics["f1_macro"]


    mean_accuracy = np.mean(accuracy)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)



    accuracy_var = np.var(accuracy)
    accuracy_std = np.std(accuracy)

    precision_var = np.var(precision)
    precision_std = np.std(precision)

    recall_var = np.var(recall)
    recall_std = np.std(recall)

    f1_var = np.var(f1)
    f1_std = np.std(f1)

    alt_metrics = {"accuracy":[mean_accuracy,accuracy_var,accuracy_std],
                   "precision_macro":[mean_precision,precision_var,precision_std],
                   "recall_macro":[mean_recall,recall_var,recall_std],
                   "f1_macro":[mean_f1,f1_var,f1_std]}

    
    columns = ["accuracy","precision_macro","recall_macro","f1_macro"]
    rows = ["split 1","split 2","split 3","split 4","split 5","split 6","split 7","split 8","split 9","split 10","media","varianza","std"]
   

    metric = {"accuracy": []
              ,"precision_macro": []
              ,"recall_macro": []
              ,"f1_macro": []
              }
    
    
    for c in columns:
         a = metrics[c]
         b = alt_metrics[c]
         metric[c] = np.concatenate((a,b),axis=None).tolist()

    

    df = pd.DataFrame(metric,index=rows)

    


    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title(modelname)
    
    ax.axis("off")
    pd.plotting.table(
            ax, df, loc="center", cellLoc="center").set_fontsize(70)


    plt.show()
    
    bar_width = 0.2
    index = 1
    plt.bar(index, mean_accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, mean_precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, mean_recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, mean_f1, bar_width, label='F1')
    
    plt.xlabel('Metriche')
    plt.ylabel('Punteggi medi')
    plt.title(f'{modelname}')
    plt.legend()

    
    plt.show()


def getModelbiasvar(model,X,y,X_test,y_test):
     
    skf = StratifiedKFold(n_splits=10)

    pred = []
    i = 0
    size = 0
    for train_index, test_index in skf.split(X, y):
        X_train= X.iloc[train_index]
        y_train = y.iloc[train_index]    
        model.fit(X_train,y_train)
        pred.append(model.predict(X_test))
    

    pred = np.array(pred)    
    y_test = np.array(y_test)

    newpred = []

    for i in range(pred.shape[0]):
        p = []
        for j in range(pred.shape[1]):
            
            p.append( 1 if pred[i][j] == y_test[j] else 0)
            

        newpred.append(p)    

    newpred = np.array(newpred)
    mean_pred = newpred.mean(axis=0)

    bias = ((1 - mean_pred)**2).mean()

    var = newpred.var(axis=0).mean()
    

    return {"biasSq": bias,"var":var}




def plotLearningCurve(model,X,y,modelname):
        train_sizes,train_scores,test_scores = learning_curve(model, X, y, cv=10, scoring='accuracy',n_jobs=-1)



        train_errors = 1 - train_scores
        test_errors = 1 - test_scores

        # Calcola la deviazione standard e la varianza degli errori su addestramento e test
        train_errors_std = np.std(train_errors, axis=1)
        test_errors_std = np.std(test_errors, axis=1)
        train_errors_var = np.var(train_errors, axis=1)
        test_errors_var = np.var(test_errors, axis=1)


        # Calcola gli errori medi su addestramento e test
        mean_train_errors = 1 - np.mean(train_scores, axis=1)
        mean_test_errors = 1 - np.mean(test_scores, axis=1)

        print("---------S C O R E S--------")
        print(f"\ntrain sizes:\n{train_sizes}\nmean train err:\n{mean_train_errors}\nmean test err:\n{mean_test_errors}")

        plt.figure(figsize=(16, 10))
        plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
        plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
        plt.title(f'Curva di apprendimento per {model}')
        plt.xlabel('Dimensione del training set')
        plt.ylabel('Errore')
        plt.title(f'{modelname}')
        plt.legend()
        plt.show()

        


def traintestModelsKfold(model,X,y):
    fitmodels = cross_validate(model,X,y,scoring=["accuracy","f1_macro",'precision_macro', 'recall_macro'],cv=10,n_jobs=-1)
    return {"f1_macro":fitmodels["test_f1_macro"],
            "accuracy":fitmodels["test_accuracy"],
            "precision_macro":fitmodels["test_precision_macro"],
            "recall_macro":fitmodels["test_recall_macro"]}

if __name__ == "__main__":
    #PREPROCESSING

    print(f"------------------PRE-PROCESSING DEI DATI----------------------\n\n")

    df = pd.read_csv(files[0])


    X = df.drop("classe",axis=1)
    y = df["classe"]


    X = pd.get_dummies(X, columns=["gender", "preferred_category"])

    print(X.columns)

    

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.50,stratify=y,random_state=42)


    Xf=X
    X_testf=X_test
    yf = y
    y_testf = y_test



    #Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_test = scaler.transform(X_test)


    #Riduzione dimensionalità

    n_feats = X.shape[1]

    print("Riduzione dimensionalità : PCA")
    print(f"Numero features prima della pca :{n_feats}")

    elbowcomp,var,splits = getnumComp(X,n_feats)

    pca_nfeats = int(np.round(elbowcomp*n_feats))

    print(f"Numero features dopo la pca :{pca_nfeats}\nExplained Variance Ratio con {pca_nfeats} features usando la pca è : {np.round(var[int(elbowcomp*10) - 1],2)}")

    pca = PCA(n_components=pca_nfeats)  
    X = pd.DataFrame(pca.fit_transform(X))


    X_test = pd.DataFrame(pca.transform(X_test))


    plotKneeComp(var,splits,elbowcomp,'explained variance','features ratio')






    print(f"------------------RICERCA IPERPARAMETRI E TRAIN-TEST DEI MODELLI----------------------\n\n")


    #KNN
    print(f"------------------K-NEAREST NEIGHBOUR----------------------\n\n")

    Knnparams = {
                "n_neighbors" : [3,5,7,13],
                "weights" : ["uniform","distance"],
                "algorithm" : ["ball_tree","kd_tree"],
                "leaf_size": [30,40,50],
                "p":[1,2,3]


                }

    


    Knnbestparams = getBestHyperparams(KNeighborsClassifier(),Knnparams,X,y)

    print(f"Iperparametri:\n{Knnbestparams}\n\n")

    knnmodel = KNeighborsClassifier(n_neighbors=Knnbestparams["n_neighbors"],weights=Knnbestparams["weights"],
                                    algorithm=Knnbestparams["algorithm"],leaf_size=Knnbestparams["leaf_size"],p=Knnbestparams["p"])



    knnmetrics = traintestModelsKfold(knnmodel,X_test,y_test)
    knnmodel.fit(X,y)

    #joblib.dump(knnmodel, 'knn.pkl')

    
    #RANDOM FOREST
    
    print(f"------------------RANDOM FOREST----------------------\n\n")


    RandomForestparams = {"criterion":["gini","entropy","log_loss"],
                        "max_depth": [5,10,20],
                        "min_samples_split":[2,5,10],
                        "min_samples_leaf": [1,5,10],
                        "max_features":["sqrt","log2",None],
                        "n_estimators":[10,20,60,80,100]
                        
                        }



    


    RandomForestbestparams = getBestHyperparams(RandomForestClassifier(random_state=42),RandomForestparams,Xf,yf)

    print(f"Iperparametri:\n{RandomForestbestparams}\n\n")

    RandomForest = RandomForestClassifier(criterion=RandomForestbestparams["criterion"],
                                        max_depth=RandomForestbestparams["max_depth"],
                                        min_samples_split=RandomForestbestparams["min_samples_split"],
                                        min_samples_leaf=RandomForestbestparams["min_samples_leaf"],
                                        max_features=RandomForestbestparams["max_features"],
                                        n_estimators=RandomForestbestparams["n_estimators"],
                                        random_state=42)



    
    RandomForestmetrics= traintestModelsKfold(RandomForest,X_testf,y_testf)
    RandomForest.fit(Xf,yf)

    #joblib.dump(RandomForest, 'RandomForest.pkl')

    
    print(f"------------------RANDOM FOREST For Stacking----------------------\n\n")


    RandomForestparams = {"criterion":["gini","entropy","log_loss"],
                        "max_depth": [5,10,20],
                        "min_samples_split":[2,5,10],
                        "min_samples_leaf": [1,5,10],
                        "max_features":["sqrt","log2",None],
                        "n_estimators":[10,20,60,80,100]
                        
                        }





    RandomForestbestparams = getBestHyperparams(RandomForestClassifier(random_state=42),RandomForestparams,X,y)

    print(f"Iperparametri:\n{RandomForestbestparams}\n\n")

    RandomForeststack = RandomForestClassifier(criterion=RandomForestbestparams["criterion"],
                                        max_depth=RandomForestbestparams["max_depth"],
                                        min_samples_split=RandomForestbestparams["min_samples_split"],
                                        min_samples_leaf=RandomForestbestparams["min_samples_leaf"],
                                        max_features=RandomForestbestparams["max_features"],
                                        n_estimators=RandomForestbestparams["n_estimators"],
                                        random_state=42)



    
    RandomForeststackmetrics= traintestModelsKfold(RandomForeststack,X_test,y_test)
    RandomForeststack.fit(X,y)


    #joblib.dump(RandomForeststack, 'RandomForeststack.pkl')
    

    #LOGREGRESS
    print(f"------------------MULTINOMIAL REGRESSION----------------------\n\n")



    logregressparams = {"solver":["lbfgs","newton-cholesky","newton-cg"],
                        "C": [0.001, 0.01, 0.1, 1, 10, 100],
                        "l1_ratio" : [0],
                        "max_iter" : [1000,5000,10000,25000,50000,100000]
                        }



    logregressbestparams = getBestHyperparams(LogisticRegression(random_state=42),logregressparams,X,y)

    print(f"Iperparametri:\n{logregressbestparams}\n\n")

    LogRegressor = LogisticRegression(C=logregressbestparams["C"],solver=logregressbestparams["solver"],
                                    l1_ratio=logregressbestparams["l1_ratio"],max_iter=logregressbestparams["max_iter"])




    


    #STACKING
    print(f"------------------MODELS STACKING----------------------\n\n")

    knnmodel = joblib.load('knn.pkl')
    RandomForeststack  = joblib.load('RandomForeststack.pkl')

    y_knn = pd.DataFrame(knnmodel.predict(X_test))
    y_knn.columns = ["classe"]
    y_rf = pd.DataFrame(RandomForeststack.predict(X_test))
    y_rf.columns = ["classe"]

    ytrain = pd.concat([y_knn,y_rf])


    Xtrain = pd.DataFrame(X_test)
    Xtrain = pd.concat([Xtrain,Xtrain])

    logregmetrics = traintestModelsKfold(LogRegressor,Xtrain,ytrain)

    #joblib.dump(LogRegressor, 'stackedmlogregress.pkl')




    #PLOTTING:LEARNING CURVES AND METRICS

    print(f"------------------LEARNING CURVES----------------------\n\n")



 
    plotLearningCurve(LogRegressor,Xtrain,ytrain,"multinomial regression")
    plotLearningCurve(knnmodel,X,y,"k-nearest neighbors")
    plotLearningCurve(RandomForest,Xf,yf,"random forest")
    plotLearningCurve(RandomForeststack,X,y,"random forest for stacking")




    print(f"------------------PERFORMANCE METRICS----------------------\n\n")


    PlotClassMetrics(logregmetrics,"multinomial regression")
    PlotClassMetrics(knnmetrics,"k-nearest neighbors")
    PlotClassMetrics(RandomForestmetrics,"random forest")
    PlotClassMetrics(RandomForeststackmetrics,"random forest for stacking")



    metone = getModelbiasvar(knnmodel,X,y,X_test,y_test)
    mettwo = getModelbiasvar(LogRegressor,X,y,X_test,y_test)
    metthree = getModelbiasvar(RandomForest,Xf,yf,X_testf,y_testf)

    print(f"KNN\n:bias:{metone["biasSq"]},var:{metone["var"]}\n\n")
    print(f"MULTINOMIAL REGRESSOR:\nbias:{mettwo["biasSq"]},var:{mettwo["var"]}\n\n")
    print(f"RANDOM FOREST:\nbias:{metthree["biasSq"]},var:{metthree["var"]}\n\n")








def profila(modello, colonne_addestramento):
    """
    Chiede interattivamente all'utente i valori delle feature.
    Effettua controlli stringenti sulle stringhe, sui limiti numerici dello spending_score (1-100)
    e sull'età (10-100).
    
    Parametri:
    - modello: Il RandomForestClassifier già addestrato (su colonne post-One-Hot).
    - colonne_addestramento: Xf.columns (le colonne DOPO il One-Hot Encoding).
    """
    print("\n=============================================")
    print("      INSERIMENTO DATI NUOVO UTENTE          ")
    print("=============================================\n")
    
    # Inizializziamo tutte le colonne a 0.0 (fondamentale per il One-Hot)
    dati_utente = {col: 0.0 for col in colonne_addestramento}
    
    # 1. GESTIONE CONTROLLATA: Gender
    opzioni_gender = ['Female', 'Male', 'Other']
    while True:
        print(f"Scegli il genere tra: {opzioni_gender}")
        scelta_gender = input("Inserisci la tua scelta: ").strip()
        scelta_matching = [opt for opt in opzioni_gender if opt.lower() == scelta_gender.lower()]
        
        if scelta_matching:
            valore_corretto = scelta_matching[0]
            dati_utente[f"gender_{valore_corretto}"] = 1.0
            break
        else:
            print(f"[!] Errore: '{scelta_gender}' non è un valore lecito. Riprova.\n")

    # 2. GESTIONE CONTROLLATA: Preferred Category
    opzioni_category = ['Clothing', 'Electronics', 'Groceries', 'Home & Garden', 'Sports']
    while True:
        print(f"\nScegli la categoria preferita tra: {opzioni_category}")
        scelta_category = input("Inserisci la tua scelta: ").strip()
        scelta_matching = [opt for opt in opzioni_category if opt.lower() == scelta_category.lower()]
        
        if scelta_matching:
            valore_corretto = scelta_matching[0]
            dati_utente[f"preferred_category_{valore_corretto}"] = 1.0
            break
        else:
            print(f"[!] Errore: '{scelta_category}' non è un valore lecito. Riprova.")

    # 3. GESTIONE FEATURE NUMERICHE (con controlli su spending_score ed age)
    colonne_one_hot = [
        'gender_Female', 'gender_Male', 'gender_Other', 
        'preferred_category_Clothing', 'preferred_category_Electronics', 
        'preferred_category_Groceries', 'preferred_category_Home & Garden', 
        'preferred_category_Sports'
    ]
    
    colonne_numeriche = [col for col in colonne_addestramento if col not in colonne_one_hot]
    
    print("\n--- Inserisci i valori numerici richiesti ---")
    for col in colonne_numeriche:
        while True:
            try:
                # Controllo specifico sull'età
                if col == 'age':
                    valore = input(f"Inserisci il valore per '{col}' (deve essere tra 10 e 100): ")
                    numero = float(valore)
                    if not (10 <= numero <= 100):
                        print("[!] Errore: l'età inserita non è accettabile. Deve essere compresa tra 10 e 100 anni.")
                        continue
                
                # Controllo specifico sullo spending score
                elif col == 'spending_score':
                    valore = input(f"Inserisci il valore per '{col}' (scala da 1 a 100): ")
                    numero = float(valore)
                    if not (1 <= numero <= 100):
                        print("[!] Errore: lo spending score deve essere compreso tra 1 e 100.")
                        continue
                
                # Input generico per le altre colonne numeriche (es. income, membership_years, ecc.)
                else:
                    valore = input(f"Inserisci il valore per '{col}': ")
                    numero = float(valore)
                
                dati_utente[col] = numero
                break
            except ValueError:
                print("[!] Input non valido. Devi inserire un numero (usa il punto per i decimali).")

    # Creazione DataFrame e Predizione
    df_input = pd.DataFrame([dati_utente], columns=colonne_addestramento)
    classe_predetta = modello.predict(df_input)[0]
    
    print("\n================ RISULTATO ================")
    if hasattr(modello, "predict_proba"):
        probabilita = modello.predict_proba(df_input)[0]
        max_prob = np.max(probabilita) * 100
        print(f"Classe Predetta: {classe_predetta} (Affidabilità: {max_prob:.2f}%)")
    else:
        print(f"Classe Predetta: {classe_predetta}")
    print("===========================================")
    
    return classe_predetta