def cross_validation(donnees0,labels0,n_split,C,sigma,spectrum=True,shuffle=False):
    n_splits= n_split
    skf = StratifiedKFold(n_splits=n_splits,random_state=42,shuffle=shuffle)

    list_accuracy = []
    list_accuracy_train = []

    for train_id_patient, val_id_patient in skf.split(donnees0,labels0):

        X_train = donnees0[train_id_patient]
        print(X_train)
        X_val = donnees0[val_id_patient]

        Y_train = labels0[train_id_patient]
        Y_val = labels0[val_id_patient]

        acc_ , mean_val_main_ , val_main_  = SVM(sigma,C,Y_train,X_train,X_val, spectrum)
        val_main_ = val_main_ * 2 - 1
        list_accuracy.append(np.mean(val_main_==Y_val))
        #print(list_accuracy)

        list_accuracy_train.append(acc_)

return list_accuracy , list_accuracy_train



best_val_acc_0 = 0
best_C_0 = -1

#C = np.array([5.e-3,6.e-3,7.e-3,8.e-3,9.e-3,1.e-2,5.e-2,0.1,0.25,0.5,0.75,1,1.5])

C = np.array([2.e-3,3.e-3,3.e-3,5.e-3,6.e-3,7.e-3,8.e-3,9.e-3,1.e-2])
#,5.e-2,0.1,0.25,0.5,0.75,1,1.5])

d_val_0 = {}
for c in C:
    d_val_0[c]=[]

d_train_0 = {}
for c in C:
    d_train_0[c]=[]



for c in tqdm(C):
    
    list_accuracy_0 , list_accuracy_train_0 = cross_validation(X_train_0,labels0,4,c,1,shuffle=True)
    acc= np.mean(list_accuracy_0)
    acc_train = np.mean(list_accuracy_train_0)
    
    d_val_0[c].append(acc)
    d_train_0[c].append(acc_train)


    if acc > best_val_acc_0 :
        best_val_acc_0 = acc
        best_C_0 = c
        print('Best acc saved {}, c : {}'.format(best_val_acc_0,best_C_0))