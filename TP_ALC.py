import numpy as np
import alc

def cargarDataset(carpeta):

    X_train_cats = np.load(carpeta + "/train/cats/efficientnet_b3_embeddings.npy")
    X_train_dogs = np.load(carpeta + "/train/dogs/efficientnet_b3_embeddings.npy")

    Xt = np.concatenate((X_train_cats, X_train_dogs), axis=1)

    n_train_cats = X_train_cats.shape[1]
    n_train_dogs = X_train_dogs.shape[1]
    n_train_total = n_train_cats + n_train_dogs

    Yt = np.zeros((2, n_train_total))
    Yt[0, :n_train_cats] = 1 
    Yt[1, n_train_cats:] = 1  

    X_val_cats = np.load(carpeta + "/val/cats/efficientnet_b3_embeddings.npy")
    X_val_dogs = np.load(carpeta + "/val/dogs/efficientnet_b3_embeddings.npy")

    Xv = np.concatenate((X_val_cats, X_val_dogs), axis=1)

    n_val_cats = X_val_cats.shape[1]
    n_val_dogs = X_val_dogs.shape[1]
    n_val_total = n_val_cats + n_val_dogs

    Yv = np.zeros((2, n_val_total))
    Yv[0, :n_val_cats] = 1
    Yv[1, n_val_cats:] = 1

    return Xt, Yt, Xv, Yv

def pinvEcuacionesNormales(X, Y):
    rango = alc.rango(X)
    print("1")
    n,p = X.shape
    if rango == p and n>p:
        print("a")
        Xt = alc.traspuesta(X)
        L,Lt = alc.cholesky(alc.multiplicar_matrices(Xt,X))
        print("1.1")
        Z = np.zeros((p,n))
        for i in range(n):
            Z[:,i] = alc.sustitucionInferior(L,Xt[:,i])
        print("1.2")
        U = np.zeros((p,n))
        for i in range(n):
            U[:,i] = alc.sustitucionSuperior(Lt,Z[:,i])
        print("1.3")
        W = alc.multiplicar_matrices(Y,U)
        print("1.4")
    elif rango == n:
        if n < p:
            print("b")
            Xt = alc.traspuesta(X)
            print("2")
            L,Lt = alc.cholesky(X@Xt)
            print("2.1")
            Z = np.zeros((n,p))
            print(p)
            for i in range(p):
                Z[:,i] = alc.sustitucionInferior(L,X[:,i])
                print(i)
            print("2.2")
            Vt = np.zeros((n,p))
            for i in range(p):
                Vt[:,i] = alc.sustitucionSuperior(Lt,Z[:,i])
            print("2.3")
            V = alc.traspuesta(Vt)
            W = alc.multiplicar_matrices(Y,V)
            print("2.4")
        elif n == p:
            print("c")
            print("3")
            W = alc.multiplicar_matrices(Y,alc.inversa(X))
            print("3.1")
    return W



            
Xt,Yt,Xv,Yv = cargarDataset("template-alumnos\cats_and_dogs") 
W = pinvEcuacionesNormales(Xt,Yt)

def pinvHouseHolder(Q, R, Y):
    n,p=R.shape
    Qt = alc.traspuesta(Q)
    Vt = np.zeros((p,n))
    for i in range(n):
        Vt[:,i] = alc.sustitucionSuperior(R,Qt)
    W = alc.multiplicar_matrices(Y,alc.traspuesta(Vt))
    return W

def pinvGramSchmidt(Q, R, Y):
    n,p=R.shape
    Qt = alc.traspuesta(Q)
    Vt = np.zeros((p,n))
    for i in range(n):
        Vt[:,i] = alc.sustitucionSuperior(R,Qt)
    W = alc.multiplicar_matrices(Y,alc.traspuesta(Vt))
    return W

def esPseudoInversa(X, pX, tol=1e-8):
    #Calculo las matrices que voy a necesitar para verifiar de antemano para ahorrar recursos
    XpX = alc.multiplicar_matrices(X,pX)
    pXX = alc.multiplicar_matrices(pX,X)
    
    
    if not alc.matricesiguales(alc.multiplicar_matrices(XpX,X),X,tol):
        return False
    elif not alc.matricesiguales(alc.multiplicar_matrices(pX,XpX),pX,tol):
        return False
    elif not alc.matricesiguales(alc.traspuesta(XpX),XpX,tol):
        return False
    elif not alc.matricesiguales(alc.traspuesta(pXX),pXX):
        return False
    return True

