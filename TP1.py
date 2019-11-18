#!/usr/bin/env python
# coding: utf-8

# # TP1 K-NN

# ## Importation des library nécessaires

# In[1]:


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


# ## Importation du jeu de données

# In[2]:


mnist = fetch_openml('mnist_784') 


# In[3]:


#print(mnist.data[0])


# In[4]:


# Affichage des labels en cas d'apprentissage supervisé
print(mnist.target)


# In[5]:


len(mnist.data)


# In[6]:


print(mnist.data.shape)


# ## Affichage d'une image

# In[7]:


images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[5],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show() 
print(mnist.target[5]) # Affichage du label


# ## Utilisation de l'algo de K-NN

# In[8]:


from sklearn import neighbors, model_selection


# In[9]:


import numpy as np


# In[10]:


data = np.random.randint(70000,size=5000)


# In[11]:


xtrain, xtest, ytrain, ytest =  model_selection.train_test_split(mnist.data[data], mnist.target[data],train_size=0.8)


# ### Creation du classifieur

# In[29]:


clf = neighbors.KNeighborsClassifier(10)


# ### Apprentissage

# In[30]:


clf.fit(xtrain, ytrain)


# ### Test de prédiction en utilisant le quatrième élément du set de test

# In[14]:


print("valeur prédite : ",clf.predict(xtest[4].reshape(1,-1)))
print("valeur réelle : ",ytest[4])


# In[15]:


print("probabilités : ",clf.predict_proba(xtest[4].reshape(1,-1)))


# In[16]:


images = xtest.reshape((-1, 28, 28))
plt.imshow(images[4],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show() 


# ### Autre test, non valide cette fois ci

# In[17]:


print("valeur prédite : ",clf.predict(xtest[5].reshape(1,-1)))
print("valeur réelle : ",ytest[5])
print("probabilités : ",clf.predict_proba(xtest[5].reshape(1,-1)))
images = xtest.reshape((-1, 28, 28))
plt.imshow(images[5],cmap=plt.cm.gray_r,interpolation="nearest")
print("Image correspondante : ")
plt.show()


# ### Score global du classifieur

# In[31]:


clf.score(xtest, ytest)
# score sur le jeu de test
clf.score(xtrain, ytrain)


# Ce chiffre est normal, avec 5000 images seulement et des ecritures différentes entre les personnes, l'algorithme ne parvient pas à reconnaitre toutes les images

# ### Test avec des k différents

# In[19]:


plt.plot(range(2,6),[3,3,4,8],'o')
plt.xlabel("wololo")
plt.title("test")
plt.show()
from operator import add,truediv
liste = [x / 10 for x in [5,6]]
liste[1]
list(map(add,[12,23],[21,32]))


# In[20]:


liste_finale = [0]*14
for i in range (1,6):
    temp = []
    for k in range(2,16):
        clf = neighbors.KNeighborsClassifier(k,n_jobs=-1)
        clf.fit(xtrain, ytrain)
        a = clf.score(xtest, ytest)
        temp.append(a)
        print(a)
    liste_finale = list(map(add,liste_finale,temp))
liste_finale = [x/5 for x in liste_finale]
print(liste_finale)
plt.plot(range(2,16),liste_finale,'o')
plt.xlabel("k")
plt.ylabel("Score")
plt.title("Moyenne des scores en fonction de k")
plt.show()


# On remarque que les meilleurs scores sont pour k = 3 ou 5,6<br>

# In[21]:


#On cache les warnings pour ne pas avoir un affichage trop long
import warnings
warnings.filterwarnings('ignore')


# ### Changement du pourcentage train/test (et k = 7)

# In[22]:


liste_finale = [0]*14
for i in range (1,6):
    temp = []
    for k in range(30,100,5):
        xtrain, xtest, ytrain, ytest =  model_selection.train_test_split(mnist.data[data], mnist.target[data],train_size=k/100)
        clf = neighbors.KNeighborsClassifier(5,n_jobs=-1)
        clf.fit(xtrain, ytrain)
        a = clf.score(xtest, ytest)
        temp.append(a)
        print(a)
    liste_finale = list(map(add,liste_finale,temp))
liste_finale = [x/5 for x in liste_finale]
print(liste_finale)
plt.plot(range(30,100,5),liste_finale,'o')
plt.xlabel("Pourcentage d'utilisation du set pour l'apprentissage")
plt.ylabel("Score")
plt.title("Moyenne des scores en fonction du pourcentage")
plt.show()


# On remarque que plus le pourcentage est grand plus le score est bon. Cela est expliqué par le fait que plus le pourcentage est grand et plus la taille du set de training est grande. Si l'algorithme peut s'entrainer sur plus de données alors il sera meilleur

# ### Changement du type de distance
# Type 1 : Distance de Manhattan <br>
# Type 2 : (default) Distance Euclidienne

# In[23]:


for p in range(1,8):
   xtrain, xtest, ytrain, ytest =  model_selection.train_test_split(mnist.data[data], mnist.target[data],train_size=0.9)
   clf = neighbors.KNeighborsClassifier(7,p=p)
   clf.fit(xtrain, ytrain)
   print("score avec le type de distance ",p," : ",clf.score(xtest, ytest))


# In[24]:


plt.plot(range(1,8),[0.892,0.93,0.95,0.936,0.948,0.932,0.966],'o')
plt.xlabel("Type de distance utilisé")
plt.ylabel("Score")
plt.title("Evolution du score en fonction du type de distance utilisé")
plt.show()


# ### Changement de n_job, fait varier l'utilisation du nombre de processeurs

# In[25]:


import time


# In[26]:


for jobs in [-1,1]:
   temps1 = time.time()
   xtrain, xtest, ytrain, ytest =  model_selection.train_test_split(mnist.data[data], mnist.target[data],train_size=0.8)
   clf = neighbors.KNeighborsClassifier(7,n_jobs=jobs)
   clf.fit(xtrain, ytrain)
   clf.score(xtest, ytest)
   temps_final = time.time() - temps1
   print("temps d'execution avec jobs = ",jobs," : ",temps_final)


# Changer le paramètre n_jobs ne change pas le temps d'apprentissage mais change le temps pour calculer le score. Dans la documentation il est marqué : "Doesn't affect fit method" <br>
# Sur l'image si dessous on remarque un pic lors de l'utilisation de tous les processeurs puis un plateau correspondant au deuxième calcul avec un seul processeur.
# ![alt text](./capture_proco.png)

# In[32]:


# Changement taille échantillon
liste_finale = [0]*10
for i in range (1,6):
    temp = []
    for j in range(2000,22000,2000):
        ran = np.random.randint(70000,size=j)
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data[ran],mnist.target[ran],train_size=0.8)
        clf = neighbors.KNeighborsClassifier(5,n_jobs=-1)
        clf.fit(xtrain, ytrain)
        a = clf.score(xtest, ytest)
        temp.append(a)
        print(a)
    liste_finale = list(map(add,liste_finale,temp))
liste_finale = [x/5 for x in liste_finale]
print(liste_finale)
plt.plot(range(2000,22000,2000),liste_finale,'o')
plt.xlabel("Taille échantillon")
plt.ylabel("Score")
plt.title("Moyenne des scores en fonction de la taille de l'échantillon")
plt.show()


# In[ ]:





# In[ ]:




