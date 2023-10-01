
# Application de classification d'images (Kivy + Pytorch)

Le but de ce projet est de créer une application avec la librairie **Kivy**. Des modèles CNN de classification d'images sont conçus et entrainés avec **Pytorch**. Via cette application l'utilisateur pourra choisir une architecture CNN ensuite charger une image ou dessiner un sketch afin de définir sa classe. Selont la base choisie, les resultats rendus par le modèle sont affichés ainsi que les classes associées.

Nous avons utilisé trois bases de données pour entrainer les modèles de classification : 
* **FER2013 ->**  La base de données FER2013 est composée de plus de 35 000 images de visages en niveaux de gris d'une taille de 48*48 pixels. Chaque image de la base de données FER2013 est étiquetée avec une émotion spécifique qui représente l'expression faciale du sujet. Les émotions couramment étiquetées incluent la joie, la tristesse, la peur, la colère, le dégoût, la surprise et la neutralité.

* **FashionMnist ->** La base de données Fashion MNIST est constituée d'un ensemble de 70 000 images en niveaux de gris. Chaque image a une résolution de 28 pixels de largeur et 28 pixels de hauteur. La BDD Fashion MNIST se concentre sur la classification d'articles de mode. Il y a un total de 10 classes, chacune représentant un type d'article de vêtement ou de chaussure. Les classes incluent des choses comme des t-shirts, des pantalons, des robes, des chaussures, des sacs à main, etc.

* **MNIST ->** La base de données MNIST est composée d'un ensemble de 70 000 images en niveaux de gris. Chaque image a une résolution de 28*28 pixels. La BDD MNIST est utilisée pour la classification des chiffres manuscrits de 0 à 9. Chacune des 10 classes représente l'un de ces chiffres. Chaque image est étiquetée avec la classe correspondante, c'est-à-dire le chiffre manuscrit qu'elle représente.

## 1. Prérequis
Ce projet nécessite les packages suivants pour fonctionner :
* [Python 3](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [MatPlotLib](https://matplotlib.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [Pytorch](https://pytorch.org/)
* [torchvision](https://pytorch.org/vision/stable/index.html)
* [Kivy](https://kivy.org/)

## 2. Fichiers
Ce projet est composé des fichiers suivants:
* **main.py ->** Fichier principal qui contient le code et qui permet d'executer l'application.
* Le dossier **Models ->** contient trois modèles PyTorch sous forme de notebooks au format IPYNB ainsi qu'une version pré-entraînée et enregistrée de ces modèles.
* Les dossiers **Test-Digits**, **Test-FER2013** et **Test-Fashion** contiennent des images de test, vous pouvez tester d'autres images.
* **sketch.py ->** Ce fichier vous permet de créer une interface graphique qui vous permettra de dessiner des sketchs tels que des chiffres pour les utiliser ensuite dans le modèle de classification.

## 3. Execution du projet
1. Pour mettre en marche ce projet, importez et executez les fichiers **FER2013_Pytorch.ipynb**, **mnist_fashion_pytorch.ipynb** et **mnist_pytorch2.ipynb** dans Google Colab. Cela vous permettra de l'exécuter en utilisant le processeur graphique de colab, réduisant ainsi les temps d'apprentissage. 
2. Importez les poids des modèles enregistrés et entrainés précedement au fichier principal **main.py** et executez le.
3. Une fois que l'interface Kivy est affichée, vous pouvez sélectionner le modèle de classification de votre choix ainsi qu'une image pour effectuer le test.

## 4. Résultats 
![Interface Kivy](https://github.com/YoucefAnis/Application-classification-image/blob/main/R%C3%A9sultats/MnistPrediction.png "Test 1") ![Résultat](https://github.com/YoucefAnis/Application-classification-image/blob/main/R%C3%A9sultats/Mnist%20prediction.png "Test 2")

