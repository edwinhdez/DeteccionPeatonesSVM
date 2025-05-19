import os
import pickle
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from skimage.feature import hog
from skimage.transform import resize # Para asegurar el tamaño de las imágenes
import glob
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class ImageLoader:
    def __init__(self, 
        positive_path=r"C:\Git\TecMonterrrey\NavegacionAutonoma\navegacionautonoma_pip_p310\navegacionautonoma_py310\DeteccionPeatonesSVM\data\archive (3)\Dataset\Positive\*.jpg", 
        negative_path=r"C:\Git\TecMonterrrey\NavegacionAutonoma\navegacionautonoma_pip_p310\navegacionautonoma_py310\DeteccionPeatonesSVM\data\archive (3)\Dataset\Negative\*.jpg"
        ):

        """
        Inicializa la clase con las rutas de las imágenes positivas y negativas.
        """
        self.positive_path = positive_path
        self.negative_path = negative_path

    def load_images(self):
        """
        Carga las imágenes de las rutas especificadas.
        :return: Diccionario con listas de rutas de imágenes positivas y negativas.
        """
        pedestrian = glob.glob(self.positive_path)
        no_pedestrian = glob.glob(self.negative_path)
        return {
            "pedestrian": pedestrian,
            "no_pedestrian": no_pedestrian
        }
    
    def show_pedestrian_image(self, index=0):
        """
        Muestra una imagen de peatón en el índice especificado.
        :param index: Índice de la imagen a mostrar.
        """
        images = self.load_images()
        pedestrian = images["pedestrian"]
        if index < 0 or index >= len(pedestrian):
            print("Índice fuera de rango.")
            return
        img_color = cv2.imread(pedestrian[index])
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib
        plt.imshow(img_color)
        plt.axis("off")  # Ocultar los ejes
        plt.show()

        return img_color

class HOGProcessor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Inicializa la clase con los parámetros de HOG.
        :param orientations: Número de orientaciones para el histograma.
        :param pixels_per_cell: Tamaño de cada celda en píxeles.
        :param cells_per_block: Tamaño del bloque en celdas.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def apply_hog(self, image_path=None, image=None, visualize=False):
        """
        Aplica HOG a una imagen.
        :param image_path: Ruta de la imagen a procesar (opcional).
        :param image: Imagen cargada directamente (opcional).
        :param visualize: Si es True, devuelve también la visualización de HOG.
        :return: Características HOG y, opcionalmente, la visualización.
        """
        if image is None:
            if image_path is None:
                raise ValueError("Debe proporcionar una ruta de imagen o una imagen cargada.")
            # Leer la imagen en escala de grises desde la ruta
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen desde la ruta: {image_path}")
        else:
            # Convertir la imagen cargada a escala de grises si no lo está
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Aplicar HOG
        if visualize:
            features, hog_image = hog(image, 
                                      orientations=self.orientations, 
                                      pixels_per_cell=self.pixels_per_cell, 
                                      cells_per_block=self.cells_per_block, 
                                      visualize=visualize, 
                                      channel_axis=None)
            return features, hog_image
        else:
            features = hog(image, 
                           orientations=self.orientations, 
                           pixels_per_cell=self.pixels_per_cell, 
                           cells_per_block=self.cells_per_block, 
                           visualize=visualize, 
                           channel_axis=None)
            return features

    def show_hog(self, image_path=None, image=None):
        """
        Muestra la imagen original y su visualización HOG.
        :param image_path: Ruta de la imagen a procesar (opcional).
        :param image: Imagen cargada directamente (opcional).
        """
        features, hog_image = self.apply_hog(image_path=image_path, image=image, visualize=True)
        
        if image is None:
            # Leer la imagen original desde la ruta
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertir a RGB para matplotlib
        else:
            # Convertir la imagen cargada a RGB si no lo está
            if len(image.shape) == 3:
                original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                original_image = image  # Si ya está en escala de grises, no se convierte

        # Mostrar la imagen original y la visualización HOG
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Imagen Original")
        plt.imshow(original_image, cmap="gray" if len(original_image.shape) == 2 else None)
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title("Visualización HOG")
        plt.imshow(hog_image, cmap="gray")
        plt.axis("off")
        
        plt.show()

    def process_images(self, image_paths):
        """
        Procesa una lista de rutas de imágenes y extrae las características HOG.
        :param image_paths: Lista de rutas de imágenes.
        :return: Lista de características HOG extraídas.
        """
        hog_accum = []
        for image_path in image_paths:
            # Leer la imagen en color
            img_color = cv2.imread(image_path)
            if img_color is None:
                print(f"No se pudo cargar la imagen: {image_path}")
                continue
            
            # Convertir a escala de grises
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Redimensionar la imagen para asegurar un tamaño uniforme
            img_gray = resize(img_gray, (100, 100))
            
            # Extraer características HOG
            features = self.apply_hog(image=img_gray, visualize=False)  # Ajuste aquí: solo se recibe `features`
            
            # Agregar las características a la lista acumulada
            hog_accum.append(features)
        
        return hog_accum

class ModelTrainer:
    def __init__(self, columnasTransformer):
        """
        Inicializa la clase con el ColumnTransformer para preprocesar los datos.
        :param columnasTransformer: ColumnTransformer para las transformaciones.
        """
        self.columnasTransformer = columnasTransformer 
    

    def plot_roc_curve(self, model, Xtrain, ytrain, Xval, yval):
        """
        Genera la curva ROC para el modelo entrenado en los conjuntos de entrenamiento y validación.
        :param model: Modelo entrenado.
        :param Xtrain: Conjunto de características de entrenamiento.
        :param ytrain: Etiquetas de entrenamiento.
        :param Xval: Conjunto de características de validación.
        :param yval: Etiquetas de validación.
        """
        # Calcular las probabilidades de la clase positiva para los conjuntos de entrenamiento y validación
        train_probs = model.predict_proba(Xtrain)[:, 1]
        val_probs = model.predict_proba(Xval)[:, 1]

        # Calcular la curva ROC para los conjuntos de entrenamiento y validación
        train_fpr, train_tpr, _ = roc_curve(ytrain, train_probs)
        val_fpr, val_tpr, _ = roc_curve(yval, val_probs)

        # Calcular el área bajo la curva (AUC) para los conjuntos de entrenamiento y validación
        train_auc = auc(train_fpr, train_tpr)
        val_auc = auc(val_fpr, val_tpr)

        # Generar el gráfico
        plt.figure(figsize=(10, 7))
        plt.plot(train_fpr, train_tpr, label='Training AUC: {:.2f}'.format(train_auc))
        plt.plot(val_fpr, val_tpr, label='Validation AUC: {:.2f}'.format(val_auc))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def random_search_svc(self, Xtrainval, ytrainval, X_test, y_test, n_iter=3, cv=3, random_state=42, best_params=None, param_grid_svc=None):
        """
        Ejecuta RandomizedSearchCV para optimizar los hiperparámetros de un modelo SVC o entrena directamente con los mejores parámetros.
        :param Xtrainval: Conjunto de características de entrenamiento y validación.
        :param ytrainval: Etiquetas de entrenamiento y validación.
        :param X_test: Conjunto de características de prueba.
        :param y_test: Etiquetas de prueba.
        :param n_iter: Número de iteraciones para RandomizedSearchCV.
        :param cv: Número de particiones para validación cruzada.
        :param random_state: Semilla para reproducibilidad.
        :param param_grid_svc: Espacio de búsqueda de hiperparámetros.
        :param best_params: Diccionario con los mejores parámetros para entrenar directamente.
        :return: Modelo entrenado y los parámetros utilizados.
        """
        if best_params:
            # Si se proporcionan los mejores parámetros, entrena directamente el modelo
            print("Entrenando el modelo directamente con los mejores parámetros proporcionados...")
            svc_pipeline = Pipeline(steps=[
                ('preprocessor', self.columnasTransformer),
                ('classifier', SVC(**best_params))
            ])
            svc_pipeline.fit(Xtrainval, ytrainval)
            y_pred = svc_pipeline.predict(X_test)
            report = classification_report(y_test, y_pred)
            print("Reporte de clasificación:\n", report)
            return svc_pipeline, best_params
        else:
            # Si no se proporcionan los mejores parámetros, realiza RandomizedSearchCV
            print("Ejecutando RandomizedSearchCV para encontrar los mejores parámetros...")
            if param_grid_svc is None:
                param_grid_svc = {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'classifier__gamma': ['scale', 'auto', 0.1, 1],
                    'classifier__degree': [2, 3, 4],
                    'classifier__coef0': [0.0, 0.1, 0.5],
                    'classifier__shrinking': [True, False],
                    'classifier__probability': [True, False],
                    'classifier__class_weight': [None, 'balanced'],
                    'classifier__random_state': [random_state]
                }
    
            svc_pipeline = Pipeline(steps=[
                ('preprocessor', self.columnasTransformer),
                ('classifier', SVC(probability=True))
            ])
    
            mismetricas = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    
            random_search = RandomizedSearchCV(
                estimator=svc_pipeline,
                param_distributions=param_grid_svc,
                n_iter=n_iter,
                cv=cv,
                scoring=mismetricas,
                refit='roc_auc',
                random_state=random_state,
                verbose=2
            )
    
            random_search.fit(Xtrainval, ytrainval)
    
            best_estimator = random_search.best_estimator_
            y_pred = best_estimator.predict(X_test)
            report = classification_report(y_test, y_pred)
            score = random_search.best_score_
    
            print("Mejores parámetros encontrados de SVC:")
            print(random_search.best_params_)
    
            return random_search.best_estimator_, random_search.best_params_

    
class DecisionBoundaryPlotter:
    def __init__(self, model, X, y, pca):
        """
        Inicializa la clase con el modelo, los datos y las etiquetas.
        :param model: Modelo SVC entrenado.
        :param X: Características de los datos.
        :param y: Etiquetas de los datos.
        """
        self.model = model
        self.X = X
        self.y = y
        self.pca = pca

    def reduce_dimensions(self):
        """
        Reduce las dimensiones de los datos a 2D si tienen más de 2 características.
        :return: Datos reducidos a 2D.
        """
        if self.X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(self.X)
            print("Datos reducidos a 2D usando PCA.")
            return X_reduced
        return self.X

    def train_model_for_2d(self, X_reduced, y):
        """
        Entrena un modelo SVC con los datos reducidos a 2 dimensiones.
        :param X_reduced: Datos reducidos a 2 dimensiones.
        :param y: Etiquetas de los datos.
        :return: Modelo SVC entrenado.
        """
        svc_2d = SVC(kernel='rbf', gamma='scale', C=1)
        svc_2d.fit(X_reduced, y)
        print("Modelo SVC entrenado con datos reducidos a 2D.")
        return svc_2d

    def plot_decision_boundary(self):
        """
        Grafica la frontera de decisión del modelo junto con los puntos de datos.
        """
        # Reducir las dimensiones si es necesario
        X_reduced = self.reduce_dimensions()
    
        # Entrenar un modelo SVC con los datos reducidos
        svc_2d = self.train_model_for_2d(X_reduced, self.y)
    
        # Crear una malla para graficar las fronteras de decisión
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
    
        # Predecir las etiquetas para cada punto en la malla
        Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        # Graficar las fronteras de decisión
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
        # Graficar los puntos de datos
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.coolwarm)
    
        # Agregar nombres de las clases en las regiones
        class_labels = {0: "No Pedestrian", 1: "Pedestrian"}  # Etiquetas de las clases
        for class_value, class_name in class_labels.items():
            # Encontrar el centro aproximado de la región de la clase
            mask = Z == class_value
            if np.any(mask):
                x_center = xx[mask].mean()
                y_center = yy[mask].mean()
                plt.text(x_center, y_center, class_name, color="white", fontsize=12, ha="center", va="center",
                         bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
    
        # Configurar el título y etiquetas de los ejes
        plt.title("Fronteras de decisión del modelo SVC (2D)")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.show()

    def plot_decision_boundary_with_model(self):
        """
        Grafica la frontera de decisión del modelo junto con los puntos de datos,
        mostrando las clases correspondientes a cada componente.
        """
        # Crear una malla para graficar las fronteras de decisión
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                np.arange(y_min, y_max, 0.01))

        # Predecir las etiquetas para cada punto en la malla usando el modelo entrenado
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Graficar las fronteras de decisión
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

        # Graficar los puntos de datos con colores según las clases
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c="blue", label="Peatón", edgecolors='k', alpha=0.7)
        plt.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], c="red", label="No Peatón", edgecolors='k', alpha=0.7)

        # Configurar el título y etiquetas de los ejes
        plt.title("Fronteras de decisión del modelo SVC (2D)")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.legend(loc="upper right")
        plt.show()


    def plot_decision_boundary_with_random_image(self, image_path, hog_processor):
        """
        Grafica la frontera de decisión del modelo junto con los puntos de datos
        y un punto verde que representa una imagen aleatoria.
        :param image_path: Ruta de la imagen aleatoria.
        :param hog_processor: Objeto HOGProcessor para procesar la imagen.
        """
        # Crear una malla para graficar las fronteras de decisión
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # Predecir las etiquetas para cada punto en la malla usando el modelo entrenado
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Graficar las fronteras de decisión
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

        # Graficar los puntos de datos
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.coolwarm)

        # Procesar la imagen aleatoria
        img_color = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_gray = resize(img_gray, (100, 100))
        hog_features = hog_processor.apply_hog(image=img_gray, visualize=False)
        hog_features = np.array(hog_features).reshape(1, -1)

        # Reducir la imagen a 2D con PCA
        img_pca = self.pca.transform(hog_features)

        # Predecir la clase de la imagen
        predicted_class = self.model.predict(img_pca)[0]
        print(f"Clase predicha para la imagen: {'Peatón' if predicted_class == 1 else 'No Peatón'}")

        # Graficar el punto verde en el gráfico
        plt.scatter(img_pca[0, 0], img_pca[0, 1], color='green', s=100, label='Imagen Aleatoria')
        plt.legend()

        # Configurar el título y etiquetas de los ejes
        plt.title("Fronteras de decisión del modelo SVC (2D)")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.show()

def show_image(image_path, title="Imagen"):
    """
    Muestra una imagen dada su ruta.
    :param image_path: Ruta de la imagen a mostrar.
    :param title: Título del gráfico.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"No se pudo cargar la imagen desde la ruta: {image_path}")
        return
    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(img_color_rgb)
    plt.axis("off")
    plt.show()
    
    

# Main code
if __name__ == "__main__":

    # 1. Cargar rutas de imágenes
    loader = ImageLoader()
    images = loader.load_images()
    paths_pedestrian = images["pedestrian"]
    paths_nopedestrian = images["no_pedestrian"]

    # 2. Crear lista de rutas y etiquetas
    X_paths = paths_pedestrian + paths_nopedestrian
    y_labels = [1] * len(paths_pedestrian) + [0] * len(paths_nopedestrian)

    # 3. Separar train/test/val sobre las rutas (no sobre HOG aún)
    X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(X_paths, y_labels, test_size=0.3, stratify=y_labels, random_state=42)
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Train: {len(X_train_paths)}, Val: {len(X_val_paths)}, Test: {len(X_test_paths)}")

    # 4. Aplicar HOG a cada conjunto (sin fuga)
    hog_processor = HOGProcessor()

    X_train = hog_processor.process_images(X_train_paths)
    X_val = hog_processor.process_images(X_val_paths)
    X_test = hog_processor.process_images(X_test_paths)

    X_train = np.vstack(X_train).astype(np.float64)
    X_val = np.vstack(X_val).astype(np.float64)
    X_test = np.vstack(X_test).astype(np.float64)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # 5. Reducir a 2D con PCA
    model_dir = r"C:\Git\TecMonterrrey\NavegacionAutonoma\navegacionautonoma_pip_p310\navegacionautonoma_py310\DeteccionPeatonesSVM\models"  # Usa tu ruta real
    pca = PCA(n_components=2)
    # Ajustar y transformar el conjunto de entrenamiento
    X_train_pca = pca.fit_transform(X_train)  
    # Transformar los conjuntos de validación y prueba
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # 6. Entrenar modelo con datos reducidos
    trainer = ModelTrainer(columnasTransformer=None)  # No se necesita ColumnTransformer para datos reducidos

    best_param_grid_svc = {
        'shrinking': True, 
        'random_state': 42, 
        'probability': True, 
        'kernel': 'rbf', 
        'gamma': 'scale', 
        'degree': 2, 
        'coef0': 0.5, 
        'class_weight': 'balanced', 
        'C': 1
    }

    modelo_pickle_path = os.path.join(model_dir, "modelo_entrenado_pca.pkl")

    if os.path.exists(modelo_pickle_path):
        with open(modelo_pickle_path, "rb") as f:
            best_model = pickle.load(f)
        print("Modelo cargado.")
    else:
        best_model, used_params = trainer.random_search_svc(X_train_pca, y_train, X_test_pca, y_test, best_params=best_param_grid_svc)
        with open(modelo_pickle_path, "wb") as f:
            pickle.dump(best_model, f)
        print("Modelo entrenado y guardado.")

    # 7. Visualizaciones
    plotter = DecisionBoundaryPlotter(best_model, X_train_pca, y_train, pca)
    plotter.plot_decision_boundary_with_model()
    trainer.plot_roc_curve(best_model, X_train_pca, y_train, X_test_pca, y_test)

    # Seleccionar una imagen aleatoria del dataset
    print("Seleccionando una imagen aleatoria del dataset.")
    random_image_path = random.choice(X_paths)
    # Mostrar la imagen seleccionada
    show_image(random_image_path, title="Imagen Aleatoria Seleccionada")
    plotter.plot_decision_boundary_with_random_image(random_image_path, hog_processor)