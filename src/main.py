import os
import pickle
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

    def random_search_svc(self, Xtrainval, ytrainval, X_test, y_test, n_iter=2, cv=2, random_state=42, best_params=None, param_grid_svc=None):
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
    def __init__(self, model, X, y):
        """
        Inicializa la clase con el modelo, los datos y las etiquetas.
        :param model: Modelo SVC entrenado.
        :param X: Características de los datos.
        :param y: Etiquetas de los datos.
        """
        self.model = model
        self.X = X
        self.y = y

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
    
    

# Main code
if __name__ == "__main__":

    # Cargar y mostrar imágenes de peatones
    loader = ImageLoader()
    images = loader.load_images()
    #image_sample = loader.show_pedestrian_image(index=0)  # Cambia el índice según sea necesario
    pedestrian_image = cv2.imread(images["pedestrian"][0])  # Cargar una imagen directamente
    
    
    # Aplicar HOG a una imagen de ejemplo
    hog_processor = HOGProcessor()
    # Usar una imagen cargada directamente
    hog_processor.show_hog(image=pedestrian_image)  
    # Usar una ruta de imagen
    hog_processor.show_hog(image_path=images["pedestrian"][0])

    # Directorio para guardar los archivos
    model_dir = r"C:\Git\TecMonterrrey\NavegacionAutonoma\navegacionautonoma_pip_p310\navegacionautonoma_py310\DeteccionPeatonesSVM\models"
    os.makedirs(model_dir, exist_ok=True)  # Crear el directorio si no existe

    # Rutas completas para los archivos
    pedestrian_hog_file = os.path.join(model_dir, "pedestrian_hog_accum.pkl")
    no_pedestrian_hog_file = os.path.join(model_dir, "no_pedestrian_hog_accum.pkl")

     # Verificar si ya existe el archivo de características HOG para peatones
    try:
        with open(pedestrian_hog_file, "rb") as f:
            pedestrian_hog_accum = pickle.load(f)
            print("Características HOG de peatones cargadas desde el archivo.")
    except FileNotFoundError:
        # Si no existe, procesar las imágenes y guardar los resultados
        pedestrian_hog_accum = hog_processor.process_images(images["pedestrian"])
        print(f"Características HOG de peatones procesadas: {len(pedestrian_hog_accum)} imágenes.")
        with open(pedestrian_hog_file, "wb") as f:
            pickle.dump(pedestrian_hog_accum, f)
            print(f"Características HOG de peatones guardadas en {pedestrian_hog_file}.")

    # Verificar si ya existe el archivo de características HOG para no peatones
    try:
        with open(no_pedestrian_hog_file, "rb") as f:
            no_pedestrian_hog_accum = pickle.load(f)
            print("Características HOG de no peatones cargadas desde el archivo.")
    except FileNotFoundError:
        # Si no existe, procesar las imágenes y guardar los resultados
        no_pedestrian_hog_accum = hog_processor.process_images(images["no_pedestrian"])
        print(f"Características HOG de no peatones procesadas: {len(no_pedestrian_hog_accum)} imágenes.")
        with open(no_pedestrian_hog_file, "wb") as f:
            pickle.dump(no_pedestrian_hog_accum, f)
            print(f"Características HOG de no peatones guardadas en {no_pedestrian_hog_file}.")

    # Convertir las listas acumuladas en matrices numpy
    # y crear etiquetas para cada clase
    X_pedestrian = np.vstack(pedestrian_hog_accum).astype(np.float64)
    y_pedestrian = np.ones(len(X_pedestrian))

    # Convertir las listas acumuladas en matrices numpy
    # y crear etiquetas para cada clase
    X_nopedestrian = np.vstack(no_pedestrian_hog_accum).astype(np.float64)
    y_nopedestrian = np.zeros(len(X_nopedestrian))
    
    # Concatenar las características y etiquetas
    # para crear el conjunto de datos final
    X = np.vstack((X_pedestrian,X_nopedestrian))
    y = np.hstack((y_pedestrian,y_nopedestrian))

    # Primero, dividimos los datos en entrenamiento (70%) y un conjunto temporal (30%) con estratificación
    Xtrain, Xtemp, ytrain, ytemp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Luego, dividimos el conjunto temporal en validación y prueba (ambos 15% del total) con estratificación
    Xval, Xtest, yval, ytest = train_test_split(Xtemp, ytemp, test_size=0.5, stratify=ytemp, random_state=42)

    # Mostremos las dimensiones de la partición generada:
    print(Xtrain.shape, ytrain.shape)
    print(Xval.shape, yval.shape)
    print(Xtest.shape, ytest.shape)


    # pipeline para las características numéricas (HOG)
    hog_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputar valores faltantes (si los hay)
        ('scaler', MinMaxScaler())  # Escalar las características entre 0 y 1
    ])

    # Dado que todas las características HOG son numéricas, aplicamos el pipeline a todas las columnas
    columnasTransformer = ColumnTransformer(
        transformers=[
            ('hog_features', hog_pipe, slice(0, Xtrain.shape[1]))  # Aplica el pipeline a todas las columnas de HOG
        ],
        remainder='passthrough'  # No hay columnas adicionales, pero esto asegura que no se pierda nada
    )

    # Aplicar las transformaciones a los conjuntos de datos
    Xtrain_transformed = columnasTransformer.fit_transform(Xtrain)
    Xval_transformed = columnasTransformer.transform(Xval)
    Xtest_transformed = columnasTransformer.transform(Xtest)

    # Mostremos las dimensiones después de la transformación
    print("Dimensiones después de la transformación:")
    print("Xtrain:", Xtrain_transformed.shape)
    print("Xval:", Xval_transformed.shape)
    print("Xtest:", Xtest_transformed.shape)

    # Convertir Xtrain y Xval a DataFrame antes de concatenarlos
    Xtrain_df = pd.DataFrame(Xtrain, columns=[f"feature_{i}" for i in range(Xtrain.shape[1])])
    Xval_df = pd.DataFrame(Xval, columns=[f"feature_{i}" for i in range(Xval.shape[1])])

    # Convertir ytrain y yval a Series antes de concatenarlos
    ytrain_series = pd.Series(ytrain, name="label")
    yval_series = pd.Series(yval, name="label")

    # Entrenar con los parámetros específicos para SVC   
    best_param_grid_svc = {
        'shrinking': True, 
        'random_state': 42, 
        'probability': True, 
        'kernel': 'rbf', 
        'gamma': 'scale', 
        'degree': 2, 
        'coef0': 0.5, 
        'class_weight': 'balanced', 
        'C': 100
    }

    # Concatenar los conjuntos de entrenamiento y validación
    Xtrainval = pd.concat([Xtrain_df, Xval_df], axis=0)
    ytrainval = pd.concat([ytrain_series, yval_series], axis=0)

    # Mostrar las dimensiones después de la concatenación
    print("Dimensiones después de la concatenación:")
    print(Xtrainval.shape, ytrainval.shape)

    

    # Ruta del archivo pickle para guardar/cargar el modelo entrenado
    modelo_pickle_path = os.path.join(model_dir, "modelo_entrenado.pkl")

    # Crear una instancia de ModelTrainer
    trainer = ModelTrainer(columnasTransformer)

    # Verificar si el archivo pickle del modelo ya existe
    if os.path.exists(modelo_pickle_path):
        # Cargar el modelo entrenado desde el archivo pickle
        with open(modelo_pickle_path, "rb") as f:
            best_model = pickle.load(f)
        print(f"Modelo cargado desde el archivo pickle: {modelo_pickle_path}")
            
    else:
        # Si no existe, entrenar el modelo y guardarlo
        print("El archivo pickle no existe. Entrenando el modelo...")

        # Ejecutar RandomizedSearchCV
        best_model, used_params = trainer.random_search_svc(Xtrainval, ytrainval, Xtest, ytest, best_params=best_param_grid_svc)
        print("Mejor modelo:", best_model)
        print("Mejores parámetros:", used_params)

        # Guardar el modelo entrenado en un archivo pickle
        with open(modelo_pickle_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Modelo entrenado y guardado en: {modelo_pickle_path}")

    # Mostrar el modelo entrenado o cargado
    print("Modelo entrenado o cargado:")
    print(best_model)

     # Crear una instancia de la clase DecisionBoundaryPlotter
    plotter = DecisionBoundaryPlotter(best_model, Xtrainval.values, ytrainval.values)

    # Graficar la frontera de decisión
    plotter.plot_decision_boundary()

    # Generar la curva ROC
    trainer.plot_roc_curve(best_model, Xtrainval, ytrainval, Xtest, ytest)

