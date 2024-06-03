from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QColorDialog, QFileDialog, QDialog, QStyle
from PyQt5.QtCore import Qt, QCoreApplication

import traceback
import configuracion

from pathlib import Path
import shutil

from paso1_preprocesar import extract_dicom_images
from paso2_entrenar_segmentador import predecir_pacientes
from paso3_predecir import predecir_cubos
from paso4_guardar import combinar_resultados, combinar_predicciones_nodulos

app = QtWidgets.QApplication([])
ventana = uic.loadUi("GUI.ui")

def actualizar_pacientes():
    print("preprocesando...")
    extract_dicom_images()
    print("preprocesado\n")

    print("prediccion 1...")
    predecir_pacientes(patients_dir=configuracion.NDSB3_EXTRACTED_IMAGE_DIR, model_path="modelos/masses_model_h0_best.h5", holdout=0, patient_predictions=[], model_type="masas")
    print("fin 1ra prediccion\n")

    print("prediccion 2...")
    holdout = 0
    for magnification in [1, 1.5, 2]:  #
        predecir_cubos("modelos/model_luna_posnegndsb_v2__fs_h" + str(holdout) + "_end.h5", True, only_patient_id=False, magnification=magnification, flip=False, train_data=False, holdout_no=holdout, ext_name="luna_posnegndsb_v2", fold_count=2)
    print("fin 1ra prediccion\n")

    print("exportando resultados...")
    combinar_predicciones_nodulos(None, train_set=False, nodule_th=0.7, extensions=["_luna_posnegndsb_v2"])
    combinar_resultados()
    print("exportado")

def agregar_paciente():    
    try:
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        ruta = QFileDialog.getExistingDirectory(ventana, "Elija la carpeta del paciente", "", options=options)
        paciente = Path(ruta).stem
        if Path("raw/"+paciente).is_dir():
            print("El paciente ya existe")
        else:
            print("copiando "+paciente+" dentro del directorio local...")
            shutil.copytree(ruta, "raw/"+paciente)
            print("copiado\n")

        print("actualizando")
        actualizar_pacientes()
        print("actualizado\n")
    except Exception as e:
        print(traceback.format_exc(e))

if __name__ == "__main__":
    ventana.btn_predecir.clicked.connect(actualizar_pacientes)
    ventana.btn_agregar.clicked.connect(agregar_paciente)

    ventana.show()
    app.exec()



