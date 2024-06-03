import configuracion
import herr
import sys
import os
from collections import defaultdict
import glob
import random
import pandas
import ntpath
import numpy
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import log_loss
import csv

map_labels = dict(csv.reader(open("tmp_lbl", newline="")))

def combinar_predicciones_nodulos(dirs, train_set=False, nodule_th=0.5, extension=""):
    print("Combining nodule predictions: ", "Train" if train_set else "Submission")
    if train_set:
        labels_df = pandas.read_csv("res/paso1_etiquetas.csv")
    else:
        labels_df = pandas.read_csv("res/paso2_ejemplo_resultado.csv")

    mass_df = pandas.read_csv(configuracion.BASE_DIR + "temp/predicciones_masas.csv")
    mass_df.set_index(["patient_id"], inplace=True)
    mass_df_list = list(mass_df.index)
    # meta_df = pandas.read_csv(configuracion.BASE_DIR + "patient_metadata.csv")
    # meta_df.set_index(["patient_id"], inplace=True)

    data_rows = []
    for patient_id in mass_df_list:
        # patient_id = row["patient_id"]
        # mask = herr.cargar_imagenes_paciente(patient_id, configuracion.EXTRACTED_IMAGE_DIR, "*_m.png")
        print(len(data_rows), " : ", patient_id)
        # if len(data_rows) > 19:
        #     break
        #cancer_label = row["cancer"]
        cancer_label = map_labels[patient_id] if patient_id in map_labels else str(numpy.random.rand())
        mass_pred = int(mass_df.loc[patient_id]["prediction"])
        # meta_row = meta_df.loc[patient_id]
        # z_scale = meta_row["slice_thickness"]
        # x_scale = meta_row["spacingx"]
        # vendor_low = 1 if "1.2.276.0.28.3.145667764438817.42.13928" in meta_row["instance_id"] else 0
        # vendor_high = 1 if "1.3.6.1.4.1.14519.5.2.1.3983.1600" in meta_row["instance_id"] else 0
        #         row_items = [cancer_label, 0, mass_pred, x_scale, z_scale, vendor_low, vendor_high] # mask.sum()

        row_items = [cancer_label, 0, mass_pred] # mask.sum()

        for magnification in [1, 1.5, 2]:
            pred_df_list = []
            src_dir = configuracion.NDSB3_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + extension + "/"
            pred_nodules_df = pandas.read_csv(src_dir + patient_id + ".csv")
            pred_nodules_df = pred_nodules_df[pred_nodules_df["diameter_mm"] > 0]
            pred_nodules_df = pred_nodules_df[pred_nodules_df["nodule_chance"] > nodule_th]
            pred_df_list.append(pred_nodules_df)

            pred_nodules_df = pandas.concat(pred_df_list, ignore_index=True)

            nodule_count = len(pred_nodules_df)
            nodule_max = 0
            nodule_median = 0
            nodule_chance = 0
            nodule_sum = 0
            coord_z = 0
            second_largest = 0
            nodule_wmax = 0

            count_rows = []
            coord_y = 0
            coord_x = 0

            if len(pred_nodules_df) > 0:
                max_index = pred_nodules_df["diameter_mm"].argmax()
                max_row = pred_nodules_df.loc[max_index]
                nodule_max = round(max_row["diameter_mm"], 2)
                nodule_chance = round(max_row["nodule_chance"], 2)
                nodule_median = round(pred_nodules_df["diameter_mm"].median(), 2)
                nodule_wmax = round(nodule_max * nodule_chance, 2)
                coord_z = max_row["coord_z"]
                coord_y = max_row["coord_y"]
                coord_x = max_row["coord_x"]


                rows = []
                for row_index, row in pred_nodules_df.iterrows():
                    dist = herr.calcular_distancia(max_row, row)
                    if dist > 0.2:
                        nodule_mal = row["diameter_mm"]
                        if nodule_mal > second_largest:
                            second_largest = nodule_mal
                    rows.append(row)

                count_rows = []
                for row in rows:
                    ok = True
                    for count_row in count_rows:
                        dist = herr.calcular_distancia(count_row, row)
                        if dist < 0.2:
                            ok = False
                    if ok:
                        count_rows.append(row)
            nodule_count = len(count_rows)
            row_items += [nodule_max, nodule_chance, nodule_count, nodule_median, nodule_wmax, coord_z, second_largest, coord_y, coord_x]

        row_items.append(patient_id)
        data_rows.append(row_items)

    # , "x_scale", "z_scale", "vendor_low", "vendor_high"
    columns = ["cancer_label", "mask_size", "mass"]
    for magnification in [1, 1.5, 2]:
        str_mag = str(int(magnification * 10))
        columns.append("mx_" + str_mag)
        columns.append("ch_" + str_mag)
        columns.append("cnt_" + str_mag)
        columns.append("med_" + str_mag)
        columns.append("wmx_" + str_mag)
        columns.append("crdz_" + str_mag)
        columns.append("mx2_" + str_mag)
        columns.append("crdy_" + str_mag)
        columns.append("crdx_" + str_mag)

    columns.append("patient_id")
    res_df = pandas.DataFrame(data_rows, columns=columns)

    if not os.path.exists(configuracion.BASE_DIR + "resultados/"):
        os.mkdir(configuracion.BASE_DIR + "resultados/")
    target_path = configuracion.BASE_DIR + "resultados/" "train" + extension + ".csv" if train_set else configuracion.BASE_DIR + "xgboost_trainsets/" + "res" + extension + ".csv"
    res_df.to_csv(target_path, index=False)

def entrenar_xgboost_con_nodulos_combinados(extension, fixed_holdout=False, submission=False, resultado_tiene_holdout_fijo=False):
    df_train = pandas.read_csv(configuracion.BASE_DIR + "xgboost_trainsets/" + "train" + extension + ".csv")
    if submission:
        df_resultados = pandas.read_csv(configuracion.BASE_DIR + "xgboost_trainsets/" + "res" + extension + ".csv")
        resultado_y = numpy.zeros((len(df_resultados), 1))

    if resultado_tiene_holdout_fijo:
        df_resultados = df_train[:300]
        df_train = df_train[300:]
        resultado_y = df_resultados["cancer_label"].as_matrix()
        resultado_y = resultado_y.reshape(resultado_y.shape[0], 1)

    y = df_train["cancer_label"].as_matrix()
    y = y.reshape(y.shape[0], 1)
    # print("Mean y: ", y.mean())

    cols = df_train.columns.values.tolist()
    cols.remove("cancer_label")
    cols.remove("patient_id")

    train_cols = ["mass", "mx_10", "mx_20", "mx_15", "crdz_10", "crdz_15", "crdz_20"]
    x = df_train[train_cols].as_matrix()
    if submission:
        x_submission = df_resultados[train_cols].as_matrix()

    if resultado_tiene_holdout_fijo:
        x_submission = df_resultados[train_cols].as_matrix()

    runs = 20 if fixed_holdout else 1000
    scores = []
    lista_predicciones = []
    train_preds_list = []
    holdout_preds_list = []
    for i in range(runs):
        test_size = 0.1 if submission else 0.1
        # stratify=y,
        x_train, x_holdout, y_train, y_holdout = train_test_split(x, y,  test_size=test_size)
        # print(y_holdout.mean())
        if fixed_holdout:
            x_train = x[300:]
            y_train = y[300:]
            x_holdout = x[:300]
            y_holdout = y[:300]

        seed = random.randint(0, 500) if fixed_holdout else 4242
        if True:
            clf = xgboost.XGBRegressor(max_depth=4,
                                       n_estimators=80, #55
                                       learning_rate=0.05,
                                       min_child_weight=60,
                                       nthread=8,
                                       subsample=0.95, #95
                                       colsample_bytree=0.95, # 95
                                       # subsample=1.00,
                                       # colsample_bytree=1.00,
                                       seed=seed)
            #
            clf.fit(x_train, y_train, verbose=fixed_holdout and False, eval_set=[(x_train, y_train), (x_holdout, y_holdout)], eval_metric="logloss", early_stopping_rounds=5, )
            holdout_preds = clf.predict(x_holdout)

        holdout_preds = numpy.clip(holdout_preds, 0.001, 0.999)
        # holdout_preds *= 0.93
        holdout_preds_list.append(holdout_preds)
        train_preds_list.append(holdout_preds.mean())
        score = log_loss(y_holdout, holdout_preds, normalize=True)

        print(score, "\tbest:\t", clf.best_score, "\titer\t", clf.best_iteration, "\tmean:\t", train_preds_list[-1], "\thomean:\t", y_holdout.mean())
        scores.append(score)

        if resultado_tiene_holdout_fijo:
            predicciones = clf.predict(x_submission)
            lista_predicciones.append(predicciones)

    if fixed_holdout:
        all_preds = numpy.vstack(holdout_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.001] = 0.001
        avg_preds[avg_preds > 0.999] = 0.999
        deltas = numpy.abs(avg_preds.reshape(300) - y_holdout.reshape(300))
        df_train = df_train[:300]
        df_train["deltas"] = deltas
        # df_train.to_csv("c:/tmp/deltas.csv")
        loss = log_loss(y_holdout, avg_preds)
        print("Fixed holout avg score: ", loss)
        # print("Fixed holout mean: ", y_holdout.mean())

    if resultado_tiene_holdout_fijo:
        all_preds = numpy.vstack(lista_predicciones)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.01] = 0.01
        avg_preds[avg_preds > 0.99] = 0.99
        lista_predicciones = avg_preds.tolist()
        loss = log_loss(resultado_y, lista_predicciones)
        # print("First 300 patients : ", loss)
    if resultado_tiene_holdout_fijo:
        print("Primeros 300: ", sum(scores) / len(scores), " chance promedio: ", sum(train_preds_list) / len(train_preds_list))
    else:
        print("Probabilidad promedio: ", sum(scores) / len(scores), " chance promedio: ", sum(train_preds_list) / len(train_preds_list))


def combinar_resultados(model_type=None):
    print("Combinación de resultados.. modelo: ", model_type)

    src_dir = "resultados/"
    dst_dir = "resultados/"

    resultados_df = pandas.read_csv("res/paso2_ejemplo_resultado.csv")
    resultados_df["id2"] = resultados_df["id"]
    resultados_df.set_index(["id2"], inplace=True)
    search_expr = "*.csv" if model_type is None else "*" + model_type + "*.csv"
    csvs = glob.glob(src_dir + search_expr)
    print(len(csvs), " encontrados..")
    for idx_resultado, path_resultado in enumerate(csvs):
        print(ntpath.basename(path_resultado))
        column_name = "s" + str(idx_resultado)
        resultados_df[column_name] = 0
        sub_df = pandas.read_csv(path_resultado)
        for index, row in sub_df.iterrows():
            patient_id = row["id"]
            cancer = row["cancer"]
            resultados_df.loc[patient_id, column_name] = cancer

    resultados_df["cancer"] = 0
    for i in range(len(csvs)):
        resultados_df["cancer"] += resultados_df["s" + str(i)]
    resultados_df["cancer"] /= len(csvs)

    destino = dst_dir + "resultados_finales.csv"
    resultados_df.to_csv(destino, index=False)
    #resultados_df[["id", "cancer"]].to_csv(destino, index=False)

if __name__ == "__main__":
    if True:
        for model_variant in ["_luna_posnegndsb_v2"]:
            print("Variante: ", model_variant)
            #combinar_predicciones_nodulos(None, train_set=True, nodule_th=0.7, extensions=[model_variant])
            combinar_predicciones_nodulos(None, train_set=False, nodule_th=0.7, extension=model_variant)
            #if True:
            #    entrenar_xgboost_con_nodulos_combinados(fixed_holdout=False, submission=True, resultado_tiene_holdout_fijo=False, extension=model_variant)
            #    entrenar_xgboost_con_nodulos_combinados(fixed_holdout=True, extension=model_variant)

    #combinar_resultados(level=1, model_type="luna_posnegndsb")
    #combinar_resultados(level=1, model_type="luna16_fs")
    #combinar_resultados(level=1, model_type="daniel")
    #combinar_resultados(level=2)
