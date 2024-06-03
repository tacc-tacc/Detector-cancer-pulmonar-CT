import configuracion
import glob
import datetime
import os
import sys
import numpy
import cv2
from collections import defaultdict
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import math
import pandas

def calculo_dado(label_img, pred_img, p_threshold=0.5):
    p = pred_img.astype(numpy.float)
    l = label_img.astype(numpy.float)
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = numpy.clip(p, 0, 1.0)
    l = numpy.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = numpy.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val


class Stopwatch(object):

    def start(self):
        self.start_time = Stopwatch.get_time()

    def get_elapsed_time(self):
        current_time = Stopwatch.get_time()
        res = current_time - self.start_time
        return res

    def get_elapsed_seconds(self):
        elapsed_time = self.get_elapsed_time()
        res = elapsed_time.total_seconds()
        return res

    @staticmethod
    def get_time():
        res = datetime.datetime.now()
        return res

    @staticmethod
    def start_new():
        res = Stopwatch()
        res.start()
        return res


def cargar_imagenes_paciente(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    if base_dir == None:
        base_dir = configuracion.LUNA_16_TRAIN_DIR
    src_dir = base_dir + patient_id + "/"
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = numpy.vstack(images)
    return res


def guardar_imagen_cubo(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def cargar_imagen_cubo(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # assert rows * size == cube_img.shape[0]
    # assert cols * size == cube_img.shape[1]
    res = numpy.zeros((rows * cols, size, size))

    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]

    return res


def normalizar_imagen_uint8(img):
    img = img.astype(numpy.float)
    min = img.min()
    max = img.max()
    img -= min
    img /= max - min
    img *= 255
    res = img.astype(numpy.uint8)
    return res


def normalizar_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def reescalar_imagenes_paciente(images_zyx, org_spacing_xyz, TAM_VOXEL_MM, is_mask_image=False, verbose=False):
    if verbose:
        print("Espaciado: ", org_spacing_xyz)
        print("Forma: ", images_zyx.shape)

    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(TAM_VOXEL_MM)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    resize_x = float(org_spacing_xyz[0]) / float(TAM_VOXEL_MM)
    resize_y = float(org_spacing_xyz[1]) / float(TAM_VOXEL_MM)

    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Nueva  forma: ", res.shape)
    return res


def reescalar_imagenes_paciente2(images_zyx, target_shape, verbose=False):
    if verbose:
        print("Objetivo: ", target_shape)
        print("Forma: ", images_zyx.shape)

    resize_x = 1.0
    interpolation = cv2.INTER_NEAREST if False else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=(target_shape[1], target_shape[0]), interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Nueva forma: ", res.shape)
    return res


def imp_variables_globales(globs, names):
    # globs = globals()
    print("-- GLOBALES --")
    for key in globs.keys():
        if key in names:
            print(key, ": ", globs[key])
    print("")


PRINT_TAB_MAP = defaultdict(lambda: [])
def imp_tab(value_list, justifications=None, map_id=None, show_map_idx=True):
    map_entries = None
    if map_id is not None:
        map_entries = PRINT_TAB_MAP[map_id]

    if map_entries is not None and show_map_idx:
        idx = str(len(map_entries))
        if idx == "0":
            idx = "idx"
        value_list.insert(0, idx)
        if justifications is not None:
            justifications.insert(0, 6)

    value_list = [str(v) for v in value_list]
    if justifications is not None:
        new_list = []
        assert(len(value_list) == len(justifications))
        for idx, value in enumerate(value_list):
            str_value = str(value)
            just = justifications[idx]
            if just > 0:
                new_value = str_value.ljust(just)
            else:
                new_value = str_value.rjust(just)
            new_list.append(new_value)

        value_list = new_list

    line = "\t".join(value_list)
    if map_entries is not None:
        map_entries.append(line)
    print(line)


def segmentar_pulmones(im, plot=False):
    binary = im < -400 # convierto a imagen binaria
    cleared = clear_border(binary) #recorto los bordes
    # Step 3: Label the image.
    label_image = label(cleared) #pongo el label
    areas = [r.area for r in regionprops(label_image)] # me quedo con las dos áreas más grandes
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2) 
    binary = binary_erosion(binary, selem) # erosiono la imagen con un disco radio 2 para separar los nódulos que están junto a los vasos sanguíneos.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem) # operación de cierre con un disco radio 10 para no perder nódulos que están adheridos a la frontera del pulmón.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges) # relleno los huecos que queden dentro de la máscara de los pulmones
    get_high_vals = binary == 0
    im[get_high_vals] = -2000 #aplico la máscara en la imagen
    return im, binary


def preparar_imagen_3d(img, mean_value=None):
    img = img.astype(numpy.float32)
    if mean_value is not None:
        img -= mean_value
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def calcular_distancia(df_row1, df_row2):
    dist = math.sqrt(math.pow(df_row1["coord_x"] - df_row2["coord_x"], 2) + math.pow(df_row1["coord_y"] - df_row2["coord_y"], 2) + math.pow(df_row1["coord_y"] - df_row2["coord_y"], 2))
    return dist


def porcentaje_a_pixeles(x_perc, y_perc, z_perc, cube_image):
    res_x = int(round(x_perc * cube_image.shape[2]))
    res_y = int(round(y_perc * cube_image.shape[1]))
    res_z = int(round(z_perc * cube_image.shape[0]))
    return res_x, res_y, res_z


PATIENT_LIST = None
def obtener_carpeta_paciente(patient_id, resultado_set_neg=False):
    global PATIENT_LIST
    if PATIENT_LIST is None:
        df = pandas.read_csv("res/paso1_etiquetas.csv")
        PATIENT_LIST = df["id"].tolist()

    if resultado_set_neg:
        if patient_id not in PATIENT_LIST:
            return -1

    res = PATIENT_LIST.index(patient_id)
    res %= 6
    return res



