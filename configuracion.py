import os

N_HILOS = 8

TAM_VOXEL_MM = 1.00
NODULO_VALOR_MEDIO_PX = 41
LUNA_SUBSET_START_INDEX = 0
TAM_IMG_SEGMENTADOR = 320

BASE_DIR_SSD = os.path.abspath(os.getcwd()) + "/"
BASE_DIR = os.path.abspath(os.getcwd()) + "/"
EXTRA_DATA_DIR = "res/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "/raw/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "/raw_luna/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "/ext/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "/ext_luna/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "/pred/"

