# FLAGS
TIME_FLAG = True

ADS_FLAG = True
TV_FLAG = True

DEVELOP_FLAG = True
SAVE_DESCRIPTORS = True
READ_DESCRIPTORS = True
DRAW = False


# PARAMETERS
SAMPLE_RATE = 10
FRAME_RESIZE = (72,40)
ORIENTATION_THRESHOLD = 20
N_NEIGHBORS = 3
# por el momento solo uso el mejor vecino

#DETECTION PARAMETERS
# La diferencia entre los dos siguientes
# es el margen de error tolerado
WINDOW = 4
INTERVALO_PENDIENTE = (0.88,1.11)
THRESHOLD = 0.6