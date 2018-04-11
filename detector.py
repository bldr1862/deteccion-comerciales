# Source code
# Comerciales - TV
# Direcctorios deben ser ingresados en comillas en la linea de comandos
import sys
import time
import utils
import config
import numpy as np

import matplotlib.pyplot as plt


def main():
    ADS_PATH = sys.argv[1]
    TV_PATH = sys.argv[2]

    if config.TIME_FLAG:
        start_time = time.time()

    # GET DESCRIPTORS

    if not config.READ_DESCRIPTORS:
        #obtener todos los paths de los comerciales
        ADS = utils.getMPEGfiles(ADS_PATH)

        # Inicializar la clase EHD para obtener los descriptores
        # de los comerciales y el video de TV
        EHD = utils.EHD_DESCRIPTOR()
        ADS_DESCRIPTORS, N_FRAMES_ADS = EHD.getDescriptors(ADS,title='ADS_DESCRIPTORS.dat')
        #ADS DESCRIPTOR = VIDEOS x FRAMES x DESCRIPTOR
        TV_DESCRIPTORS, N_FRAMES_TV = EHD.getDescriptors([TV_PATH], title='TV_DESCRIPTORS.dat')
        # obtener numero de video, basename y duracion
        ADS_INFO = utils.ads_information(ADS, N_FRAMES_ADS)


        # Guardar descriptores
        if config.SAVE_DESCRIPTORS:
            np.save('ADS_DESCRIPTORS.npy',ADS_DESCRIPTORS)
            np.save('TV_DESCRIPTORS.npy',TV_DESCRIPTORS)
            np.save('N_FRAMES_ADS.npy', N_FRAMES_ADS)
            np.save('N_FRAMES_TV',N_FRAMES_TV)
            print("SAVED SUCCEED")

    else:
        # Leer descriptores
        ADS = utils.getMPEGfiles(ADS_PATH)
        ADS_DESCRIPTORS = np.load('ADS_DESCRIPTORS.npy')
        TV_DESCRIPTORS = np.load('TV_DESCRIPTORS.npy')
        N_FRAMES_ADS = np.load('N_FRAMES_ADS.npy')
        ADS_INFO = utils.ads_information(ADS, N_FRAMES_ADS)


    # GET NEIGHBORS
    # Obtener los k(3) vecinos mas cercanos para cada frame del video de TV y guardar los resultados
    if not config.READ_DESCRIPTORS:
        k_nearest_neighbors = []
        GET_NEIGHBORS = utils.K_NEIGHBORS()
        for tv_video in TV_DESCRIPTORS:
            for frame_descriptor in tv_video:
                k_nearest_neighbors.append(GET_NEIGHBORS.get_k_nearest_neighbors(frame_descriptor, ADS_DESCRIPTORS))
        
        k_nearest_neighbors = np.array(k_nearest_neighbors)
        if config.SAVE_DESCRIPTORS:
            np.save('vecinos.npy', k_nearest_neighbors)
    else:
        # Leer los resultados
        k_nearest_neighbors = np.load('vecinos.npy')

    # Separar los resultados en distintas list para una manipulacion mas sencilla
    tiempo_array = []
    videos_array = []
    frames_array = []
    for i,element in enumerate(k_nearest_neighbors):
        #print(i,element, ADS[element[0][0]])
        tiempo_array.append(i)
        videos_array.append(element[0][0])
        frames_array.append(element[0][1])

    tiempo_array = np.array(tiempo_array)
    frames_array = np.array(frames_array)
    videos_array = np.array(videos_array)
    #utils.pendiente(tiempo_array, frames_array)

    # Generar las detecciones y guardar los resultados
    DETECCIONES = utils.algoritmo_deteccion(tiempo_array,frames_array,videos_array, ADS_INFO)
    utils.save_txt(DETECCIONES, ADS_INFO, TV_PATH)

    if config.DRAW:
        for i in range(21):
            t = tiempo_array[videos_array == i]
            f = frames_array[videos_array == i]
            if f.shape[0]>0:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(t, f, c='r', marker='o')
                ax.set_xlabel('TIEMPO')
                ax.set_ylabel('FRAMES')
                ax.set_title('COMERCIAL N: {}'.format(i+1))
                plt.show()


    if config.TIME_FLAG:
        print("Total time: {}".format((time.time() - start_time)/60))

if __name__ == '__main__':
    main()