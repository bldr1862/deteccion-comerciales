import os
import cv2
import numpy as np
import config
from sklearn import linear_model

def getMPEGfiles(ADS_PATH):
    MPEGfiles = []
    for root, dirs, files in os.walk(ADS_PATH):
        for fichero in files:
            name, extension = os.path.splitext(fichero)
            if extension == ".mpg":
                MPEGfiles.append(os.path.join(ADS_PATH,fichero))
    return MPEGfiles

class EHD_DESCRIPTOR:
    def __init__(self):
        self.edge_0 = np.matrix([ [1,-1], [1,-1] ]) #0
        self.edge_90 = np.matrix([ [1,1], [-1,-1] ]) #1
        self.edge_135 = np.matrix([ [1.4,0], [0,-1.4] ]) #2
        self.edge_45 = np.matrix([ [0,1.4], [-1.4,0] ]) #3
        self.edge_iso = np.matrix([ [2,-2], [-2,2] ]) #4
        # no edge equal to 5

    def edge_orientation_conv(self, frame, threshold = config.ORIENTATION_THRESHOLD):
        
        conv_0 = cv2.filter2D(frame,-1,self.edge_0) # same shape with original frame
        conv_90 = cv2.filter2D(frame,-1,self.edge_90)
        conv_135 = cv2.filter2D(frame,-1,self.edge_135)
        conv_45 = cv2.filter2D(frame,-1,self.edge_45)
        conv_iso = cv2.filter2D(frame,-1,self.edge_iso)
        position_array = np.array([])

        for row in range(conv_0.shape[0]):
            for col in range(conv_0.shape[1]):
                list_conv = np.array([conv_0[row,col],
                                    conv_90[row,col], 
                                    conv_135[row,col],
                                    conv_45[row,col],
                                    conv_iso[row,col]])
                max_value = np.max(list_conv)
                position = np.where(list_conv == max_value)[0][0]+1

                if max_value < threshold:
                    position = 0
                position_array = np.append(position_array, position)
        position_array = position_array.reshape(conv_0.shape[0], conv_0.shape[1])
        return position_array

    def zonification(self, frame):
        zone_frames = []
        rows = frame.shape[0]
        cols = frame.shape[1]
        # recordar que las dimensiones del frame deben ser diviibles por 4
        if ((rows%4!=0) or (cols%4!=0)):
            return -1

        r_n = int(rows/4)
        c_n = int(cols/4)
        for n_row in range(4):
            for n_col in range(4):
                zone_frames.append(  
                            frame[n_row*r_n: (n_row+1)*r_n , 
                                    n_col*c_n: (n_col+1)*c_n ].reshape(1,-1)  )
        return np.array(zone_frames)

    def zone_histogram(self, zones):
        hist_descriptor = []
        bins = [0,1,2,3,4,5,6]
        for zone in zones:
            hist, bins = np.histogram(zone,bins = bins, density = False)
            hist = hist / zone.shape[0]
            hist_descriptor = np.concatenate((hist_descriptor,hist), axis = 0)
        return hist_descriptor

    def getDescriptors(self, PATH_LIST, title='data.dat', fps_sample = config.SAMPLE_RATE):
        #fps sample is the number of frames per second I am going to analyze
        descriptors = []
        cantidad_frames = []
        for i,PATH in enumerate(PATH_LIST):
            #print("VIDEO NUMBER: {}".format(i+1))
            video_descriptor = []
            video = cv2.VideoCapture(PATH)
            #print(video.get(cv2.CAP_PROP_FPS)) == 30!
            count = 0
            while video.grab():
                frame_number = video.get(1)
                if (frame_number-1) % fps_sample == 0:
                    ret, frame = video.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, config.FRAME_RESIZE)
                    edge_frame = self.edge_orientation_conv(frame)
                    zones = self.zonification(edge_frame)
                    frame_descriptor = self.zone_histogram(zones)
                    video_descriptor.append(frame_descriptor)
                    count = count + 1
            descriptors.append(np.array(video_descriptor))
            cantidad_frames.append(count)
        return np.array(descriptors), np.array(cantidad_frames)


class K_NEIGHBORS:

    def __init__(self, n_neighbors = config.N_NEIGHBORS):
        self.n_neighbors = n_neighbors

    def manhattan_distance(self, array1, array2):
        return np.sum(np.abs(np.subtract(array1,array2)))

    def get_k_nearest_neighbors(self,new_array, reference_array):
        neighbors = []
        distances = []
        dtype = [('video',int), ('frame',int), ('distance',float)]
        n_video = reference_array.shape[0]
        for nv in range(n_video):
            n_frame = reference_array[nv].shape[0]
            for nf in range(n_frame):
                dist = self.manhattan_distance(new_array, reference_array[nv][nf])
                distances.append((nv,nf,dist)) #video, frame, distancia
        distances = np.array(distances, dtype=dtype)
        distances = np.sort(distances, order='distance')
        for k in range(self.n_neighbors):
            neighbors.append(distances[k])
        return np.array(neighbors)

def get_pendiente(tiempo_array,frames_array):
    regr = linear_model.LinearRegression(fit_intercept = True)
    regr.fit(tiempo_array.reshape(-1,1),
            frames_array.reshape(-1,1))
    return regr.coef_[0][0]


def algoritmo_deteccion(tiempo_array, frames_array, videos_array, ads_info,
                    threshold= config.THRESHOLD, _pendiente = config.INTERVALO_PENDIENTE):
    detecciones = []
    video_detectado_anterior = -1

    i=0
    while i < tiempo_array.shape[0]:

        video_actual = videos_array[i]
        frame_actual = frames_array[i]
        frames_video_actual = ads_info[video_actual][3]
        num_frames_actuales = np.sum(videos_array[i:i+frames_video_actual] == video_actual)
        #print(round(float(i*0.3344),2),frames_video_actual, num_videos_actuales, video_actual, frame_actual)
        pendiente = get_pendiente(tiempo_array[i:i+frames_video_actual][videos_array[i:i+frames_video_actual] == video_actual], 
                                frames_array[i:i+frames_video_actual][videos_array[i:i+frames_video_actual] == video_actual])
        pendiente_max_bool = (pendiente < _pendiente[1])
        pendiente_min_bool = (pendiente > _pendiente[0])
        n_frames_bool = (num_frames_actuales/frames_video_actual > threshold)

        if  pendiente_min_bool & pendiente_max_bool & n_frames_bool:
            tiempo_inicio = (i)*0.334
            detecciones.append((tiempo_inicio, video_actual))
            i = i + frames_video_actual - 1
            #print(tiempo_inicio,num_frames_actuales/frames_video_actual, pendiente, video_actual, ads_info[video_actual][1])
        else:
            tiempo_inicio = (i)*0.334
            #print(tiempo_inicio,num_frames_actuales/frames_video_actual, pendiente, video_actual, ads_info[video_actual][1])
        i = i+1
    return detecciones

def ads_information(PATH_LIST, N_FRAMES_ADS):
    ads_info = []
    for i,PATH in enumerate(PATH_LIST):
        video = cv2.VideoCapture(PATH)
        fps = video.get(cv2.CAP_PROP_FPS)
        n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        #numero,nombre,duracion,numero efectivo de frames
        tupla = (i, os.path.basename(PATH), n_frames/fps, N_FRAMES_ADS[i])
        ads_info.append(tupla)
    return ads_info

def save_txt(detecciones, ads_info, tv_video):
    with open('detecciones.txt', 'w') as file:
        for deteccion in detecciones:
            tiempo_inicio = deteccion[0]
            nombre_comercial = ads_info[deteccion[1]][1]
            duracion = ads_info[deteccion[1]][2]
            nombre_tv = os.path.basename(tv_video)
            string = "{}\t{}\t{}\t{}\n".format(nombre_tv,
                                                tiempo_inicio,
                                                duracion,
                                                nombre_comercial)
            #print(string)
            file.write(string)