import cv2
import mediapipe as mp
import json
import numpy as np
import os
import sys
import shutil

# Inicializar BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

Dx_left_s, Dy_left_s, Dx_right_s, Dy_right_s = [], [], [], []
left_rects_s, right_rects_s = [], []
last_step_left_idx, last_step_right_idx = -1, -1
last_step_left_bbox, last_step_right_bbox = None, None
min_displacement = 10
weights = [4, 2, 1]

def step_criteria_advanced(index, Dx, Dy, left):
    global last_step_left_idx, last_step_right_idx, last_step_left_bbox, last_step_right_bbox
    if abs(Dx) < min_displacement and abs(Dy) < min_displacement:
        if left:
            last_step_left_idx = index
            last_step_left_bbox = left_rects_s[index]
        else:
            last_step_right_idx = index
            last_step_right_bbox = right_rects_s[index]
        return True
    elif index > 0:
        if abs(Dy) < min_displacement:
            if orientation_change(index, Dx, left, True):
                if left:
                    last_step_left_idx = index
                    last_step_left_bbox = left_rects_s[index]
                else:
                    last_step_right_idx = index
                    last_step_right_bbox = right_rects_s[index]
                return True
        else:
            if orientation_change(index, Dy, left, False):
                if left:
                    last_step_left_idx = index
                    last_step_left_bbox = left_rects_s[index]
                else:
                    last_step_right_idx = index
                    last_step_right_bbox = right_rects_s[index]
                return True
    return False

def orientation_change(index, value, left, is_x):
    if value == 0:
        return False
    global last_step_left_idx, last_step_right_idx
    d, acc_d = 0, 0
    last_step_idx = last_step_left_idx if left else last_step_right_idx
    crit = 0 if last_step_idx == -1 else last_step_idx

    D = Dx_left_s if left else Dx_right_s
    if not is_x:
        D = Dy_left_s if left else Dy_right_s

    for i in range(index - 1, crit - 1, -1):
        d = D[i]
        acc_d += d
        if d == 0:
            continue
        if (value > 0 and d < 0) or (value < 0 and d > 0):
            if abs(acc_d) > min_displacement:
                return True

    return False

def smooth_displacement(index):
    global Dx_left_s, Dy_left_s, Dx_right_s, Dy_right_s
    if index >= len(Dx_left_s) or index >= len(Dy_left_s) or index >= len(Dx_right_s) or index >= len(Dy_right_s):
        return

    sum_dx_left = weights[0] * Dx_left_s[index]
    sum_dy_left = weights[0] * Dy_left_s[index]
    sum_dx_right = weights[0] * Dx_right_s[index]
    sum_dy_right = weights[0] * Dy_right_s[index]
    norm = weights[0]
    s = len(Dx_left_s)
    for i in range(index + 1, min(s, index + 3)):
        j = i - index
        sum_dx_left += weights[j] * Dx_left_s[i]
        sum_dy_left += weights[j] * Dy_left_s[i]
        sum_dx_right += weights[j] * Dx_right_s[i]
        sum_dy_right += weights[j] * Dy_right_s[i]
        norm += weights[j]
    for i in range(index - 1, max(-1, index - 3), -1):
        j = index - i
        sum_dx_left += weights[j] * Dx_left_s[i]
        sum_dy_left += weights[j] * Dy_left_s[i]
        sum_dx_right += weights[j] * Dx_right_s[i]
        sum_dy_right += weights[j] * Dy_right_s[i]
        norm += weights[j]
    Dx_left_s[index] = sum_dx_left / norm
    Dy_left_s[index] = sum_dy_left / norm
    Dx_right_s[index] = sum_dx_right / norm
    Dy_right_s[index] = sum_dy_right / norm

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def clear_output_directory():
    # Eliminar directorios y archivos previos si existen
    if os.path.exists('./output'):
        shutil.rmtree('./output')

def process_video(video_path, test_mode):
    global Dx_left_s, Dy_left_s, Dx_right_s, Dy_right_s, left_rects_s, right_rects_s
    
    if test_mode:
        clear_output_directory()  # Limpiar la salida antes de generar nuevos archivos
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Hardcode FPS si detecta un valor inusualmente alto o bajo, esto ya que a veces los videos no tienen un valor de FPS correcto, por ejemplo 1000
    if fps > 65 or fps <= 0:  # Valores típicos de FPS
        fps = 30  # Se hardcodea a 30 FPS
        print(f'FPS hardcoded to {fps}')
    print(f'Video resolution: {frame_width}x{frame_height} and {fps} FPS')
    
    frames_info = []

    if test_mode:
        output_directory_frames = './output/frames'
        create_directory(output_directory_frames)
        out = cv2.VideoWriter('./output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    steps = 0
    frame_idx = 0
    left_points = []
    right_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        stepDetection = False
        stepSide = 'None'
        left_position = None
        right_position = None

        if results.pose_landmarks:
            # Puntos del pie izquierdo
            left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Puntos del pie derecho
            right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
            right_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Coordenadas del pie izquierdo
            left_heel_point = (int(left_heel.x * frame_width), int(left_heel.y * frame_height))
            left_foot_index_point = (int(left_foot_index.x * frame_width), int(left_foot_index.y * frame_height))
            left_ankle_point = (int(left_ankle.x * frame_width), int(left_ankle.y * frame_height))
            
            # Coordenadas del pie derecho
            right_heel_point = (int(right_heel.x * frame_width), int(right_heel.y * frame_height))
            right_foot_index_point = (int(right_foot_index.x * frame_width), int(right_foot_index.y * frame_height))
            right_ankle_point = (int(right_ankle.x * frame_width), int(right_ankle.y * frame_height))

            # Calcular el punto central (promedio) de los tres puntos para cada pie
            left_center_point = (
                int((left_heel_point[0] + left_foot_index_point[0] + left_ankle_point[0]) / 3),
                int((left_heel_point[1] + left_foot_index_point[1] + left_ankle_point[1]) / 3)
            )
            right_center_point = (
                int((right_heel_point[0] + right_foot_index_point[0] + right_ankle_point[0]) / 3),
                int((right_heel_point[1] + right_foot_index_point[1] + right_ankle_point[1]) / 3)
            )

            # Dibujar la superficie del pie izquierdo y derecho, se unen los tres puntos del pie (polígono)
            left_foot_color = (255, 0, 255)  # Color morado por defecto
            right_foot_color = (255, 0, 255)  # Color morado por defecto

            if frame_idx > 0:
                dx_left = left_heel_point[0] - left_points[-1][0]
                dy_left = left_heel_point[1] - left_points[-1][1]
                dx_right = right_heel_point[0] - right_points[-1][0]
                dy_right = right_heel_point[1] - right_points[-1][1]

                Dx_left_s.append(dx_left)
                Dy_left_s.append(dy_left)
                Dx_right_s.append(dx_right)
                Dy_right_s.append(dy_right)
                
                left_rects_s.append(left_heel_point)
                right_rects_s.append(right_heel_point)
                
                smooth_displacement(frame_idx)
                
                left_step = step_criteria_advanced(frame_idx - 1, Dx_left_s[-1], Dy_left_s[-1], True)
                right_step = step_criteria_advanced(frame_idx - 1, Dx_right_s[-1], Dy_right_s[-1], False)

                if left_step and right_step:
                    stepSide = 'Both'
                    stepDetection = True
                elif left_step:
                    stepSide = 'Left'
                    stepDetection = True
                elif right_step:
                    stepSide = 'Right'
                    stepDetection = True

                if stepDetection:
                    steps += 1
                    if test_mode:
                        # Cambiar el color de la superficie si se detecta una pisada
                        left_foot_color = (0, 255, 0) if left_step else left_foot_color
                        right_foot_color = (0, 255, 0) if right_step else right_foot_color

            left_points.append(left_heel_point)
            right_points.append(right_heel_point)

            if test_mode:
                left_foot_points = np.array([left_heel_point, left_foot_index_point, left_ankle_point], np.int32)
                cv2.fillPoly(frame, [left_foot_points], color=left_foot_color)  # Color según la detección de pasos

                right_foot_points = np.array([right_heel_point, right_foot_index_point, right_ankle_point], np.int32)
                cv2.fillPoly(frame, [right_foot_points], color=right_foot_color)  # Color según la detección de pasos

                # Dibujar el punto central (promedio) del pie izquierdo y derecho
                cv2.circle(frame, left_center_point, 5, (255, 255, 0), -1)  # Color amarillo
                cv2.circle(frame, right_center_point, 5, (255, 255, 0), -1)  # Color amarillo

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if test_mode:
            if stepDetection:
                cv2.putText(frame, f'Step Detected: {stepSide}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No Step', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Guardar cada frame en un archivo
            frame_filename = os.path.join(output_directory_frames, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)

        frame_info = {
            'frame_index': frame_count,
            'stepDetection': stepDetection,
            'stepSide': stepSide,
            'left_position': {
                'heel': left_heel_point,
                'foot_index': left_foot_index_point,
                'ankle': left_ankle_point,
                'center': left_center_point
            },
            'right_position': {
                'heel': right_heel_point,
                'foot_index': right_foot_index_point,
                'ankle': right_ankle_point,
                'center': right_center_point
            }
        }
        frames_info.append(frame_info)

        if test_mode:
            out.write(frame)
        frame_count += 1
        frame_idx += 1

    cap.release()

    if test_mode:
        out.release()

    # Guardar el JSON con la información de los cuadros procesados
    json_filename = './output/frames_info.json'
    create_directory('./output')  # Asegurar que el directorio exista
    with open(json_filename, 'w') as json_file:
        json.dump(frames_info, json_file, indent=4)

    return frames_info  # Devolver la información de los cuadros procesados

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path> [--test]")
        sys.exit(1)

    video_path = sys.argv[1]
    test_mode = '--test' in sys.argv
    frames_info = process_video(video_path, test_mode)
    print("Finished processing video")

if __name__ == "__main__":
    main()
