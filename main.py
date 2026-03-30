import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from datetime import datetime
import pandas as pd

# --- модель ---
model = YOLO("yolov8n.pt")

# --- аргументы ---
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
args = parser.parse_args()

# --- видео ---
cap = cv2.VideoCapture(args.video)


events = []

# --- polygon ROI ---
polygon = np.array([
    (5, 606),
    (258, 476),
    (543, 765),
    (101, 998),
    (8, 915),
    (7, 606)
], dtype=np.int32)

# --- параметры стабильности ---
FRAME_HISTORY = 25
ENTER_THRESHOLD = 20
EXIT_THRESHOLD = 35

# История последних N кадров
detection_history = deque(maxlen=FRAME_HISTORY)

# Временные метки
occupancy_start_time = None
last_event_time = None
total_occupied_time = 0
session_count = 0

# Трекинг объектов
tracked_objects = {}  # track_id -> {last_seen, in_roi}
next_track_id = 0

# Состояние
prev_state = "EMPTY"
current_state = "EMPTY"


def calculate_iou(box1, box2):
    """Вычисление IoU для двух bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def update_tracking(current_detections, iou_threshold=0.5):
    """Обновление трекинга объектов на основе IoU"""
    global next_track_id, tracked_objects

    matched = set()
    new_tracked = {}

    # Попытка сопоставить детекции с существующими треками
    for det in current_detections:
        best_match = None
        best_iou = 0

        for track_id, track_data in tracked_objects.items():
            if track_id in matched:
                continue

            iou = calculate_iou(det['bbox'], track_data['bbox'])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = track_id

        if best_match is not None:
            # Обновляем существующий трек
            new_tracked[best_match] = {
                'bbox': det['bbox'],
                'last_seen': time.time(),
                'in_roi': det['in_roi']
            }
            matched.add(best_match)
        else:
            # Создаем новый трек
            new_tracked[next_track_id] = {
                'bbox': det['bbox'],
                'last_seen': time.time(),
                'in_roi': det['in_roi']
            }
            next_track_id += 1

    # Удаляем старые треки (не видели более 1 секунды)
    current_time = time.time()
    for track_id, track_data in tracked_objects.items():
        if track_id not in matched and (current_time - track_data['last_seen']) < 1.0:
            new_tracked[track_id] = track_data

    tracked_objects = new_tracked
    return tracked_objects


def format_time(seconds):
    """Форматирование времени в читаемый формат"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --- основной цикл ---
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

print(f"Начало обработки видео (FPS: {fps:.1f})")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps

    # Запускаем детекцию каждые 2 кадра для производительности
    if frame_count % 2 == 0:
        results = model(frame, verbose=False)

        current_detections = []

        # --- детекция людей ---
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0 and box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # центр bbox
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # проверка попадания в polygon
                    inside = cv2.pointPolygonTest(polygon, (cx, cy), False)

                    current_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'in_roi': inside >= 0
                    })

                    # рисуем bbox человека
                    color = (0, 255, 0) if inside >= 0 else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # рисуем центр
                    cv2.circle(frame, (cx, cy), 4, color, -1)

        # Обновляем трекинг
        tracked_objects = update_tracking(current_detections)

        # Определяем, есть ли кто-то в ROI
        person_in_roi = any(obj['in_roi'] for obj in tracked_objects.values())

        # Добавляем в историю
        detection_history.append(person_in_roi)

        # --- стабилизация на основе истории ---
        # Считаем количество кадров с присутствием в истории
        occupied_frames = sum(detection_history)
        occupied_ratio = occupied_frames / len(detection_history) if detection_history else 0

        # Используем порог для определения состояния
        if occupied_ratio > 0.6:  # >60% кадров с присутствием
            current_state = "OCCUPIED"
        elif occupied_ratio < 0.3:  # <30% кадров с присутствием
            current_state = "EMPTY"
        else:
            current_state = prev_state  # оставляем предыдущее состояние в неопределенных случаях

        # --- события и логирование ---
        if prev_state == "EMPTY" and current_state == "OCCUPIED":
            # Человек зашел
            occupancy_start_time = current_time
            last_event_time = current_time
            session_count += 1
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ПРИСУТСТВИЕ: человек в зоне")
            print(f"  Сессия #{session_count}")
            events.append({
                "event": "APPROACH",
                "time": current_time
            })


        elif prev_state == "OCCUPIED" and current_state == "EMPTY":
            # Человек вышел
            if occupancy_start_time is not None:
                session_duration = current_time - occupancy_start_time
                total_occupied_time += session_duration
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] УХОД: зона свободна")
                print(f"  Длительность сессии: {format_time(session_duration)}")
                print(f"  Общее время занятости: {format_time(total_occupied_time)}")
                print(f"  Всего сессий: {session_count}")
                events.append({
                    "event": "TABLE_FREED",
                    "time": current_time
                })
                if session_duration > 0:
                    print(f"  Среднее время сессии: {format_time(total_occupied_time / session_count)}")
            last_event_time = current_time

        # Отображаем статистику на кадре
        info_text = [
            f"State: {current_state}",
            f"Stability: {occupied_ratio:.0%}",
            f"Tracked objects: {len(tracked_objects)}",
            f"Session #{session_count}",
            f"Current session: {format_time(current_time - occupancy_start_time) if occupancy_start_time and current_state == 'OCCUPIED' else 'N/A'}",
            f"Total occupied: {format_time(total_occupied_time)}"
        ]

        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        prev_state = current_state

    # --- цвет полигона ---
    color = (0, 0, 255) if current_state == "OCCUPIED" else (0, 255, 0)

    # рисуем polygon
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=3)

    # Добавляем прозрачную заливку
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # --- отображение ---
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# --- итоговая статистика ---
df = pd.DataFrame(events)
print(df)
delays = []
last_freed_time = None

for _, row in df.iterrows():
    if row["event"] == "TABLE_FREED":
        last_freed_time = row["time"]

    elif row["event"] == "APPROACH" and last_freed_time is not None:
        delay = row["time"] - last_freed_time
        delays.append(delay)
        last_freed_time = None

if delays:
    avg_delay = sum(delays) / len(delays)
    print(f"\nСреднее время между уходом и подходом: {avg_delay:.2f} сек")

print("\n" + "=" * 60)
print("ИТОГОВАЯ СТАТИСТИКА:")
print(f"Всего сессий: {session_count}")
print(f"Общее время занятости: {format_time(total_occupied_time)}")
if session_count > 0:
    print(f"Среднее время сессии: {format_time(total_occupied_time / session_count)}")
print(f"Общее время видео: {format_time(frame_count / fps)}")
print("=" * 60)

cap.release()
cv2.destroyAllWindows()