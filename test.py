import cv2, numpy as np, os, sys, tensorflow as tf

THRESHOLD = 0.7
video_filename = sys.argv[1]
model_path = os.path.join(os.path.dirname(__file__), 'model')
video_path = os.path.join(os.path.dirname(__file__), 'video')

labels = []
with open(os.path.join(model_path, 'labels.txt'), 'rt') as labels_file:
  for label in labels_file:
    labels.append(label.strip())

interpreter = tf.lite.Interpreter(model_path = os.path.join(model_path, 'model.tflite'))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(os.path.join(video_path, video_filename))

if not cap.isOpened():
  print('Error: Could not open video.')
  exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
  ret, frame = cap.read()

  if not ret:
    break

  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
  image = image.astype(np.float32)

  interpreter.set_tensor(input_details[0]['index'], [image])
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])

  height, width = frame.shape[:2]

  for i in range(len(scores)):
    if scores[i] > THRESHOLD:
      label = labels[classes[i]]
      x_min, y_min, x_max, y_max = boxes[i]
      x_min, x_max = int(x_min * width), int(x_max * width)
      y_min, y_max = int(y_min * height), int(y_max * height)
      
      box_color = (0, 0, 255) if label == 'fuego' else (255, 0, 0)

      cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
      cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

  cv2.imshow('Detections', frame)

  if cv2.waitKey(delay) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
