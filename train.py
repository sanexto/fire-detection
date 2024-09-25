from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials

import os, pandas as pd, random, time

ENDPOINT = os.environ['VISION_TRAINING_ENDPOINT']
training_key = os.environ['VISION_TRAINING_KEY']
prediction_resource_id = os.environ['VISION_PREDICTION_RESOURCE_ID']

project_name = 'Fire Detection'
iteration_name = 'FireDetectionModel'
max_number_images = 5000
max_number_images_batch = 64
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

credentials = ApiKeyCredentials(in_headers = {'Training-key': training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

print('Creando proyecto...')
object_detection_domain = [domain for domain in trainer.get_domains() if domain.type == 'ObjectDetection' and domain.name == 'General (compact) [S1]'][0]
project = trainer.create_project(project_name, domain_id = object_detection_domain.id)

fire_tag = trainer.create_tag(project.id, 'fuego')
smoke_tag = trainer.create_tag(project.id, 'humo')

print('Procesando dataset...')
dataframe = pd.read_csv(os.path.join(dataset_path, 'train', '_annotations.csv'))
dataframe_grouped = dataframe.groupby('filename')
dataframe_groups = [df for _, df in dataframe_grouped]

random.shuffle(dataframe_groups)

print('Agregando imágenes...')
i = 0
tagged_images_with_regions = []

for df in dataframe_groups:
  regions = []

  for row in df.to_numpy():
    filename = row[0]
    width = row[1]
    height = row[2]
    tag = row[3].lower()
    region_left = row[4] / width
    region_top = row[5] / height
    region_width = (row[6] - row[4]) / width
    region_height = (row[7] - row[5]) / height
    tag_id = fire_tag.id if tag == 'fire' else smoke_tag.id

    regions.append(Region(tag_id = tag_id, left = region_left, top = region_top, width = region_width, height = region_height))

  with open(os.path.join(dataset_path, 'train', filename), mode = 'rb') as image_file:
    tagged_images_with_regions.append(ImageFileCreateEntry(name = filename, contents = image_file.read(), regions = regions))

  i += 1

  if i % max_number_images_batch == 0 or i == len(dataframe_groups):
    time.sleep(1)

    upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images = tagged_images_with_regions))

    for image in upload_result.images:
      print(image.source_url, image.status)

    tagged_images_with_regions.clear()

  if i >= max_number_images:
    break

print('Entrenando...')
iteration = trainer.train_project(project.id)

while iteration.status != 'Completed':
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Estado:', iteration.status)
  time.sleep(1)

trainer.publish_iteration(project.id, iteration.id, iteration_name, prediction_resource_id)
print('Listo :)')
