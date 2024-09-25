# fire-detection

Python project using [Azure AI Custom Vision](https://www.customvision.ai/) to detect fire and smoke sources.

### Setting up

Add the environment variables, run source ~/.bashrc from your terminal to make the changes effective.

```bash
export VISION_TRAINING_KEY=your-custom-vision-training-key
export VISION_TRAINING_ENDPOINT=your-custom-vision-training-endpoint
export VISION_PREDICTION_RESOURCE_ID=your-custom-vision-resource-id
```

Create a virtual environment for the project by running the following command.

```bash
python3 -m venv env
```

Activate the virtual environment.

```bash
source env/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run train.py script to train the model.

```bash
python3 train.py
```

Run test.py script to test the model with some video from the video folder.

```bash
python3 test.py video_1.mp4
```
