This project implements a CNN + Transformer feature extractor combined with a DDPG decision-making module for smart agriculture control. To use the code, you need to prepare three dataset folders:
  1.data/imgs/: Tomato images in RGB format;
  2.data/env/: Environmental parameter JSON files corresponding to each image (fields: Temperature, Humidity, LightIntensity, WindForce, Precipitation);
  3.data/actions/: Expert action JSON files corresponding to each image (fields: irrigation, fertilizer, ventilation, light_supplement, optional yield_score).
The three types of files must match one-to-one in order (we recommend consistent naming, e.g., 0001.jpg, 0001.json). If you have min–max statistics for environmental values, place them in data/meta/env_minmax.json; otherwise, default ranges will be used.
Usage:
  python main.py
Training results (model checkpoint, logs, training curves) will be saved in the results/ folder.
Notes:
  1.Make sure the number of images, env files, and action files are the same and correctly aligned;
  2.The code will use GPU if available, otherwise falls back to CPU;
  3.You can adjust parameters and paths in config.py.
