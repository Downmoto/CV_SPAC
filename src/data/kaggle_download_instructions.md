# kaggle dataset download instructions

dataset: andrewmvd/car-plate-detection

1. install kaggle cli:
   - `/Users/arad/Developer/CV_proj/.venv/bin/python -m pip install kaggle`
2. create kaggle api token from your kaggle account settings.
3. place `kaggle.json` at `~/.kaggle/kaggle.json`.
4. set permissions:
   - `chmod 600 ~/.kaggle/kaggle.json`
5. download and unzip into raw data path:
   - `kaggle datasets download -d andrewmvd/car-plate-detection -p data/raw --unzip`
6. expected extracted path:
   - `data/raw/car-plate-detection/images`
   - `data/raw/car-plate-detection/annotations`
