## UCF

This model has been adapted from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

## 

- **Train UCF model**:
   - Use `train_ucf.py`, which will download necessary pretrained `xception` backbone weights from HuggingFace (if not present locally) and start a training job with logging outputs in `.logs/`.
   - Customize the training job by editing `config/ucf.yaml`
     - `pm2 start train_ucf.py --no-autorestart` to train a generalist detector on datasets from `DATASET_META`
     - `pm2 start train_ucf.py --no-autorestart -- --faces_only` to train a face expert detector on preprocessed-face only datasets

- **Miner Neurons**:
   - The `UCF` class in `pretrained_ucf.py` is used by miner neurons to load and perform inference with pretrained UCF model weights.