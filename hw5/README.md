## Train
```bash
python3 -m salaryreg.train_cli x_data.npy y_data.npy
```
This will create model weights at:

`resources/model.npz`

### Predict
```bash
python3 app.py x_data.npy
```

Prints a list of predicted salaries (RUB) as floats (JSON array) to stdout.