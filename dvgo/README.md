# DVGO

An implementation of [DVGO](https://sunset1995.github.io/dvgo/)
wholly implemented in Python.

## Data

Please download [Synthetic NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) dataset and put in `dvgo/data`.

## Running

to train and run the model,

```sh
python main.py --object [object]
```

where object is the name of one of the objects in the Synthetic-NeRF dataset (e.g. chair, drums).

Do not forget to run python in an environment that has the requirements found in `requirements.txt` installed!