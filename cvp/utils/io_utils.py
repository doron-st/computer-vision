import pickle


def write_pickle(model, file: str) -> None:
    f = open(file, "wb")
    f.write(pickle.dumps(model))
    f.close()

