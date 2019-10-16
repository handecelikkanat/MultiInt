import pickle

pathAH=["path/to/pud.1ah.pkl",
"path/to/pud.2ah.pkl",
"path/to/pud.3ah.pkl",
"path/to/pud.4ah.pkl",
"path/to/pud.5ah.pkl",
"path/to/pud.6ah.pkl",
"path/to/pud.8ah.fullModel.pkl"]

for path in pathAH:
    print("loading "+path)
    with open(path, 'rb') as f:
        listAH = pickle.load(f)
        print(len(listAH)) # 1000 -> number of sentences
        print(listAH[0].shape)  # shape of first numpy array, dimensionality = (attention heads, sentence length (bpe), sentence length (bpe))


