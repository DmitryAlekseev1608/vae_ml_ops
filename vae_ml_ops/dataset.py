import os

import numpy as np
import pandas as pd
import skimage.io
from skimage.transform import resize


def fetch_dataset(cfg, images_name="data/lfw-deepfunneled", dx=80, dy=80):

    """Функция объявления датасета"""

    dimx = cfg.model.size_img
    dimy = cfg.model.size_img

    # read attrs
    df_attrs = pd.read_csv(
        "data/lfw_attributes.txt",
        sep='\t',
        skiprows=1,
    )
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])

    # read photos
    photo_ids = []
    for dirpath, _, filenames in os.walk(images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_id = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append(
                    {'person': person_id, 'imagenum': photo_number, 'photo_path': fpath}
                )

    photo_ids = pd.DataFrame(photo_ids)
    # print(photo_ids)
    # mass-merge
    # (photos now have same order as attributes)
    df = pd.merge(df_attrs, photo_ids, on=('person', 'imagenum'))

    assert len(df) == len(df_attrs), "lost some data when merging dataframes"

    # print(df.shape)
    # image preprocessing
    all_photos = (
        df['photo_path']
        .apply(skimage.io.imread)
        .apply(lambda img: img[dy:-dy, dx:-dx])
        .apply(lambda img: resize(img, [dimx, dimy]))
    )

    all_photos = np.stack(all_photos.values)  # .astype('uint8')
    all_attrs = df.drop(["photo_path", "person", "imagenum"], axis=1)

    return all_photos, all_attrs
