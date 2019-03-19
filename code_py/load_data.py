import pandas as pd
import json


def colnames_to_lower(df):
    """"
    Function for lower column names
    """
    df.columns = df.columns.str.lower()
    return df


def load_jsons(ids, type):
    """
    Function for loading sentiments from jsons
    :param ids: pet ids
    :param type: train or test set
    :return:
    """

    # Path
    path = 'data/raw/sentiment/' + type + '_sentiment/'

    # Containers
    doc_magnitudes = []
    doc_scores = []

    # Loop over ids, try open document and extract magnitude/score
    for pet_id in ids:
        try:
            with open(path + pet_id + '.json', 'r') as f:
                sentiment = json.load(f)
            doc_magnitudes.append(sentiment['documentSentiment']['magnitude'])
            doc_scores.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            doc_magnitudes.append(-1)
            doc_scores.append(-1)

    # Combine results
    result = pd.DataFrame({'petid': ids,
                           'sentiment_magnitude': doc_magnitudes,
                           'sentiment_score': doc_scores})

    return result


def load_meta(ids, type):
    """
    Function for loading image meta data
    :param ids: pet ids
    :param type: train or test set
    :return:
    """

    # Path
    path = 'data/raw/metadata/' + type + '_metadata/'

    # Containers
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []

    # Loop over ids, try open document and extract info
    for pet_id in ids:
        try:
            with open(path + pet_id + '-1.json', 'r') as f:
                meta = json.load(f)
            vertex_x = meta['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = meta['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = meta['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = meta['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = meta['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = meta['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = meta['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = meta['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = meta['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if meta.get('labelAnnotations'):
                label_description = meta['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = meta['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)

    # Combine results
    result = pd.DataFrame({'petid': ids,
                           'vertex_x': vertex_xs,
                           'vertex_y': vertex_ys,
                           'bounding_confidence': bounding_confidences,
                           'bounding_importance_frac': bounding_importance_fracs,
                           'dominant_blue': dominant_blues,
                           'dominant_green': dominant_greens,
                           'dominant_red': dominant_reds,
                           'dominant_pixel_frac': dominant_pixel_fracs,
                           'dominant_score': dominant_scores,
                           'label_description': label_descriptions,
                           'label_score': label_scores
                           })

    return result




