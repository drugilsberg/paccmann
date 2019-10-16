"""Tensorflow datasets for paccmann."""
import os
import logging
import tensorflow as tf

logger = logging.getLogger('datasets')

# TFRecords metadata.
DATASETS_METADATA = {
    'smiles_character_tokens': tf.FixedLenFeature((158), tf.int64),
    'smiles_atom_tokens': tf.FixedLenFeature((155), tf.int64),
    'fingerprints_256': tf.FixedLenFeature((256), tf.int64),
    'fingerprints_512': tf.FixedLenFeature((512), tf.int64),
    'targets_10': tf.FixedLenFeature(((10, 1)), tf.float32),
    'targets_20': tf.FixedLenFeature(((20, 1)), tf.float32),
    'targets_50': tf.FixedLenFeature(((50, 1)), tf.float32),
    'selected_genes_10': tf.FixedLenFeature((1121), tf.float32),
    'selected_genes_20': tf.FixedLenFeature((2128), tf.float32),
    'cnv_min': tf.FixedLenFeature((2128), tf.int64),
    'cnv_max': tf.FixedLenFeature((2128), tf.int64),
    'disrupt': tf.FixedLenFeature((2128), tf.int64),
    'zigosity': tf.FixedLenFeature((2128), tf.int64),
    'ic50': tf.FixedLenFeature((), tf.float32),
    'ic50_labels': tf.FixedLenFeature((), tf.int64),
}

CNV_FEATURES = ['cnv_min', 'cnv_max', 'disrupt', 'zigosity']


def record_parser(record, params, feature_names):
    """
    Process a record to create input tensors and labels.

    Process a record with DATASETS_METADATA structure.
    Args:
        - record: a record (<tf.TFRecord>).
        - feature_names: list of features to yield (<list<string>>).
    Returns:
        A dict of tf.Tensor where the keys are
        feature_names
    """
    keys_to_features = {
        feature_name: DATASETS_METADATA[feature_name]
        for feature_name in feature_names
    }
    keys_to_features['ic50'] = DATASETS_METADATA['ic50']
    features = tf.parse_single_example(record, keys_to_features)
    label = features.pop('ic50')
    return features, label


def validate_feature_names(feature_names):
    """
    Validate feature names.

    Validate feature names raising a RuntimeError in case of no valid features
    are found.

    Args:
        - feature_names: list of features to yield (<list<string>>).
    Returns:
        A list of valid feature names.
    """
    feature_names = [
        feature_name for feature_name in feature_names
        if feature_name in DATASETS_METADATA
    ]
    if len(feature_names) < 1:
        message = (
            'No valid feature_names provided!\n'
            'Please provide some from the following list: {}'.format(
                list(DATASETS_METADATA.keys())
            )
        )
        raise RuntimeError(message)
    return feature_names


def generate_dataset(
        filepath, buffer_size=int(256e+6), num_parallel_reads=None
):
    """
    Generate a tf.Dataset given a path.

    Args:
        - filepath: path to a file or a folder containing data (<string>).
        - buffer_size: size of the buffer in bytes (<int>). Defaults to 256MB.
    Returns:
        A tf.Dataset iterator over file/s in .tfrecords format.
    """
    if os.path.isdir(filepath):
        filenames = get_sorted_filelist(filepath)
    else:
        filenames = [filepath]

    logger.debug(
        'Parsing examples from the following files: {}'.format(filenames)
    )

    return tf.data.TFRecordDataset(
        filenames,
        buffer_size=buffer_size,
        num_parallel_reads=num_parallel_reads
    )


def get_sorted_filelist(filepath, pattern='tfrecords'):
    """
    Gets a sorted list of all files with suffix pattern in a given filepath.

    Args:
        - filepath: path to folder to retrieve files (<string>).
        - pattern: suffix specifying file types to retrieve (<string>).
    Returns:
        A list of sorted strings to all matching files in filepath.

    """
    return sorted([
        filename for filename in
        map(lambda entry: os.path.join(filepath, entry), os.listdir(filepath))
        if os.path.isfile(filename) and pattern in filename
    ])


def feature_name_to_placeholder(feature_name, params):
    """
    Get a placeholder for the given feature.
    
    Args:
        - feature_name: name of the feature (<string>).
    Returns:
        Return as a tensor tf.placeholder.
    """
    feature_type, shape = (
        DATASETS_METADATA[feature_name].dtype,
        DATASETS_METADATA[feature_name].shape
    )
    placeholder_shape = [params.get('serving_batch_size', None)]
    if shape is not None:
        placeholder_shape += ([shape] if isinstance(shape, int) else shape)
    return tf.placeholder(feature_type, placeholder_shape)


def assemble_cnv_gep_data(features, genes=[]):
    """
    Takes in transcriptomic and cnv data and assembles it
    into a single matrix.

    Args:
        - features: dictionary with keys `[cnv_min, cnv_max, di, zigosity]` 
            to the cnv features (<dict>).
        - genes: a tensor (<tf.Tensor>) of shape `[batch_size x num_genes]`.

    Returns:
        - gene_cnv_data: (<tf.Tensor>) of shape 
            `[batch_size, num_genes, num_features].`

    """
    data = tf.concat([
        tf.expand_dims(tf.cast(features['cnv_min'], tf.float32), axis=2),
        tf.expand_dims(tf.cast(features['cnv_max'], tf.float32), axis=2),
        tf.expand_dims(tf.cast(features['disrupt'], tf.float32), axis=2),
        tf.expand_dims(tf.cast(features['zigosity'], tf.float32), axis=2),
    ],
                     axis=2)
    data = (
        tf.concat([tf.expand_dims(genes, axis=2), data], axis=2)
        if genes != [] else data
    )
    return data