# model4.py – Advanced CNN builder for Rock-Paper-Scissors
# ────────────────────────────────────────────────────────────
# This script provides functions to build and train CNNs.
# - create_model: A "factory" for automated tuning (e.g., GridSearchCV).
# - train: A helper for running single training sessions (e.g., for initial comparisons).

from typing import Tuple, List
from tensorflow.keras import layers, optimizers, initializers, Model, callbacks
from .utils import seed_everything # Assuming seed_everything is in utils.py

def _build_cnn(img_shape: Tuple[int, int, int], *,
               blocks: int,
               filters: List[int],
               dense_units: int,
               dropout_rate: float,
               kernel_size: int,
               use_batch_norm: bool,
               pooling_method: str,
               seed: int) -> Model:
    """Builds a CNN with extensive topological options."""
    if len(filters) != blocks:
        raise ValueError(f"Length of filters list {filters} must equal blocks={blocks}")
    
    init = initializers.GlorotUniform(seed=seed)
    img_input = layers.Input(shape=img_shape)
    x = img_input

    for f in filters:
        if use_batch_norm:
            x = layers.Conv2D(f, kernel_size, padding="same", use_bias=False, kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        else:
            x = layers.Conv2D(f, kernel_size, padding="same", activation="relu", kernel_initializer=init)(x)

        if pooling_method == 'max':
            x = layers.MaxPooling2D(2)(x)
        elif pooling_method == 'average':
            x = layers.AveragePooling2D(2)(x)
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_initializer=init)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, seed=seed)(x)
    
    output = layers.Dense(3, activation="softmax", kernel_initializer=init)(x)
    model = Model(inputs=img_input, outputs=output)
    return model

def create_model(*,
                 blocks: int,
                 filters: List[int],
                 dense_units: int,
                 dropout_rate: float,
                 lr: float,
                 kernel_size: int,
                 use_batch_norm: bool,
                 pooling_method: str,
                 seed: int = 42) -> Model:
    """
    Builds and compiles a CNN model for use with Keras wrappers.
    This is the main "factory" function for automated tuning.
    """
    seed_everything(seed)
    model = _build_cnn((150, 100, 3),
                       blocks=blocks,
                       filters=filters,
                       dense_units=dense_units,
                       dropout_rate=dropout_rate,
                       kernel_size=kernel_size,
                       use_batch_norm=use_batch_norm,
                       pooling_method=pooling_method,
                       seed=seed)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train(train_gen,
          val_gen,
          *,
          blocks: int,
          filters: List[int],
          dense_units: int,
          dropout_rate: float,
          lr: float,
          kernel_size: int,
          use_batch_norm: bool,
          pooling_method: str,
          batch_size: int = 32,
          epochs: int = 25,
          seed: int = 42):
    """
    Helper function for single training runs.
    Used for the initial topology comparison.
    """
    model = create_model(blocks=blocks,
                         filters=filters,
                         dense_units=dense_units,
                         dropout_rate=dropout_rate,
                         lr=lr,
                         kernel_size=kernel_size,
                         use_batch_norm=use_batch_norm,
                         pooling_method=pooling_method,
                         seed=seed)

    # Optional: Add EarlyStopping to speed up runs
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5, # Number of epochs with no improvement before stopping
        restore_best_weights=True
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    
    return history, model